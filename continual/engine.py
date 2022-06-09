# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import json
import os
import math
from typing import Iterable, Optional

import torch
from timm.data import Mixup
from timm.utils import accuracy
from timm.loss import SoftTargetCrossEntropy
from torch.nn import functional as F

import continual.utils as utils
from continual.losses import DistillationLoss
from continual.pod import pod_loss


CE = SoftTargetCrossEntropy()


def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, task_id: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, debug=False, args=None,
                    teacher_model: torch.nn.Module = None,
                    model_without_ddp: torch.nn.Module = None,
                    sam: torch.optim.Optimizer = None,
                    loader_memory=None,
                    pod=None, pod_scales=[1]):
    """Code is a bit ugly to handle SAM, sorry! :upside_down_face:"""
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Task: [{}] Epoch: [{}]'.format(task_id, epoch)
    print_freq = 10

    for batch_index, (samples, targets, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if batch_index == 0:
            print(f'Image size is {samples.shape}.')

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad()

        lam = None
        if mixup_fn is not None:
            samples, targets, lam = mixup_fn(samples, targets)

        if sam is not None and (args.sam_first == 'memory' and task_id > 0):
            # If you want to do the first step of SAM only on memory samples.
            x, y, _ = loader_memory.get()
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=not args.no_amp):
                loss_tuple = forward(x, y, model, teacher_model, criterion, lam, args)
        else:
            with torch.cuda.amp.autocast(enabled=not args.no_amp):
                loss_tuple = forward(samples, targets, model, teacher_model, criterion, lam, args)

        loss = sum(filter(lambda x: x is not None, loss_tuple))
        internal_losses = model_without_ddp.get_internal_losses(loss)
        for internal_loss_value in internal_losses.values():
            loss += internal_loss_value

        if pod is not None and teacher_model is not None:
            if args.pod_scaling:
                nb_classes = sum(model.module.nb_classes_per_task)
                nb_new_classes = model.module.nb_classes_per_task[-1]
                pod_scaling = math.sqrt(nb_new_classes / nb_classes)
            else:
                pod_scaling = 1.0

            loss += pod_scaling * pod * compute_pod(
                model.module.feats, teacher_model.feats, pod_scales)

        check_loss(loss)

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order

        if sam is not None and args.look_sam_k > 0:
            # Look-sam only apply the costly sam estimation every k step.
            look_sam_update = False
            if batch_index % args.look_sam_k == 0:
                loss_scaler.pre_step(loss, optimizer, clip_grad=max_norm,
                        parameters=model.parameters(), create_graph=is_second_order)
                loss_scaler.update()
                sam.first_step()  # modify weights to worse neighbor
                optimizer.zero_grad()

                look_sam_update = True

                with torch.cuda.amp.autocast(enabled=not args.no_amp):
                    loss_tuple = forward(samples, targets, model, teacher_model, criterion, lam, args)
                loss = sum(filter(lambda x: x is not None, loss_tuple))
                internal_losses = model_without_ddp.get_internal_losses(loss)
                for internal_loss_value in internal_losses.values():
                    loss += internal_loss_value

            check_loss(loss)
            loss_scaler.pre_step(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)
            sam.second_step(look_sam_update=look_sam_update)
            loss_scaler.post_step(optimizer, model_without_ddp)
        elif sam is not None:
            loss_scaler.pre_step(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)
            loss_scaler.update()
            sam.first_step()  # modify weights to worse neighbor
            optimizer.zero_grad()

            if args.sam_second == 'memory' and task_id > 0:
                x, y, _ = loader_memory.get()
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=not args.no_amp):
                    loss_tuple = forward(x, y, model, teacher_model, criterion, lam, args)
            else:
                with torch.cuda.amp.autocast(enabled=not args.no_amp):
                    loss_tuple = forward(samples, targets, model, teacher_model, criterion, lam, args)

            loss = sum(filter(lambda x: x is not None, loss_tuple))
            internal_losses = model_without_ddp.get_internal_losses(loss)
            for internal_loss_value in internal_losses.values():
                loss += internal_loss_value

            check_loss(loss)
            loss_scaler.pre_step(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)
            sam.second_step()
            loss_scaler.post_step(optimizer, model_without_ddp)
        else:
            loss_scaler(loss, optimizer, model_without_ddp, clip_grad=max_norm,
                        parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()

        metric_logger.update_dict(internal_losses)
        metric_logger.update(loss=loss_tuple[0])
        metric_logger.update(kd=loss_tuple[1])
        metric_logger.update(div=loss_tuple[2])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if debug:
            print('Debug, only doing one epoch!')
            break

    if hasattr(model_without_ddp, 'hook_after_epoch'):
        model_without_ddp.hook_after_epoch()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def check_loss(loss):
    if not math.isfinite(loss.item()):
        raise Exception('Loss is {}, stopping training'.format(loss.item()))


def forward(samples, targets, model, teacher_model, criterion, lam, args):
    main_output, div_output = None, None

    outputs = model(samples)
    if isinstance(outputs, dict):
        main_output = outputs['logits']
        div_output = outputs['div']
    else:
        main_output = outputs

    loss = criterion(main_output, targets)

    if teacher_model is not None:
        with torch.no_grad():
            main_output_old = None
            teacher_outputs = teacher_model(samples)

        if isinstance(outputs, dict):
            main_output_old = teacher_outputs['logits']
        else:
            main_output_old = teacher_outputs

    kd_loss = None
    if teacher_model is not None:
        logits_for_distil = main_output[:, :main_output_old.shape[1]]

        kd_loss = 0.
        if args.auto_kd:
            # Knowledge distillation on the probabilities
            # I called that 'auto_kd' because the right factor is automatically
            # computed, by interpolation between the main loss and the KD loss.
            # This is strongly inspired by WA (CVPR 2020) --> https://arxiv.org/abs/1911.07053
            lbd = main_output_old.shape[1] / main_output.shape[1]
            loss = (1 - lbd) * loss
            kd_factor = lbd

            tau = args.distillation_tau

            _kd_loss = F.kl_div(
                    F.log_softmax(logits_for_distil / tau, dim=1),
                    F.log_softmax(main_output_old / tau, dim=1),
                    reduction='mean',
                    log_target=True
            ) * (tau ** 2)
            kd_loss += kd_factor * _kd_loss
        elif args.kd > 0.:
            _kd_loss = F.kl_div(
                    F.log_softmax(logits_for_distil / tau, dim=1),
                    F.log_softmax(main_output_old / tau, dim=1),
                    reduction='mean',
                    log_target=True
            ) * (tau ** 2)
            kd_loss += args.kd * _kd_loss


    div_loss = None
    if div_output is not None:
        # For the divergence heads, we need to create new targets.
        # If a class belong to old tasks, it will be 0.
        # If a class belong to the new task, it will be a class id between
        # 1 (not 0!) and 'nb_class_in_new_task'.
        # When doing that with mixup, some trickery is needed. (see if lam is not None).
        nb_classes = main_output.shape[1]
        nb_new_classes = div_output.shape[1] - 1
        nb_old_classes = nb_classes - nb_new_classes

        if lam is not None:  # 'lam' is the interpolation Lambda of mixup
            # If using mixup / cutmix
            div_targets = torch.zeros_like(div_output)
            nb_classes = main_output.shape[1]
            nb_new_classes = div_output.shape[1] - 1
            nb_old_classes = nb_classes - nb_new_classes

            div_targets[:, 0] = targets[:, :nb_old_classes].sum(-1)
            div_targets[:, 1:] = targets[:, nb_old_classes:]
        else:
            div_targets = torch.clone(targets)
            mask_old_cls = div_targets < nb_old_classes
            mask_new_cls = ~mask_old_cls

            div_targets[mask_old_cls] = 0
            div_targets[mask_new_cls] -= nb_old_classes - 1

        div_loss = args.head_div * criterion(div_output, div_targets)

    return loss, kd_loss, div_loss


def compute_pod(feats, old_feats, scales):
    if len(feats[0].shape) == 3:
        # transformer archi and not cnn
        bs, nb_tokens, dim = feats[0].shape
        w = int(math.sqrt(nb_tokens))
        feats = [f.view(bs, w, w, dim) for f in feats]
        old_feats = [f.view(bs, w, w, dim) for f in old_feats]

    return pod_loss(feats, old_feats, scales)


@torch.no_grad()
def evaluate(data_loader, model, device, logger):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target, task_ids in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            if isinstance(output, dict):
                output = output['logits']
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, min(5, output.shape[1])))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

        logger.add([output.cpu().argmax(dim=1), target.cpu(), task_ids], subset='test')

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f}  loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def eval_and_log(args, output_dir, model, model_without_ddp, optimizer, lr_scheduler,
                 epoch, task_id, loss_scaler, max_accuracy, accuracy_list,
                 n_parameters, device, data_loader_val, train_stats, log_store, log_path, logger,
                 model_log, skipped_task=False):
    if args.output_dir:
        if os.path.isdir(args.resume):
            checkpoint_paths = [os.path.join(args.resume, f'checkpoint_{task_id}.pth')]
        else:
            checkpoint_paths = [output_dir / f'checkpoint_{task_id}.pth']
        for checkpoint_path in checkpoint_paths:
            if skipped_task:
                continue

            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'task_id': task_id,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }, checkpoint_path)

    test_stats = evaluate(data_loader_val, model, device, logger)
    print(f"Accuracy of the network on the {len(data_loader_val.dataset)} test images: {test_stats['acc1']:.1f}%")
    max_accuracy = max(max_accuracy, test_stats["acc1"])
    print(f'Max accuracy: {max_accuracy:.2f}%')
    accuracy_list.append(test_stats['acc1'])

    log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters}

    mean_acc5 = -1.0
    if log_store is not None:
        log_store['results'][task_id] = log_stats
        all_acc5 = [task_log['test_acc5'] for task_log in log_store['results'].values()]
        mean_acc5 = sum(all_acc5) / len(all_acc5)

    if log_path is not None and utils.is_main_process():
        with open(log_path, 'a+') as f:
            f.write(json.dumps({
                'task': task_id,
                'epoch': epoch,
                'acc': round(100 * logger.accuracy, 2),
                'avg_acc': round(100 * logger.average_incremental_accuracy, 2),
                'forgetting': round(100 * logger.forgetting, 6),
                'acc_per_task': [round(100 * acc_t, 2) for acc_t in logger.accuracy_per_task],
                'train_lr': log_stats.get('train_lr', 0.),
                'bwt': round(100 * logger.backward_transfer, 2),
                'fwt': round(100 * logger.forward_transfer, 2),
                'test_acc1': round(log_stats['test_acc1'], 2),
                'test_acc5': round(log_stats['test_acc5'], 2),
                'mean_acc5': round(mean_acc5, 2),
                'train_loss': round(log_stats.get('train_loss', 0.), 5),
                'test_loss': round(log_stats['test_loss'], 5),
                **model_log
            }) + '\n')
    if args.output_dir and utils.is_main_process():
        with (output_dir / "log.txt").open("a") as f:
            f.write(json.dumps(log_stats) + "\n")

    return max_accuracy


def indexes_task_outputs(logits, targets, increment_per_task):
    if increment_per_task[0] != increment_per_task[1]:
        raise NotImplementedError(f'Not supported yet for non equal task size')

    inc = increment_per_task[0]
    indexes = torch.zeros(len(logits), inc).long()

    for r in range(indexes.shape[0]):
        for c in range(indexes.shape[1]):
            indexes[r, c] = (targets[r] // inc) * inc + r * logits.shape[1] + c

    indexed_logits = logits.view(-1)[indexes.view(-1)].view(len(logits), inc)
    indexed_targets = targets % inc

    return indexed_logits, indexed_targets
