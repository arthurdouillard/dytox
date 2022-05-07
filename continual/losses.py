# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Implements the knowledge distillation loss
"""
import torch
from torch import nn
from torch.nn import functional as F


class DistillationLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module,
                 distillation_type: str, alpha: float, tau: float):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        outputs_kd = None
        if not isinstance(outputs, torch.Tensor):
            # assume that the model outputs a tuple of [outputs, outputs_kd]
            outputs, outputs_kd = outputs
        base_loss = self.base_criterion(outputs, labels)
        if self.distillation_type == 'none':
            return base_loss

        if outputs_kd is None:
            raise ValueError("When knowledge distillation is enabled, the model is "
                             "expected to return a Tuple[Tensor, Tensor] with the output of the "
                             "class_token and the dist_token")
        # don't backprop throught the teacher
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        if self.distillation_type == 'soft':
            T = self.tau
            # taken from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
            # with slight modifications
            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / outputs_kd.numel()
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(dim=1))

        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        return loss


def bce_with_logits(x, y):
    return F.binary_cross_entropy_with_logits(
        x,
        torch.eye(x.shape[1])[y].to(y.device)
    )


def soft_bce_with_logits(x, y):
    return F.binary_cross_entropy_with_logits(
        x, y)



def bce_smooth_pos_with_logits(smooth):
    def _func(x, y):
        return F.binary_cross_entropy_with_logits(
            x,
            torch.clamp(
                torch.eye(x.shape[1])[y].to(y.device) - smooth,
                min=0.0
            )
        )
    return _func


def bce_smooth_posneg_with_logits(smooth):
    def _func(x, y):
        return F.binary_cross_entropy_with_logits(
            x,
            torch.clamp(
                torch.eye(x.shape[1])[y].to(y.device) + smooth,
                max=1 - smooth
            )
        )
    return _func


class LabelSmoothingCrossEntropyBoosting(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1, alpha=1, gamma=1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super().__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

        self.alpha = alpha
        self.gamma = gamma

    def forward(self, x, target, boosting_output=None, boosting_focal=None):
        if boosting_output is None:
            return self._base_loss(x, target)
        return self._focal_loss(x, target, boosting_output, boosting_focal)

    def _focal_loss(self, x, target, boosting_output, boosting_focal):
        logprobs = F.log_softmax(x, dim=-1)

        if boosting_focal == 'old':
            pt = boosting_output.softmax(-1)[..., :-1]

            f = torch.ones_like(logprobs)
            f[:, :boosting_output.shape[1] - 1] = self.alpha * (1 - pt) ** self.gamma
            logprobs = f * logprobs
        elif boosting_focal == 'new':
            pt = boosting_output.softmax(-1)[..., -1]
            nb_old_classes = boosting_output.shape[1] - 1

            f = torch.ones_like(logprobs)
            f[:, nb_old_classes:] = self.alpha * (1 - pt[:, None]) ** self.gamma
            logprobs = f * logprobs
        else:
            assert False, (boosting_focal)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

    def _base_loss(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class SoftTargetCrossEntropyBoosting(nn.Module):

    def __init__(self, alpha=1, gamma=1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, x, target, boosting_output=None, boosting_focal=None):
        if boosting_output is None:
            return torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1).mean()

        if boosting_focal == 'old':
            pt = boosting_output.softmax(-1)[..., :-1]

            f = torch.ones_like(x)
            f[:, :boosting_output.shape[1] - 1] = self.alpha * (1 - pt) ** self.gamma
        elif boosting_focal == 'new':
            pt = boosting_output.softmax(-1)[..., -1]

            nb_old_classes = boosting_output.shape[1] - 1

            f = torch.ones_like(x)
            f[:, nb_old_classes:] = self.alpha * (1 - pt[:, None]) ** self.gamma
        else:
            assert False, (boosting_focal)

        return torch.sum(-target * f * F.log_softmax(x, dim=-1), dim=-1).mean()
