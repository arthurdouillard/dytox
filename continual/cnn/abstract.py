from torch import nn

import continual.utils as cutils


class AbstractCNN(nn.Module):
    def reset_classifier(self):
        self.head.reset_parameters()

    def get_internal_losses(self, clf_loss):
        return {}

    def end_finetuning(self):
        pass

    def begin_finetuning(self):
        pass

    def epoch_log(self):
        return {}

    def get_classifier(self):
        return self.head

    def freeze(self, names):
        cutils.freeze_parameters(self, requires_grad=True)
        self.train()

        for name in names:
            if name == 'head':
                cutils.freeze_parameters(self.head)
                self.head.eval()
            elif name == 'backbone':
                for k, p in self.named_parameters():
                    if not k.startswith('head'):
                        cutils.freeze_parameters(p)
            elif name == 'all':
                cutils.freeze_parameters(self)
                self.eval()
            else:
                raise NotImplementedError(f'Unknown module name to freeze {name}')
