import torch


class SAM:
    """SAM, ASAM, and Look-SAM

    Modified version of: https://github.com/davda54/sam
    Only Look-SAM has been added.

    It speeds up SAM quite a lot but the alpha needs to be tuned to reach same performance.
    """
    def __init__(self, base_optimizer, model_without_ddp, rho=0.05, adaptive=False, div='', use_look_sam=False, look_sam_alpha=0., **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)

        self.base_optimizer =  base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.model_without_ddp = model_without_ddp

        self.rho = rho
        self.adaptive = adaptive
        self.div = div
        self.look_sam_alpha = look_sam_alpha
        self.use_look_sam = use_look_sam

        self.g_v = dict()

    @torch.no_grad()
    def first_step(self):
        self.e_w = dict()
        self.g = dict()

        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = self.rho / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                e_w = (torch.pow(p, 2) if self.adaptive else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.e_w[p] = e_w
                self.g[p] = p.grad.clone()

    @torch.no_grad()
    def second_step(self, look_sam_update=False):
        if self.use_look_sam and look_sam_update:
            self.g_v = dict()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue

                if not self.use_look_sam or look_sam_update:
                    p.sub_(self.e_w[p])

                if self.use_look_sam and look_sam_update:
                    cos = self._cos(self.g[p], p.grad)
                    norm_gs = p.grad.norm(p=2)
                    norm_g = self.g[p].norm(p=2)
                    self.g_v[p] = p.grad - norm_gs * cos * self.g[p] / norm_g
                elif self.use_look_sam:
                    norm_g = p.grad.norm(p=2)
                    norm_gv = self.g_v[p].norm(p=2)
                    p.grad.add_(self.look_sam_alpha * (norm_g / norm_gv) * self.g_v[p])

        self.e_w = None
        self.g = None

    def _cos(self, a, b):
        return torch.dot(a.view(-1), b.view(-1)) / (a.norm() * b.norm())

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if self.adaptive else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm
