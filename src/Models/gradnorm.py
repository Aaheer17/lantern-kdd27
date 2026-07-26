"""
GradNorm balancer — self-contained, no external dependencies beyond PyTorch.
Chen et al., "GradNorm: Gradient Normalization for Adaptive Loss Balancing
in Deep Multitask Networks", ICML 2018.  arXiv:1711.02257

Note: create_graph=True is incompatible with PyTorch efficient attention.
log_w is updated via manual gradient (chain rule on softmax jacobian).

L0 is estimated from the mean of the first 100 batches rather than a single
noisy first batch, to avoid extreme r values driving weight collapse.
During the first 100 batches a plain equal weighted sum is used as fallback.
"""
import torch
import torch.nn as nn


class GradNormBalancer(nn.Module):

    def __init__(self, num_tasks: int, alpha: float = 1.5,
                 lr: float = 1e-3, device: str = 'cuda'):
        super().__init__()
        self.log_w     = nn.Parameter(torch.zeros(num_tasks, device=device))
        self.w_optim   = torch.optim.Adam([self.log_w], lr=lr)
        self.alpha     = alpha
        self.num_tasks = num_tasks
        self.L0        = None
        self.device    = device

    @property
    def weights(self) -> torch.Tensor:
        w = torch.softmax(self.log_w, dim=0) * self.num_tasks
        w = torch.clamp(w, min=0.5)
        w = w / w.sum() * self.num_tasks  # renormalize so sum stays = num_tasks
        return w
    def step(
        self,
        losses:        list,
        objectives:    list,
        shared_params: list,
        all_params:    list | None = None,
    ) -> tuple[torch.Tensor, dict, list]:

        # ── L0 warmup: accumulate first 100 batches before locking ───────
        if self.L0 is None:
            if not hasattr(self, '_L0_accum'):
                self._L0_accum = [0.0] * self.num_tasks
                self._L0_count = 0
            self._L0_count += 1
            for i, l in enumerate(losses):
                self._L0_accum[i] += l.detach().item()
            if self._L0_count >= 100:
                self.L0 = torch.tensor(
                    [v / self._L0_count for v in self._L0_accum],
                    device=self.device
                )
            # warmup fallback: equal weighted sum, no GradNorm update yet
            L_total = sum(losses[i] for i in range(self.num_tasks)) / self.num_tasks
            info = {"gradnorm_L_grad": 0.0}
            for i, name in enumerate(objectives):
                info[f"gradnorm_w_{name}"]       = 1.0
                info[f"gradnorm_G_{name}"]       = 0.0
                info[f"gradnorm_Gtarget_{name}"] = 0.0
                info[f"gradnorm_r_{name}"]       = 0.0
            return L_total, info, []

        # ── Normal GradNorm step (after L0 is locked) ────────────────────
        w        = self.weights                      # (T,) connected to log_w
        w_detach = w.detach()
        L_total  = sum(w_detach[i] * losses[i] for i in range(self.num_tasks))

        # ── Per-task gradient norms at shared layer ───────────────────────
        # create_graph=False: efficient attention does not support higher-order
        # gradients. log_w is updated via manual gradient (chain rule).
        G = []
        for i in range(self.num_tasks):
            grads_shared = torch.autograd.grad(
                w_detach[i] * losses[i],
                shared_params,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )
            grads_shared = [g for g in grads_shared if g is not None]
            if grads_shared:
                G.append(torch.norm(
                    torch.cat([g.flatten() for g in grads_shared])
                ).detach())
            else:
                G.append(torch.tensor(0.0, device=self.device))

        G_stack = torch.stack(G)                     # (T,) all detached
        G_bar   = G_stack.mean()

        # ── Relative inverse training rates ──────────────────────────────
        L_hat = torch.stack([
            losses[i].detach() / (self.L0[i] + 1e-8)
            for i in range(self.num_tasks)
        ])
        r = L_hat / (L_hat.mean() + 1e-8)

        # ── Manual gradient update on log_w ──────────────────────────────
        # L_grad = sum_i |G_i - G_target_i|
        # dL_grad/d(log_w_j) via chain rule through softmax:
        #   = num_tasks * w_j * (sign_r_j - sum_i sign_r_i * w_i)
        G_target   = (G_bar * r ** self.alpha).detach()
        residual   = G_stack - G_target
        # sign_r     = torch.sign(residual)

        # w_s        = torch.softmax(self.log_w.detach(), dim=0)
        # dot        = (sign_r * w_s).sum()
        # grad_log_w = self.num_tasks * w_s * (sign_r - dot)


        # Correct implementation:
        dL_dw      = torch.sign(residual) * G_stack / (w_detach + 1e-8)  # ∂L_grad/∂w_i
        w_s        = torch.softmax(self.log_w.detach(), dim=0)
        dot        = (dL_dw * w_s).sum()
        grad_log_w = w_s * (dL_dw - dot)          # softmax Jacobian chain rule

        self.w_optim.zero_grad()
        self.log_w.grad = grad_log_w
        self.w_optim.step()

        L_grad = residual.abs().sum()                # scalar for logging only

        # ── Full-network gradient vectors per task (for diagnostics) ──────
        full_grads = []
        if all_params is not None:
            for i in range(self.num_tasks):
                retain = (i < self.num_tasks - 1)
                grads_full = torch.autograd.grad(
                    losses[i],
                    all_params,
                    retain_graph=retain,
                    create_graph=False,
                    allow_unused=True,
                )
                flat = torch.cat([
                    g.flatten() for g in grads_full if g is not None
                ])
                full_grads.append(flat.detach())

        # ── Logging info ──────────────────────────────────────────────────
        w_final = self.weights.detach()
        info = {"gradnorm_L_grad": float(L_grad.item())}
        for i, name in enumerate(objectives):
            info[f"gradnorm_w_{name}"]       = float(w_final[i].item())
            info[f"gradnorm_G_{name}"]       = float(G_stack[i].item())
            info[f"gradnorm_Gtarget_{name}"] = float(G_target[i].item())
            info[f"gradnorm_r_{name}"]       = float(r[i].item())

        return L_total, info, full_grads