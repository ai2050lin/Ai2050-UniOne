
import torch
import torch.nn as nn
import torch.optim as optim


class RicciFlowOptimizer(optim.Optimizer):
    """
    Ricci Flow Optimizer: Adjusts weights based on the curvature of the feature manifold.
    
    Ricci Flow Equation: d(g_ij)/dt = -2 * R_ij
    In ML context, we interpret weights as the metric 'g'.
    The 'curvature' R_ij can be approximated by the variance/covariance of the activations.
    
    This optimizer acts as a geometric regularizer, 'smoothing' the manifold by 
    penalizing regions of extreme curvature (high-variance directions that don't contribute to logic).
    """
    def __init__(self, params, lr=1e-3, ricci_alpha=0.01):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, ricci_alpha=ricci_alpha)
        super(RicciFlowOptimizer, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            ricci_alpha = group['ricci_alpha']
            lr = group['lr']
            
            # Implementation of Dynamic Alpha Decay
            # Alpha decays over steps to allow initial exploration and late-stage geometric solidification.
            state = self.state.get('global_step', 0)
            alpha_decay = 0.999 # Smooth decay
            effective_alpha = ricci_alpha * (alpha_decay ** state)
            self.state['global_step'] = state + 1

            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Standard Gradient Descent step
                d_p = p.grad
                
                # Ricci Flow Correction:
                if p.dim() >= 2:
                    original_shape = p.shape
                    w = p.view(-1, p.size(-1)) 
                    m, n = w.shape
                    
                    if m > n:
                        w_sq = torch.matmul(w.t(), w)
                        correction = torch.matmul(w, w_sq) - w
                    else:
                        w_sq = torch.matmul(w, w.t())
                        correction = torch.matmul(w_sq, w) - w
                    
                    # Update with Ricci term
                    p.add_(d_p, alpha=-lr)
                    p.add_(correction.view(original_shape), alpha=-effective_alpha * lr)
                else:
                    # For biases/vectors, just standard grad
                    p.add_(d_p, alpha=-lr)

        return loss
