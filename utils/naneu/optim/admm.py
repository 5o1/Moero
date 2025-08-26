import torch
from torch.optim.optimizer import Optimizer

class ADMMOptimizer(Optimizer):
    """
    Custom PyTorch optimizer implementing the ADMM algorithm.

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): Learning rate.
        rho (float): Penalty parameter for the ADMM updates.
        constraint (callable): A function representing the constraint, e.g., a projection function.
                              It should project a tensor to satisfy the constraints.
    """
    def __init__(self, params, lr=1e-3, rho=1.0, constraint=None):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if rho <= 0.0:
            raise ValueError(f"Invalid rho value: {rho}")
        if constraint is None or not callable(constraint):
            raise ValueError("A valid constraint projection function must be provided.")

        defaults = dict(lr=lr, rho=rho, constraint=constraint)
        super(ADMMOptimizer, self).__init__(params, defaults)

        # Initialize the dual variables (u) and auxiliary variables (z)
        self.state = {}
        for group in self.param_groups:
            for p in group['params']:
                self.state[p] = {
                    'z': torch.zeros_like(p.data),  # Auxiliary variable
                    'u': torch.zeros_like(p.data),  # Dual variable
                }

    def step(self, closure=None):
        """
        Perform a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            rho = group['rho']
            constraint = group['constraint']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]
                z = state['z']
                u = state['u']

                # Step 1: Update primal variable (p.data)
                p.data = p.data - lr * (grad + rho * (p.data - z + u))

                # Step 2: Update auxiliary variable (z)
                z_new = constraint(p.data + u)
                state['z'] = z_new

                # Step 3: Update dual variable (u)
                state['u'] = u + (p.data - z_new)

        return loss