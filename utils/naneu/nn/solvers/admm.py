import torch
import torch.nn as nn

class ADMMSolver(nn.Module):
    def __init__(self, A, b, rho=1.0, max_iter=100, tol=1e-4):
        super(ADMMSolver, self).__init__()
        self.A = A
        self.b = b
        self.rho = rho
        self.max_iter = max_iter
        self.tol = tol

        self.x = nn.Parameter(torch.zeros(A.shape[1], 1))
        self.z = nn.Parameter(torch.zeros(A.shape[1], 1))
        self.u = nn.Parameter(torch.zeros(A.shape[1], 1))

    def solve_vec(self, A, b):
        AtA = self.A.T @ self.A
        L = AtA + self.rho * torch.eye(AtA.shape[0])
        L_inv = torch.inverse(L)

        for iter_num in range(self.max_iter):
            z_old = self.z.clone()

            q = self.A.T @ self.b + self.rho * (self.z - self.u)
            self.x.data = L_inv @ q

            self.z.data = self.soft_threshold(self.x + self.u, 1 / self.rho)

            self.u.data += self.x - self.z

            r_norm = torch.norm(self.x - self.z)
            s_norm = torch.norm(-self.rho * (self.z - z_old))

            if r_norm < self.tol and s_norm < self.tol:
                print(f"Converged in {iter_num} iterations!")
                break

        return self.x, self.z

    def solve_op(self, A, At, b):
        for iter_num in range(self.max_iter):
            z_old = self.z.clone()

            Ax_b = self.A(self.x) - self.b
            grad = self.AT(Ax_b) + self.rho * (self.x - self.z + self.u)
            self.x.data -= grad * 0.01

            self.z.data = self.soft_threshold(self.x + self.u, 1 / self.rho)

            self.u.data += self.x - self.z

            r_norm = torch.norm(self.x - self.z) 
            s_norm = torch.norm(-self.rho * (self.z - z_old))

            if r_norm < self.tol and s_norm < self.tol:
                print(f"Converged in {iter_num} iterations!")
                break

        return self.x, self.z

    @staticmethod
    def soft_threshold(x, lam):
        return torch.sign(x) * torch.maximum(torch.abs(x) - lam, torch.zeros_like(x))
