import torch
from pytorch_lightning.callbacks import Callback

class MatmulPrecisionCallback(Callback):
    def __init__(self, precision: str = "high"):
        super().__init__()
        self.precision = precision

    def on_fit_start(self, trainer, pl_module):
        torch.set_float32_matmul_precision(self.precision)
        print(f"Set float32 matmul precision to: {self.precision}")