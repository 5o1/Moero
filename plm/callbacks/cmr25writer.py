from pytorch_lightning.callbacks import BasePredictionWriter    
from collections import defaultdict
from typing import Dict, Any, Literal, Optional, Sequence
import os
from os import PathLike
import torch
from queue import Queue, Full
from concurrent.futures import ThreadPoolExecutor
import atexit
from data.constructor import GridConstructor
import pytorch_lightning as pl
# from data.cmrsample import CmrSliceSample, CmrVolumeSample

class AsyncTensorWriterPool:
    """
    A thread-safe utility for asynchronously saving PyTorch tensors to disk.

    This class uses a thread pool to handle tensor-saving tasks in the background, 
    allowing the main program to continue executing without waiting for disk I/O.
    """
    def __init__(self, output_dir="predictions", num_workers=4, max_tasks=None, timeout=300):
        self.output_dir = output_dir
        self.num_workers = num_workers
        self.max_tasks = max_tasks if max_tasks is not None else num_workers
        self.timeout = timeout

        if not 0 < self.num_workers <= 32:
            raise ValueError("Security check: num_workers must be between 1 and 32.")

        self.tasks = Queue(maxsize=self.max_tasks)
        self.executor = ThreadPoolExecutor(max_workers=self.num_workers)
        self._shutdown_called = False
        
        atexit.register(self.shutdown)

    def save(self, tensor: torch.Tensor, filename: str):
        def save_image():
            try:
                save_path, tensor = self.tasks.get()
                torch.save(tensor, save_path)
            except Exception as e:
                print(f"Error saving tensor: {e}")

        save_path = os.path.join(self.output_dir, filename)
        try:
            self.tasks.put((save_path, tensor), timeout=self.timeout)
            self.executor.submit(save_image)
        except Full:
            print(f"Task queue is full. Failed to save {filename}.")

    def __del__(self):
        self.shutdown()

    def shutdown(self):
        if not self._shutdown_called:
            self._shutdown_called = True
            while not self.tasks.empty():
                save_path, tensor = self.tasks.get()
                try:
                    torch.save(tensor, save_path)
                except Exception as e:
                    print(f"Error saving tensor during shutdown: {e}")
            self.executor.shutdown(wait=True)

class CmrWriter(BasePredictionWriter):
    def __init__(self, output_dir: PathLike | str, save_zf: bool = False, save_csm: bool = False, save_target: bool = False):
        super().__init__('batch')
        self.output_dir = output_dir
        self.save_zf = save_zf
        self.save_csm = save_csm
        self.save_target = save_target

        self.pred_constructor_dict: Dict[str, GridConstructor] = defaultdict(GridConstructor)
        if save_zf:
            self.zf_constructor_dict: Dict[str, GridConstructor] = defaultdict(GridConstructor)
        if save_csm:
            self.csm_constructor_dict: Dict[str, GridConstructor] = defaultdict(GridConstructor)
        if save_target:
            self.target_constructor_dict: Dict[str, GridConstructor] = defaultdict(GridConstructor)
        
        self.debug_constructor_dict: Dict[str, Dict[str, GridConstructor]] = defaultdict(lambda: defaultdict(GridConstructor))

        self.writer_pool = AsyncTensorWriterPool(self.output_dir, num_workers=4, max_tasks=8)

    def construct(self, constructor_dict: dict, fname: PathLike | str, seqidx : tuple, seqshape : tuple, pred : torch.Tensor):
        if fname not in constructor_dict:
            constructor_dict[fname] = GridConstructor(seqshape, torch.Tensor, torch.Tensor)
            constructor_dict[fname](pred, seqidx)
        else:
            constructor_dict[fname](pred, seqidx)
        
        if constructor_dict[fname].is_completed():
            os.makedirs(self.output_dir, exist_ok=True)
            tensor = torch.as_tensor(constructor_dict[fname].construct())
            if tensor is None:
                raise ValueError(f"Constructor for {fname} has no tensor to save.")
            save_fname = str(fname).split('/')[-1].replace('.h5', '.pt')
            self.writer_pool.save(tensor, save_fname)

            # Clear buffer
            del constructor_dict[fname]

    def write_on_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        prediction: Any,
        batch_indices: Optional[Sequence[int]],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:

        fname = prediction['fname']
        seqshape = prediction['seqshape']
        seqidx = prediction['seqidx']
        pred = prediction['img_pred']
        debug_terms = prediction['debug_terms']

        if isinstance(fname, list):
            for i, f in enumerate(fname):
                self.construct(self.pred_constructor_dict, f, tuple(seqidx[i]), tuple(seqshape[i]), pred[i])
                if self.save_zf:
                    self.construct(self.zf_constructor_dict, f"{f}.zf", tuple(seqidx[i]), tuple(seqshape[i]), prediction['img_zf'][i])
                if self.save_csm:
                    self.construct(self.csm_constructor_dict, f"{f}.csm", tuple(seqidx[i]), tuple(seqshape[i]), prediction['csm'][i])
                if self.save_target:
                    self.construct(self.target_constructor_dict, f"{f}.target", tuple(seqidx[i]), tuple(seqshape[i]), batch['target'][i])

                for debug_name, debug_tensor in debug_terms.items():
                    self.construct(
                        self.debug_constructor_dict[debug_name],
                        f"{f}.debug.{debug_name}",
                        tuple(seqidx[i]),
                        tuple(seqshape[i]),
                        debug_tensor[i]
                    )
        else:
            self.construct(self.pred_constructor_dict, fname, tuple(seqidx), tuple(seqshape), pred)
            if self.save_zf:
                self.construct(self.zf_constructor_dict, fname + ".zf", tuple(seqidx), tuple(seqshape), prediction['img_zf'])
            if self.save_csm:
                self.construct(self.csm_constructor_dict, fname + ".csm", tuple(seqidx), tuple(seqshape), prediction['csm'])
            if self.save_target:
                self.construct(self.target_constructor_dict, fname + ".target", tuple(seqidx), tuple(seqshape), batch['target'])

            for debug_name, debug_tensor in debug_terms.items():
                self.construct(
                    self.debug_constructor_dict[debug_name],
                    f"{fname}.debug.{debug_name}",
                    tuple(seqidx),
                    tuple(seqshape),
                    debug_tensor
                )
