"""Batch sampling. The two modes exist so the benchmark can compare them."""
import numpy as np
import torch


class Batches:
    def __init__(self, path, block_size, batch_size, device, mode="gpu", seed=0):
        self.path, self.block_size, self.batch_size = path, block_size, batch_size
        self.device, self.mode = device, mode
        self.n = len(np.memmap(path, dtype=np.uint16, mode="r")) - block_size - 1
        self.generator = torch.Generator(device=device if mode == "gpu" else "cpu").manual_seed(seed)
        if mode == "gpu":
            data = np.fromfile(path, dtype=np.uint16).astype(np.int64)
            self.data = torch.from_numpy(data).to(device)
            self.window = torch.arange(block_size + 1, device=device)

    def get(self):
        if self.mode == "gpu":
            ix = torch.randint(self.n, (self.batch_size, 1), device=self.device, generator=self.generator)
            batch = self.data[ix + self.window]
            return batch[:, :-1], batch[:, 1:]

        # The usual approach: index a memmap per sample in Python, then copy to the GPU.
        data = np.memmap(self.path, dtype=np.uint16, mode="r")
        ix = torch.randint(self.n, (self.batch_size,), generator=self.generator).tolist()
        x = torch.stack([torch.from_numpy(data[i : i + self.block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + self.block_size].astype(np.int64)) for i in ix])
        return x.to(self.device), y.to(self.device)
