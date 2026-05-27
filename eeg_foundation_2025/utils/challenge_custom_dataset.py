# Codes written by Liuyin Yang (liuyin.yang@kuleuven.be)
# All rights reserved.
# --------------------------------------------------------

import torch
from torch.utils.data import Dataset, get_worker_info
import h5py
import numpy as np
import random
import os


def _decode_attr(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        return _decode_attr(value.item())
    return value


def _infer_sample_rate_hz(window_len_samples, window_len_sec):
    if window_len_sec and window_len_sec > 0:
        return float(window_len_samples) / float(window_len_sec)
    return 100.0

class EEGTwoChallengeDataset(Dataset):
    """
    If both datasets are given: concatenates them, returning (eeg, target, ch_idx).
    If one is None: behaves like that single dataset but still returns ch_idx in {0,1}.
    """
    def __init__(self, ds_ch1: Dataset | None, ds_ch2: Dataset | None):
        if ds_ch1 is None and ds_ch2 is None:
            raise ValueError("At least one of ds_ch1 or ds_ch2 must be provided.")
        self.ds0 = ds_ch1  # challenge 1 (index 0)
        self.ds1 = ds_ch2  # challenge 2 (index 1)

        self.len0 = len(ds_ch1) if ds_ch1 is not None else 0
        self.len1 = len(ds_ch2) if ds_ch2 is not None else 0
        self.cum  = [self.len0, self.len0 + self.len1]
        ref_ds = ds_ch1 if ds_ch1 is not None else ds_ch2
        self.n_channels = getattr(ref_ds, "n_channels", None)
        self.window_len_samples = getattr(ref_ds, "window_len_samples", None)
        self.window_len_sec = getattr(ref_ds, "window_len_sec", None)
        self.sample_rate_hz = getattr(ref_ds, "sample_rate_hz", None)
        self.model_sample_rate_hz = getattr(ref_ds, "model_sample_rate_hz", None)
        self.apply_vit_resample = getattr(ref_ds, "apply_vit_resample", None)

    def __len__(self):
        return self.cum[-1]

    def __getitem__(self, idx):
        if idx < self.len0:
            # from challenge 1
            eeg, target = self.ds0[idx]  # ds0 must exist if len0>0
            ch = 0
        else:
            # from challenge 2
            local = idx - self.len0
            eeg, target = self.ds1[local]  # ds1 must exist if len1>0
            ch = 1

        ch_idx = torch.tensor(ch, dtype=torch.long)
        return eeg, target, ch_idx

    # Optional helpers (useful for samplers; avoid accessing .ds[0]/.ds[1] externally)
    def len_ch0(self): return self.len0
    def len_ch1(self): return self.len1
    
class EEGH5Dataset(Dataset):
    def __init__(self, h5_path, split="train"):
        """
        Args:
            h5_path (str): Path to the .h5 file
            split (str): One of "train", "valid", or "test"
        """
        self.h5_path = h5_path
        self.split = split
        self.h5_file = None  # Will be opened lazily

        # Read dataset length once without loading data
        with h5py.File(self.h5_path, "r") as f:
            data_ds = f[f"{split}/data"]
            self.length = f[f"{split}/targets"].shape[0]
            self.n_channels = int(data_ds.shape[1])
            self.window_len_samples = int(data_ds.shape[2])
            self.window_len_sec = float(_decode_attr(f.attrs.get("window_len_sec", 2.0)))
            self.sample_rate_hz = float(
                _decode_attr(
                    f.attrs.get(
                        "sample_rate_hz",
                        _infer_sample_rate_hz(self.window_len_samples, self.window_len_sec),
                    )
                )
            )
            self.model_sample_rate_hz = float(_decode_attr(f.attrs.get("model_sample_rate_hz", 128.0)))
            self.apply_vit_resample = bool(
                _decode_attr(
                    f.attrs.get(
                        "requires_vit_resample",
                        abs(self.sample_rate_hz - self.model_sample_rate_hz) > 1e-6,
                    )
                )
            )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.h5_file is None:
            # Lazy open file in worker (each worker gets its own handler)
            self.h5_file = h5py.File(self.h5_path, "r")

        X = self.h5_file[f"{self.split}/data"][idx]     # shape (C, T)
        y = self.h5_file[f"{self.split}/targets"][idx]  # scalar

        # Optionally convert to torch tensors here
        X = torch.from_numpy(X).float()
        y = torch.tensor(y).float()  # use .long() for classification

        return X, y

    

class EEGH5CropDataset(Dataset):
    """
    Efficient HDF5 reader for (N, C, T) EEG:
    - Per-worker h5 handle; no SWMR by default (fastest in your env).
    - Per-worker RNG seeding for random crop positions.
    - Optional big raw-data chunk cache (rdcc_*), harmless if file is contiguous.
    - Optional read_direct() with a per-worker buffer to avoid extra copies.

    Expect best results if your H5 is **contiguous** (chunks=None, compression=None),
    or per-sample chunked (1, C, T). Copy the H5 to node-local SSD ($TMPDIR) if possible.
    """
    def __init__(self,
                 h5_path: str,
                 split: str = "train",
                 crop_size: int = 200,
                 resample_to_length: int | None = None,
                 seed: int = 2025,
                 sample_rate_hz: float = 100.0,
                 target_sample_rate_hz: float | None = None,
                 model_sample_rate_hz: float = 128.0,
                 use_swmr: bool = False,
                 rdcc_nbytes: int = 512 * 1024 * 1024,   # 512 MiB cache
                 rdcc_nslots: int = 1_000_003,
                 rdcc_w0: float = 0.75,
                 locking: bool = False,
                 use_read_direct: bool = False):
        self.h5_path = h5_path
        self.split = split
        self.crop_size = int(crop_size)
        self.resample_to_length = int(resample_to_length) if resample_to_length is not None else None
        self.base_seed = int(seed)
        self.rng = random.Random(seed)
        self.source_sample_rate_hz = float(sample_rate_hz)
        self.target_sample_rate_hz = float(target_sample_rate_hz) if target_sample_rate_hz is not None else None
        self.model_sample_rate_hz = float(model_sample_rate_hz)
        self.use_swmr = bool(use_swmr)
        self.rdcc_nbytes = int(rdcc_nbytes)
        self.rdcc_nslots = int(rdcc_nslots)
        self.rdcc_w0 = float(rdcc_w0)
        self.locking = bool(locking)
        self.use_read_direct = bool(use_read_direct)

        # One-time metadata read (plain open; SWMR not needed just to read shapes)
        with h5py.File(self.h5_path, "r") as f:
            d = f[f"{split}/data"]
            self.length = d.shape[0]
            self.C = d.shape[1]
            self.T = d.shape[2]
            self.n_channels = self.C

        if self.crop_size > self.T:
            raise ValueError(f"crop_size {self.crop_size} > T {self.T}")

        self.window_len_samples = self.resample_to_length or self.crop_size
        self.sample_rate_hz = (
            float(self.target_sample_rate_hz) if self.target_sample_rate_hz is not None else float(self.source_sample_rate_hz)
        )
        self.window_len_sec = float(self.crop_size) / float(self.source_sample_rate_hz)
        self.apply_vit_resample = abs(self.sample_rate_hz - self.model_sample_rate_hz) > 1e-6

        # HDF5 handle and optional per-worker buffer (created after fork)
        self.h5_file = None
        self._buf = None  # numpy buffer for read_direct()

        # Avoid stale file locking warnings in some HPC stacks
        os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

    def _open(self):
        if self.h5_file is not None:
            return
        if self.use_swmr:
            self.h5_file = h5py.File(
                self.h5_path, "r",
                swmr=True, libver="latest",
                rdcc_nbytes=self.rdcc_nbytes,
                rdcc_nslots=self.rdcc_nslots,
                rdcc_w0=self.rdcc_w0,
                locking=self.locking,
            )
        else:
            # Fast path that matches your Challenge-1 setup
            self.h5_file = h5py.File(self.h5_path, "r")

        # Reseed RNG per worker so crops differ
        info = get_worker_info()
        if info is not None:
            self.rng.seed((self.base_seed + info.id) % 2**32)

        # Allocate per-worker buffer for read_direct (once)
        if self.use_read_direct and self._buf is None:
            self._buf = np.empty((self.C, self.crop_size), dtype=np.float32)

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        self._open()
        dX = self.h5_file[f"{self.split}/data"]
        dy = self.h5_file[f"{self.split}/targets"]

        start = self.rng.randint(0, self.T - self.crop_size)

        if self.use_read_direct:
            # Read one sample + time slice straight into buffer (no intermediate array)
            # Source: dX[idx, :, start:start+crop]  ->  Dest: _buf[:, :]
            dX.read_direct(self._buf,
                           source_sel=np.s_[idx, :, start:start+self.crop_size],
                           dest_sel=np.s_[:, :])
            X = self._buf  # (C, crop)
        else:
            # Simple path (fast if file is contiguous or per-sample chunked)
            X = dX[idx]                    # (C, T)
            X = X[:, start:start+self.crop_size]  # (C, crop)

        if self.resample_to_length is not None and self.resample_to_length != X.shape[1]:
            from scipy.signal import resample

            X = resample(np.asarray(X), self.resample_to_length, axis=1)

        y = dy[idx]
        return torch.from_numpy(np.asarray(X)).float(), torch.tensor(y, dtype=torch.float32)

    def __del__(self):
        try:
            if self.h5_file is not None:
                self.h5_file.close()
        except Exception:
            pass

    # Do not pickle open handles/buffers across forks
    def __getstate__(self):
        state = self.__dict__.copy()
        state["h5_file"] = None
        state["_buf"] = None
        return state
