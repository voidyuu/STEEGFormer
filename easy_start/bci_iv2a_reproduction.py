from __future__ import annotations

import argparse
import json
import math
import pickle
import shutil
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import mne
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import yaml
from scipy.signal import resample as scipy_resample
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.layers import trunc_normal_
from torch.utils.data import DataLoader, TensorDataset

THIS_FILE = Path(__file__).resolve()
EASY_START_DIR = THIS_FILE.parent
REPO_ROOT = EASY_START_DIR.parent
BENCHMARK_DIR = REPO_ROOT / "benchmark" / "neural_networks"
RAW_DIR = REPO_ROOT / "artifacts" / "bci_iv2a" / "raw"
PROCESSED_DIR = REPO_ROOT / "artifacts" / "bci_iv2a" / "processed"
INTERMEDIATE_DIR = PROCESSED_DIR / "_intermediate"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"
CONFIG_PATH = EASY_START_DIR / "configs" / "bci_iv2a.yaml"
DATASET_SPECS_PATH = EASY_START_DIR / "configs" / "bci_iv2a_dataset_specs.yaml"
TASK_SPECS_PATH = EASY_START_DIR / "configs" / "bci_iv2a_downstream_task_specs.yaml"
SMALL_CHECKPOINT = CHECKPOINT_DIR / "checkpoint-300.pth"
LARGE_CHECKPOINT = CHECKPOINT_DIR / "large_weights_only_196.pth"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output_dir" / "bci_iv2a_tutorial"
DEFAULT_LOG_DIR = REPO_ROOT / "runs" / "bci_iv2a_tutorial"
RAW_FILE_NAMES = [f"A0{sub}{split}.mat" for sub in range(1, 10) for split in ("T", "E")]
BNCI_BASE_URL = "https://bnci-horizon-2020.eu/database/data-sets/001-2014"
GITHUB_RELEASES_API = "https://api.github.com/repos/LiuyinYang1101/STEEGFormer/releases"
CLASS_LABEL = {"left_hand": 0, "right_hand": 1, "feet": 2, "tongue": 3}
CH_NAMES = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4",
    "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz",
]

sys.path.insert(0, str(BENCHMARK_DIR))

from models.eegnet import EEGNet  # noqa: E402
from models import models_vit_eeg  # noqa: E402
from util.eeg_downstream_dataset import BCI2aDataset  # noqa: E402
from util.lr_decay import param_groups_lrd  # noqa: E402

mne.set_log_level("ERROR")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce the BCI-IV-2A tutorial locally.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("download", help="Download raw BCI-IV-2A data and the small checkpoint.")
    subparsers.add_parser("preprocess", help="Convert raw .mat files into processed subject pickles.")
    subparsers.add_parser("smoke-test", help="Run the dataset smoke test on A1.pkl.")

    compare = subparsers.add_parser("train-compare", help="Run the Subject-1 EEGNet vs ST-EEGFormer comparison.")
    compare.add_argument("--epochs", type=int, default=100)
    compare.add_argument("--batch-size", type=int, default=16)
    compare.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    compare.add_argument("--skip-small", action="store_true")

    run_eval = subparsers.add_parser("benchmark-command", help="Print the Section 9 benchmark command for local use.")
    run_eval.add_argument("--model", default="vit_base_patch32")
    run_eval.add_argument("--optimizer-spec", default="linear_prob")
    run_eval.add_argument("--train-epochs", type=int, default=100)
    run_eval.add_argument("--finetune-epochs", type=int, default=50)

    subparsers.add_parser("all", help="Run download, preprocess, smoke-test, and train-compare in sequence.")
    return parser.parse_args()


def ensure_directories() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"skip existing: {destination}")
        return
    print(f"downloading {url} -> {destination}")
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=120) as response, destination.open("wb") as out:
        shutil.copyfileobj(response, out)


def fetch_json(url: str) -> list[dict]:
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/vnd.github+json",
        },
    )
    with urlopen(request, timeout=60) as response:
        return json.load(response)


def download_raw_dataset() -> None:
    missing = [name for name in RAW_FILE_NAMES if not (RAW_DIR / name).exists()]
    if not missing:
        print("all 18 raw BCI-IV-2A files are already present.")
        return
    for name in RAW_FILE_NAMES:
        download_file(f"{BNCI_BASE_URL}/{name}", RAW_DIR / name)


def resolve_small_checkpoint_asset() -> tuple[str, str]:
    releases = fetch_json(GITHUB_RELEASES_API)
    for release in releases:
        tag = (release.get("tag_name") or "").lower()
        name = (release.get("name") or "").lower()
        if "small" not in tag and "small" not in name:
            continue
        for asset in release.get("assets", []):
            asset_name = asset.get("name", "")
            if asset_name.endswith(".pth"):
                return asset_name, asset["browser_download_url"]
    raise RuntimeError("Could not find a small ST-EEGFormer .pth asset in GitHub releases.")


def download_small_checkpoint() -> None:
    if SMALL_CHECKPOINT.exists():
        print(f"small checkpoint already present: {SMALL_CHECKPOINT}")
        return
    asset_name, asset_url = resolve_small_checkpoint_asset()
    if asset_name != SMALL_CHECKPOINT.name:
        raise RuntimeError(f"Unexpected small checkpoint asset name: {asset_name}")
    download_file(asset_url, SMALL_CHECKPOINT)


def verify_large_checkpoint() -> None:
    if not LARGE_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Expected local large checkpoint at {LARGE_CHECKPOINT}, but it was not found."
        )
    print(f"large checkpoint ready: {LARGE_CHECKPOINT}")


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def process_mat_file(path: Path) -> dict:
    data = scipy.io.loadmat(path)["data"]
    return {
        session_idx: {name: data[0][session_idx][name][0, 0] for name in data[0][session_idx].dtype.names}
        for session_idx in range(len(data[0]))
    }


def process_eeg(raw: mne.io.RawArray, f_l: float, f_h: float, notch_freqs: list[int], fs: int) -> np.ndarray:
    raw = raw.notch_filter(freqs=notch_freqs)
    raw.filter(f_l, f_h, picks=raw.info["ch_names"], fir_design="firwin")
    raw.resample(fs, npad="auto")
    return raw.get_data()


def process_eeg_per_session(rec: dict, f_l: float, f_h: float, notch_freqs: list[int], downfs: int) -> dict:
    for session_idx in range(len(rec)):
        session_data = rec[session_idx]
        eeg = np.transpose(session_data["X"][:, 0:22])
        fs = int(session_data["fs"][0][0])
        info = mne.create_info(CH_NAMES, ch_types=["eeg"] * 22, sfreq=fs)
        info.set_montage("standard_1005")
        raw = mne.io.RawArray(eeg, info)
        if session_data["trial"].shape[0] != 0:
            session_data["trial"] = np.rint(session_data["trial"] * downfs / fs)
        session_data["X"] = process_eeg(raw, f_l, f_h, notch_freqs, downfs)
        session_data["fs"] = downfs
    return rec


def preprocess_raw_to_intermediate() -> None:
    cfg = load_config()
    raw_eeg_folder = Path(cfg["data"]["path"])
    all_subjects = sorted(raw_eeg_folder.glob("*.mat"))
    if len(all_subjects) != 18:
        raise FileNotFoundError(f"Expected 18 raw .mat files in {raw_eeg_folder}, found {len(all_subjects)}.")
    print(f"Found {len(all_subjects)} files to preprocess.")
    for idx, subject_path in enumerate(all_subjects, start=1):
        name = subject_path.stem
        out_path = INTERMEDIATE_DIR / f"{name}.pkl"
        print(f"[{idx}/{len(all_subjects)}] {name}")
        eeg_data = process_mat_file(subject_path)
        eeg_data = process_eeg_per_session(
            eeg_data,
            cfg["filter"]["l_freq"],
            cfg["filter"]["h_freq"],
            cfg["filter"]["notch"],
            cfg["resample_freq"],
        )
        with out_path.open("wb") as f:
            pickle.dump(eeg_data, f)


def epoch_subject(pkl_path: Path, start_t: float, end_t: float, label_map: dict[int, str]) -> tuple[np.ndarray, list[str]]:
    with pkl_path.open("rb") as f:
        sessions = pickle.load(f)
    trials, ys = [], []
    for session_idx in range(len(sessions)):
        session_data = sessions[session_idx]
        for trial_idx, onset in enumerate(session_data["trial"]):
            start = int(onset[0] + start_t * session_data["fs"])
            end = int(onset[0] + end_t * session_data["fs"])
            trials.append(session_data["X"][:, start:end])
            ys.append(label_map[int(session_data["y"][trial_idx][0])])
    return np.stack(trials, axis=0), ys


def build_subject_pickles() -> None:
    cfg = load_config()
    for sub in range(1, 10):
        train_x, train_y = epoch_subject(
            INTERMEDIATE_DIR / f"A0{sub}T.pkl",
            cfg["epochs"]["tmin"],
            cfg["epochs"]["tmax"],
            cfg["events"]["mapping"],
        )
        test_x, test_y = epoch_subject(
            INTERMEDIATE_DIR / f"A0{sub}E.pkl",
            cfg["epochs"]["tmin"],
            cfg["epochs"]["tmax"],
            cfg["events"]["mapping"],
        )
        print(f"sub{sub}: train {train_x.shape}, test {test_x.shape}")
        with (PROCESSED_DIR / f"A{sub}.pkl").open("wb") as f:
            pickle.dump(
                {
                    "trainX": train_x,
                    "trainY": train_y,
                    "testX": test_x,
                    "testY": test_y,
                },
                f,
            )


def validate_processed_dataset() -> None:
    for sub in range(1, 10):
        with (PROCESSED_DIR / f"A{sub}.pkl").open("rb") as f:
            subject = pickle.load(f)
        for split in ("trainX", "testX"):
            if subject[split].shape != (288, 22, 1024):
                raise ValueError(f"{split} for A{sub}.pkl has shape {subject[split].shape}, expected (288, 22, 1024).")
        for split in ("trainY", "testY"):
            unique = sorted(set(subject[split]))
            if unique != sorted(CLASS_LABEL.keys()):
                raise ValueError(f"{split} for A{sub}.pkl has labels {unique}, expected {sorted(CLASS_LABEL.keys())}.")
    intermediate_files = sorted(INTERMEDIATE_DIR.glob("*.pkl"))
    if len(intermediate_files) != 18:
        raise ValueError(f"Expected 18 intermediate pickles, found {len(intermediate_files)}.")
    print("processed dataset checks passed.")


def run_preprocess() -> None:
    preprocess_raw_to_intermediate()
    build_subject_pickles()
    validate_processed_dataset()


def run_smoke_test() -> None:
    ds = BCI2aDataset(
        data_path=str(PROCESSED_DIR / "A1.pkl"),
        fold=0,
        train=True,
        class_label=CLASS_LABEL,
        data_length=1024,
    )
    print(f"len(dataset) = {len(ds)}")
    x, y = ds[0]
    print(f"sample 0: x.shape={tuple(x.shape)} dtype={x.dtype} label={y}")
    loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)
    xb, yb = next(iter(loader))
    print(f"batch: x.shape={tuple(xb.shape)} y={yb.tolist()}")
    if tuple(x.shape) != (22, 1024):
        raise ValueError(f"Unexpected sample shape: {tuple(x.shape)}")
    if tuple(xb.shape) != (8, 22, 1024):
        raise ValueError(f"Unexpected batch shape: {tuple(xb.shape)}")
    print("smoke test passed.")


def zscore(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=2, keepdims=True)
    std = x.std(axis=2, keepdims=True) + 1e-8
    return (x - mean) / std


def extract_state_dict(ckpt: object) -> OrderedDict:
    if isinstance(ckpt, OrderedDict):
        return ckpt
    if isinstance(ckpt, dict):
        if "model" in ckpt and isinstance(ckpt["model"], (dict, OrderedDict)):
            return ckpt["model"]
        if all(isinstance(k, str) for k in ckpt.keys()):
            return OrderedDict(ckpt)
    raise TypeError(f"Unsupported checkpoint format: {type(ckpt)}")


def load_pretrained_vit(factory: Callable, ckpt_path: Path, num_classes: int, drop_path: float = 0.1) -> nn.Module:
    model = factory(num_classes=num_classes, drop_path_rate=drop_path, global_pool=True)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = extract_state_dict(ckpt)
    for key in ("head.weight", "head.bias"):
        if key in state_dict and state_dict[key].shape != model.state_dict()[key].shape:
            del state_dict[key]
    model.load_state_dict(state_dict, strict=False)
    trunc_normal_(model.head.weight, std=2e-5)
    return model


class BCIDatasetWithChanInfo(torch.utils.data.Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, chan_idx: torch.Tensor):
        self.x = x
        self.y = y
        self.chan_idx = chan_idx

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.x[idx]), self.y[idx], self.chan_idx


def cosine_lr(epoch_frac: float, epochs: int, warmup: int, base_lr: float, min_lr: float) -> float:
    if epoch_frac < warmup:
        return base_lr * epoch_frac / warmup
    return min_lr + (base_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * (epoch_frac - warmup) / (epochs - warmup)))


def apply_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr * group.get("lr_scale", 1.0)


def make_autocast(device: torch.device):
    enabled = device.type == "cuda"
    return torch.amp.autocast(device_type=device.type, enabled=enabled)


def run_recipe(
    name: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    test_loader: DataLoader,
    takes_chan_idx: bool,
    device: torch.device,
    epochs: int,
    warmup: int,
    base_lr: float,
    min_lr: float,
    mixup_fn: Mixup,
    criterion: SoftTargetCrossEntropy,
    num_classes: int,
) -> tuple[float, float]:
    print(f"\n=== {name} ===")
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    t0 = time.time()
    final = best = 0.0
    for epoch in range(epochs):
        model.train()
        for iteration, batch in enumerate(train_loader):
            apply_lr(optimizer, cosine_lr(epoch + iteration / len(train_loader), epochs, warmup, base_lr, min_lr))
            x, y, *rest = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            chan_info = rest[0].to(device, non_blocking=True) if takes_chan_idx else None
            if len(x) % 2 == 0:
                x_mix, y_soft = mixup_fn(x, y)
            else:
                x_mix = x
                y_soft = nn.functional.one_hot(y, num_classes).float() * 0.9 + 0.1 / num_classes
            optimizer.zero_grad()
            with make_autocast(device):
                out = model(x_mix, chan_info) if takes_chan_idx else model(x_mix)
                loss = criterion(out, y_soft)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        model.eval()
        total = hits = 0
        with torch.no_grad():
            for batch in test_loader:
                x, y, *rest = batch
                x = x.to(device)
                y = y.to(device)
                chan_info = rest[0].to(device) if takes_chan_idx else None
                with make_autocast(device):
                    out = model(x, chan_info) if takes_chan_idx else model(x)
                hits += (out.argmax(1) == y).sum().item()
                total += y.size(0)
        test_acc = hits / total
        final = test_acc
        best = max(best, test_acc)
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  ep {epoch + 1:3d}/{epochs}  test_acc={test_acc:.4f}  (best {best:.4f})")
    print(f"  -> {time.time() - t0:.0f}s; final={final * 100:.2f}%, best={best * 100:.2f}%")
    return final, best


def prepare_subject_one() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    torch.Tensor,
    np.ndarray,
    np.ndarray,
]:
    with (PROCESSED_DIR / "A1.pkl").open("rb") as f:
        subject = pickle.load(f)
    train_x = subject["trainX"].astype(np.float32)
    train_y = np.array([CLASS_LABEL[label] for label in subject["trainY"]], dtype=np.int64)
    test_x = subject["testX"].astype(np.float32)
    test_y = np.array([CLASS_LABEL[label] for label in subject["testY"]], dtype=np.int64)
    with (REPO_ROOT / "pretrain" / "senloc_file" / "sen_chan_idx.pkl").open("rb") as f:
        senloc = pickle.load(f)
    channel_mapping = {k.lower(): v for k, v in senloc["channels_mapping"].items()}
    chan_idx = torch.tensor([channel_mapping[name.lower()] for name in CH_NAMES], dtype=torch.long)
    train_x_vit = zscore(scipy_resample(train_x, 128 * 4, axis=2)).astype(np.float32)
    test_x_vit = zscore(scipy_resample(test_x, 128 * 4, axis=2)).astype(np.float32)
    return train_x, train_y, test_x, test_y, chan_idx, train_x_vit, test_x_vit


def run_train_compare(epochs: int, batch_size: int, device_name: str, skip_small: bool) -> None:
    verify_large_checkpoint()
    if not skip_small and not SMALL_CHECKPOINT.exists():
        raise FileNotFoundError(f"Expected local small checkpoint at {SMALL_CHECKPOINT}, but it was not found.")

    device = torch.device(device_name)
    torch.manual_seed(0)
    np.random.seed(0)
    base_lr, min_lr = 1e-3, 1e-6
    weight_decay, layer_decay = 0.05, 0.75
    warmup, num_classes = 10, 4

    train_x, train_y, test_x, test_y, chan_idx, train_x_vit, test_x_vit = prepare_subject_one()

    mixup_fn = Mixup(
        mixup_alpha=0.2,
        cutmix_alpha=0.0,
        prob=0.9,
        switch_prob=0.0,
        mode="batch",
        label_smoothing=0.1,
        num_classes=num_classes,
    )
    criterion = SoftTargetCrossEntropy()

    tr_eegnet = DataLoader(TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)), batch_size=batch_size, shuffle=True)
    te_eegnet = DataLoader(TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y)), batch_size=batch_size)
    tr_vit = DataLoader(BCIDatasetWithChanInfo(train_x_vit, train_y, chan_idx), batch_size=batch_size, shuffle=True)
    te_vit = DataLoader(BCIDatasetWithChanInfo(test_x_vit, test_y, chan_idx), batch_size=batch_size)

    original_bn2d = nn.BatchNorm2d

    def compat_bn2d(num_features, *args, **kwargs):
        if args and args[0] is False:
            args = ()
            kwargs.setdefault("affine", False)
        return original_bn2d(num_features, *args, **kwargs)

    nn.BatchNorm2d = compat_bn2d
    eegnet = EEGNet(
        no_spatial_filters=4,
        no_channels=22,
        no_temporal_filters=8,
        temporal_length_1=train_x.shape[-1] // 2,
        temporal_length_2=(256 // 128) * 16,
        window_length=train_x.shape[-1],
        num_class=num_classes,
        pooling2=256 // 32,
        pooling3=8,
    ).to(device)
    nn.BatchNorm2d = original_bn2d
    eegnet_opt = torch.optim.AdamW(eegnet.parameters(), lr=base_lr, weight_decay=weight_decay)
    acc_eegnet = run_recipe(
        "EEGNet",
        eegnet,
        eegnet_opt,
        tr_eegnet,
        te_eegnet,
        takes_chan_idx=False,
        device=device,
        epochs=epochs,
        warmup=warmup,
        base_lr=base_lr,
        min_lr=min_lr,
        mixup_fn=mixup_fn,
        criterion=criterion,
        num_classes=num_classes,
    )
    del eegnet, eegnet_opt
    if device.type == "cuda":
        torch.cuda.empty_cache()

    acc_small = None
    if not skip_small:
        torch.manual_seed(0)
        np.random.seed(0)
        model_small = load_pretrained_vit(models_vit_eeg.vit_small_patch16, SMALL_CHECKPOINT, num_classes).to(device)
        no_wd_small = set(model_small.no_weight_decay()) if hasattr(model_small, "no_weight_decay") else set()
        opt_small = torch.optim.AdamW(
            param_groups_lrd(
                model_small,
                weight_decay=weight_decay,
                no_weight_decay_list=no_wd_small,
                layer_decay=layer_decay,
            ),
            lr=base_lr,
        )
        acc_small = run_recipe(
            "ST-EEGFormer-small (pretrained)",
            model_small,
            opt_small,
            tr_vit,
            te_vit,
            takes_chan_idx=True,
            device=device,
            epochs=epochs,
            warmup=warmup,
            base_lr=base_lr,
            min_lr=min_lr,
            mixup_fn=mixup_fn,
            criterion=criterion,
            num_classes=num_classes,
        )
        del model_small, opt_small
        if device.type == "cuda":
            torch.cuda.empty_cache()

    torch.manual_seed(0)
    np.random.seed(0)
    model_large = load_pretrained_vit(models_vit_eeg.vit_large_patch16, LARGE_CHECKPOINT, num_classes).to(device)
    no_wd_large = set(model_large.no_weight_decay()) if hasattr(model_large, "no_weight_decay") else set()
    opt_large = torch.optim.AdamW(
        param_groups_lrd(
            model_large,
            weight_decay=weight_decay,
            no_weight_decay_list=no_wd_large,
            layer_decay=layer_decay,
        ),
        lr=base_lr,
    )
    acc_large = run_recipe(
        "ST-EEGFormer-large (pretrained)",
        model_large,
        opt_large,
        tr_vit,
        te_vit,
        takes_chan_idx=True,
        device=device,
        epochs=epochs,
        warmup=warmup,
        base_lr=base_lr,
        min_lr=min_lr,
        mixup_fn=mixup_fn,
        criterion=criterion,
        num_classes=num_classes,
    )

    print("\n=== Subject 1, benchmark recipe — final / best test acc ===")
    print(f"  EEGNet:                  {acc_eegnet[0] * 100:5.2f}% / {acc_eegnet[1] * 100:5.2f}%")
    if acc_small is not None:
        print(f"  ST-EEGFormer-small (FT): {acc_small[0] * 100:5.2f}% / {acc_small[1] * 100:5.2f}%")
    else:
        print("  ST-EEGFormer-small (FT): skipped")
    print(f"  ST-EEGFormer-large (FT): {acc_large[0] * 100:5.2f}% / {acc_large[1] * 100:5.2f}%")


def print_benchmark_command(model: str, optimizer_spec: str, train_epochs: int, finetune_epochs: int) -> None:
    command = f"""cd {BENCHMARK_DIR}
{REPO_ROOT / '.venv' / 'bin' / 'python'} wandb_downstream_evaluation.py \\
    --downstream_task bci_iv2a \\
    --evaluation_scheme per-subject \\
    --model {model} \\
    --vit_pretrained_model_dir {LARGE_CHECKPOINT} \\
    --optimizer_spec {optimizer_spec} \\
    --train_epochs {train_epochs} \\
    --finetune_epochs {finetune_epochs} \\
    --dataset_yaml {DATASET_SPECS_PATH} \\
    --downstream_task_yaml {TASK_SPECS_PATH} \\
    --output_dir {DEFAULT_OUTPUT_DIR} \\
    --log_dir {DEFAULT_LOG_DIR} \\
    --wandb_project bci_iv2a_eval"""
    print(command)


def main() -> None:
    args = parse_args()
    ensure_directories()
    if args.command == "download":
        download_raw_dataset()
        download_small_checkpoint()
        verify_large_checkpoint()
    elif args.command == "preprocess":
        run_preprocess()
    elif args.command == "smoke-test":
        run_smoke_test()
    elif args.command == "train-compare":
        run_train_compare(args.epochs, args.batch_size, args.device, args.skip_small)
    elif args.command == "benchmark-command":
        print_benchmark_command(args.model, args.optimizer_spec, args.train_epochs, args.finetune_epochs)
    elif args.command == "all":
        download_raw_dataset()
        download_small_checkpoint()
        verify_large_checkpoint()
        run_preprocess()
        run_smoke_test()
        run_train_compare(epochs=100, batch_size=16, device_name="cuda" if torch.cuda.is_available() else "cpu", skip_small=False)


if __name__ == "__main__":
    main()
