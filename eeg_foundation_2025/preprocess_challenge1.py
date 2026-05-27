#!/usr/bin/env python3
"""Build a Challenge 1 HDF5 dataset consumable by this repository.

The output layout matches ``EEGH5Dataset`` in
``eeg_foundation_2025/utils/challenge_custom_dataset.py``:

    train/data      float32 [N, C, T]
    train/targets   float32 [N]
    valid/data      float32 [N, C, T]
    valid/targets   float32 [N]
    test/data       float32 [N, C, T]
    test/targets    float32 [N]

This script follows the official EEGDash Challenge 1 CCD windowing recipe:
stimulus-locked windows starting +0.5 s after stimulus onset, with a 2.0 s
window at 100 Hz, and regression targets from ``rt_from_stimulus``.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import random
import warnings
from collections import Counter
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None


TASK_NAME = "contrastChangeDetection"
TARGET_FIELD = "rt_from_stimulus"
ANCHOR_NAME = "stimulus_anchor"
DEFAULT_CACHE_DIR = Path("/mnt/E/zhuyu_data/eeg-challenges/raw_data")
DEFAULT_DATA_DIR = Path("/mnt/E/zhuyu_data/eeg-challenges/challenge1")
DEFAULT_OUTPUT_H5 = DEFAULT_DATA_DIR / "eeg_challenge1_dataset.h5"


def require_numpy():
    if np is None:
        raise ImportError(
            "numpy is required for challenge1 preprocessing. Install it before running this script."
        )
    return np


def load_preprocessing_dependencies():
    try:
        from braindecode.preprocessing import (
            Preprocessor,
            create_windows_from_events,
            preprocess,
        )
    except ImportError as exc:
        raise ImportError(
            "braindecode is required for challenge1 preprocessing. "
            "Install it before running this script."
        ) from exc

    try:
        import mne
    except ImportError as exc:
        raise ImportError(
            "mne is required for challenge1 preprocessing. "
            "Install it before running this script."
        ) from exc

    try:
        from eegdash import EEGChallengeDataset
    except ImportError:
        try:
            from eegdash.dataset import EEGChallengeDataset
        except ImportError as exc:
            raise ImportError(
                "eegdash is required for challenge1 preprocessing. "
                "Install it before running this script."
            ) from exc

    try:
        from scipy.signal import resample as scipy_resample
    except ImportError as exc:
        raise ImportError(
            "scipy is required for challenge1 preprocessing. "
            "Install it before running this script."
        ) from exc

    try:
        from eegdash.hbn.windows import (
            add_aux_anchors,
            add_extras_columns,
            annotate_trials_with_target,
            keep_only_recordings_with,
        )
    except ImportError as exc:
        raise ImportError(
            "The installed eegdash package does not expose the required HBN window helpers."
        ) from exc

    return (
        EEGChallengeDataset,
        Preprocessor,
        create_windows_from_events,
        preprocess,
        mne,
        scipy_resample,
        add_aux_anchors,
        add_extras_columns,
        annotate_trials_with_target,
        keep_only_recordings_with,
    )


def parse_release_list(values: list[str]) -> list[str]:
    """Accept ``--datasets 1 2 3`` as the primary format.

    We also keep backward compatibility with ``--datasets "[1,2,3]"``.
    """
    if not values:
        raise argparse.ArgumentTypeError("--datasets cannot be empty")

    tokens: list[int] = []
    for value in values:
        value = value.strip()
        if not value:
            continue
        if value.startswith("["):
            parsed = ast.literal_eval(value)
            if not isinstance(parsed, (list, tuple)):
                raise argparse.ArgumentTypeError(
                    f"Expected a list-like value for --datasets, got {value!r}"
                )
            tokens.extend(int(item) for item in parsed)
            continue
        parts = [part.strip() for part in value.split(",") if part.strip()]
        tokens.extend(int(part) for part in parts)

    ordered = []
    seen = set()
    for token in tokens:
        if token <= 0:
            raise argparse.ArgumentTypeError(
                f"Release numbers must be positive integers, got {token}"
            )
        release = f"R{token}"
        if release not in seen:
            seen.add(release)
            ordered.append(release)
    if not ordered:
        raise argparse.ArgumentTypeError("No valid releases were parsed from --datasets")
    return ordered


def non_negative_float(value: str) -> float:
    result = float(value)
    if result < 0:
        raise argparse.ArgumentTypeError(f"Expected a non-negative float, got {value}")
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess EEG Challenge 1 releases into a single HDF5 file."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Release numbers, e.g. --datasets 1 2 3",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(DEFAULT_CACHE_DIR),
        help=f"Local EEGDash cache directory used to download/read releases. Default: {DEFAULT_CACHE_DIR}",
    )
    parser.add_argument(
        "--output-h5",
        default=str(DEFAULT_OUTPUT_H5),
        help=f"Destination HDF5 file path. Default: {DEFAULT_OUTPUT_H5}",
    )
    parser.add_argument(
        "--metadata-csv",
        default=None,
        help="Optional metadata CSV path. Defaults next to --output-h5.",
    )
    parser.add_argument(
        "--subject-splits-json",
        default=None,
        help="Optional JSON path for the subject split manifest. Defaults next to --output-h5.",
    )
    parser.add_argument(
        "--task",
        default=TASK_NAME,
        help=f"Challenge task to query. Default: {TASK_NAME}",
    )
    parser.add_argument(
        "--target-field",
        default=TARGET_FIELD,
        help=f"Regression target field. Default: {TARGET_FIELD}",
    )
    parser.add_argument(
        "--window-start-sec",
        type=non_negative_float,
        default=0.5,
        help="Window start offset after stimulus onset in seconds. Default: 0.5",
    )
    parser.add_argument(
        "--window-len-sec",
        type=non_negative_float,
        default=2.0,
        help="Window length in seconds. Default: 2.0",
    )
    parser.add_argument(
        "--sample-rate",
        type=non_negative_float,
        default=100.0,
        help="Expected raw challenge sample rate in Hz. Default: 100.0",
    )
    parser.add_argument(
        "--target-sfreq",
        type=non_negative_float,
        default=128.0,
        help="Output sample rate in Hz after offline resampling. Default: 128.0",
    )
    parser.add_argument(
        "--window-stride-sec",
        type=non_negative_float,
        default=1.0,
        help="Stride passed to Braindecode windowing in seconds. Default: 1.0",
    )
    parser.add_argument(
        "--valid-ratio",
        type=non_negative_float,
        default=0.1,
        help="Subject-level validation split ratio. Default: 0.1",
    )
    parser.add_argument(
        "--test-ratio",
        type=non_negative_float,
        default=0.1,
        help="Subject-level test split ratio. Default: 0.1",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="Random seed for subject splitting. Default: 2025",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="n_jobs passed to braindecode.preprocessing.preprocess. Default: 1",
    )
    parser.add_argument(
        "--compression",
        choices=("none", "lzf", "gzip"),
        default="none",
        help="Optional HDF5 compression. Default: none",
    )
    parser.add_argument(
        "--description-fields",
        nargs="+",
        default=["subject", "session", "run", "task"],
        help="Metadata fields to request from EEGDash. Default: subject session run task",
    )
    parser.add_argument(
        "--bandpass-low",
        type=non_negative_float,
        default=0.5,
        help="Bandpass low cutoff in Hz. Default: 0.5",
    )
    parser.add_argument(
        "--bandpass-high",
        type=non_negative_float,
        default=40.0,
        help="Bandpass high cutoff in Hz. Default: 40.0",
    )
    parser.add_argument(
        "--notch-freqs",
        nargs="*",
        type=float,
        default=[],
        help="Notch filter frequencies in Hz. Default: disabled",
    )
    parser.add_argument(
        "--artifact-reject",
        choices=("off",),
        default="off",
        help="Window-level artifact rejection policy. Only 'off' is supported now.",
    )
    parser.add_argument(
        "--mini",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to use EEGDash mini releases. Default: False",
    )
    parser.add_argument(
        "--preload",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to preload all windows into memory inside Braindecode. Default: False",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to overwrite an existing H5/metadata/split manifest. Default: False",
    )
    return parser


def resolve_output_path(value: str | None, fallback: Path) -> Path:
    path = Path(value) if value is not None else fallback
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def check_overwrite(paths: list[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        pretty = "\n".join(f"  - {path}" for path in existing)
        raise FileExistsError(
            "Refusing to overwrite existing output files. "
            "Pass --overwrite to replace them:\n"
            f"{pretty}"
        )


def validate_args(args: argparse.Namespace) -> None:
    if args.valid_ratio + args.test_ratio >= 1.0:
        raise ValueError("valid_ratio + test_ratio must be < 1.0")
    if args.window_len_sec <= 0:
        raise ValueError("--window-len-sec must be positive")
    if args.sample_rate <= 0:
        raise ValueError("--sample-rate must be positive")
    if args.target_sfreq <= 0:
        raise ValueError("--target-sfreq must be positive")
    if args.bandpass_high <= args.bandpass_low:
        raise ValueError("--bandpass-high must be > --bandpass-low")


def expected_window_shape(args: argparse.Namespace) -> tuple[int, int]:
    n_samples = int(round(args.window_len_sec * args.target_sfreq))
    return 129, n_samples


def discover_subjects(
    releases: list[str],
    *,
    cache_dir: str,
    mini: bool,
    task: str,
    description_fields: list[str],
) -> dict[str, list[str]]:
    (
        EEGChallengeDataset,
        _Preprocessor,
        _create_windows_from_events,
        _preprocess,
        _mne,
        _scipy_resample,
        _add_aux_anchors,
        _add_extras_columns,
        _annotate_trials_with_target,
        _keep_only_recordings_with,
    ) = load_preprocessing_dependencies()

    per_release: dict[str, list[str]] = {}
    for release in releases:
        ds = EEGChallengeDataset(
            task=task,
            release=release,
            cache_dir=cache_dir,
            mini=mini,
            description_fields=description_fields,
        )
        description = getattr(ds, "description", None)
        if description is None or "subject" not in description.columns:
            raise RuntimeError(
                f"Could not find a 'subject' column in EEGChallengeDataset.description for {release}"
            )
        subjects = sorted(
            str(subject)
            for subject in description["subject"].dropna().astype(str).unique().tolist()
        )
        if not subjects:
            raise RuntimeError(f"No subjects found for {release}")
        per_release[release] = subjects
    return per_release


def build_subject_split_map(
    subjects: list[str],
    *,
    valid_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, str]:
    if not subjects:
        raise ValueError("No subjects available for splitting")

    shuffled = list(dict.fromkeys(subjects))
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_valid = int(round(n_total * valid_ratio))
    n_test = int(round(n_total * test_ratio))

    if valid_ratio > 0 and n_valid == 0 and n_total >= 3:
        n_valid = 1
    if test_ratio > 0 and n_test == 0 and n_total >= 3:
        n_test = 1

    while n_valid + n_test >= n_total:
        if n_test > n_valid and n_test > 0:
            n_test -= 1
        elif n_valid > 0:
            n_valid -= 1
        else:
            break

    n_train = n_total - n_valid - n_test
    if n_train <= 0:
        raise ValueError(
            f"Split would leave no training subjects: total={n_total}, valid={n_valid}, test={n_test}"
        )

    train_subjects = shuffled[:n_train]
    valid_subjects = shuffled[n_train : n_train + n_valid]
    test_subjects = shuffled[n_train + n_valid :]

    split_map: dict[str, str] = {}
    for subject in train_subjects:
        split_map[subject] = "train"
    for subject in valid_subjects:
        split_map[subject] = "valid"
    for subject in test_subjects:
        split_map[subject] = "test"
    return split_map


def create_empty_h5(
    output_h5: Path,
    *,
    n_channels: int,
    n_samples: int,
    compression: str,
    releases: list[str],
    args: argparse.Namespace,
) -> h5py.File:
    import h5py
    np_mod = require_numpy()

    compression_value = None if compression == "none" else compression
    h5f = h5py.File(output_h5, "w")
    h5f.attrs["task"] = args.task
    h5f.attrs["target_field"] = args.target_field
    h5f.attrs["releases"] = json.dumps(releases)
    h5f.attrs["mini"] = bool(args.mini)
    h5f.attrs["sample_rate_hz"] = float(args.target_sfreq)
    h5f.attrs["source_sample_rate_hz"] = float(args.sample_rate)
    h5f.attrs["model_sample_rate_hz"] = float(args.target_sfreq)
    h5f.attrs["requires_vit_resample"] = False
    h5f.attrs["window_start_sec"] = float(args.window_start_sec)
    h5f.attrs["window_len_sec"] = float(args.window_len_sec)
    h5f.attrs["window_len_samples"] = int(n_samples)
    h5f.attrs["window_stride_sec"] = float(args.window_stride_sec)
    h5f.attrs["bandpass_low_hz"] = float(args.bandpass_low)
    h5f.attrs["bandpass_high_hz"] = float(args.bandpass_high)
    h5f.attrs["notch_freqs_hz"] = np_mod.asarray(args.notch_freqs, dtype=np_mod.float32)
    h5f.attrs["artifact_policy"] = args.artifact_reject
    h5f.attrs["created_by"] = "eeg_foundation_2025/preprocess_challenge1.py"
    h5f.attrs["source_url"] = "https://eeg2025.github.io/"

    for split in ("train", "valid", "test"):
        group = h5f.create_group(split)
        data_kwargs = {
            "shape": (0, n_channels, n_samples),
            "maxshape": (None, n_channels, n_samples),
            "dtype": np_mod.float32,
            "chunks": (1, n_channels, n_samples),
        }
        target_kwargs = {
            "shape": (0,),
            "maxshape": (None,),
            "dtype": np_mod.float32,
            "chunks": (1024,),
        }
        if compression_value is not None:
            data_kwargs["compression"] = compression_value
            target_kwargs["compression"] = compression_value
        group.create_dataset("data", **data_kwargs)
        group.create_dataset("targets", **target_kwargs)
    return h5f

def _apply_continuous_filters(raw, args, mne):
    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    if len(picks) == 0:
        return
    raw.load_data()
    raw.filter(
        l_freq=float(args.bandpass_low),
        h_freq=float(args.bandpass_high),
        picks=picks,
        verbose="ERROR",
    )
    if args.notch_freqs:
        raw.notch_filter(
            freqs=[float(freq) for freq in args.notch_freqs],
            picks=picks,
            verbose="ERROR",
        )


def _prepare_recordings(ds, args, mne):
    datasets = list(getattr(ds, "datasets", []))
    for base_ds in datasets:
        raw = getattr(base_ds, "raw", None)
        if raw is None:
            continue
        _apply_continuous_filters(raw, args, mne)


def _resample_window(sample, target_n_samples, scipy_resample):
    sample = scipy_resample(sample, target_n_samples, axis=1)
    return np.asarray(sample, dtype=np.float32)


def prepare_release_windows(release: str, args: argparse.Namespace):
    (
        EEGChallengeDataset,
        Preprocessor,
        create_windows_from_events,
        preprocess,
        mne,
        scipy_resample,
        add_aux_anchors,
        add_extras_columns,
        annotate_trials_with_target,
        keep_only_recordings_with,
    ) = load_preprocessing_dependencies()

    ds = EEGChallengeDataset(
        task=args.task,
        release=release,
        cache_dir=args.cache_dir,
        mini=args.mini,
        description_fields=args.description_fields,
    )
    preprocess(
        ds,
        [
            Preprocessor(
                annotate_trials_with_target,
                target_field=args.target_field,
                epoch_length=args.window_len_sec,
                require_stimulus=True,
                require_response=True,
                apply_on_array=False,
            ),
            Preprocessor(add_aux_anchors, apply_on_array=False),
        ],
        n_jobs=args.jobs,
    )
    _prepare_recordings(ds, args, mne)

    ds = keep_only_recordings_with(ANCHOR_NAME, ds)
    windows = create_windows_from_events(
        ds,
        mapping={ANCHOR_NAME: 0},
        trial_start_offset_samples=int(round(args.window_start_sec * args.sample_rate)),
        trial_stop_offset_samples=int(
            round((args.window_start_sec + args.window_len_sec) * args.sample_rate)
        ),
        window_size_samples=int(round(args.window_len_sec * args.sample_rate)),
        window_stride_samples=int(round(args.window_stride_sec * args.sample_rate)),
        preload=args.preload,
    )
    windows = add_extras_columns(
        windows,
        ds,
        desc=ANCHOR_NAME,
        keys=("target", args.target_field, "stimulus_onset", "response_onset"),
    )
    metadata = windows.get_metadata().reset_index(drop=True)
    np_mod = require_numpy()
    metadata["window_index"] = np_mod.arange(len(metadata), dtype=np_mod.int64)
    metadata["release"] = release
    release_stats = {
        "recordings_total": int(len(getattr(ds, "datasets", []))),
    }
    return windows, metadata, scipy_resample, release_stats


def metadata_field(row, field: str, default: str = "") -> str:
    value = row[field] if field in row else default
    if value is None:
        return default
    np_mod = require_numpy()
    if isinstance(value, float) and not np_mod.isfinite(value):
        return default
    return str(value)


def main() -> None:
    warnings.simplefilter("ignore", category=FutureWarning)

    parser = build_arg_parser()
    args = parser.parse_args()
    args.releases = parse_release_list(args.datasets)
    validate_args(args)

    output_h5 = resolve_output_path(args.output_h5, Path("eeg_challenge1_dataset.h5"))
    metadata_csv = resolve_output_path(
        args.metadata_csv, output_h5.with_suffix(output_h5.suffix + ".metadata.csv")
    )
    split_json = resolve_output_path(
        args.subject_splits_json, output_h5.with_suffix(output_h5.suffix + ".subject_splits.json")
    )
    check_overwrite([output_h5, metadata_csv, split_json], overwrite=args.overwrite)

    np_mod = require_numpy()
    n_channels, n_samples = expected_window_shape(args)
    print(f"Preparing releases: {args.releases}")
    print(f"Window contract: channels={n_channels}, samples={n_samples}, target={args.target_field}")
    print(f"Output H5: {output_h5}")

    subjects_by_release = discover_subjects(
        args.releases,
        cache_dir=args.cache_dir,
        mini=args.mini,
        task=args.task,
        description_fields=args.description_fields,
    )
    all_subjects = sorted(
        {
            subject
            for release_subjects in subjects_by_release.values()
            for subject in release_subjects
        }
    )
    subject_to_split = build_subject_split_map(
        all_subjects,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    split_manifest = {
        "releases": args.releases,
        "seed": args.seed,
        "mini": bool(args.mini),
        "valid_ratio": args.valid_ratio,
        "test_ratio": args.test_ratio,
        "subjects_by_release": subjects_by_release,
        "split_by_subject": subject_to_split,
        "preprocessing": {
            "bandpass_low_hz": args.bandpass_low,
            "bandpass_high_hz": args.bandpass_high,
            "notch_freqs_hz": [float(freq) for freq in args.notch_freqs],
            "target_sfreq": args.target_sfreq,
            "artifact_policy": args.artifact_reject,
        },
    }

    h5f = create_empty_h5(
        output_h5,
        n_channels=n_channels,
        n_samples=n_samples,
        compression=args.compression,
        releases=args.releases,
        args=args,
    )

    csv_fields = [
        "split",
        "split_index",
        "release",
        "subject",
        "session",
        "run",
        "target",
        args.target_field,
        "stimulus_onset",
        "response_onset",
    ]
    write_positions = {"train": 0, "valid": 0, "test": 0}
    release_summaries = {}

    with h5f, metadata_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
        writer.writeheader()

        for release in args.releases:
            print(f"\n[release {release}] building EEGDash windows...")
            windows, metadata, scipy_resample, release_stats = prepare_release_windows(release, args)

            if "subject" not in metadata.columns:
                raise RuntimeError(
                    f"Window metadata for {release} is missing 'subject'; cannot do subject-level split."
                )

            metadata["subject"] = metadata["subject"].astype(str)
            metadata["split"] = metadata["subject"].map(subject_to_split)
            metadata["target_float"] = metadata[args.target_field].astype(float)

            valid_mask = metadata["split"].notna() & np_mod.isfinite(metadata["target_float"].to_numpy())
            metadata = metadata.loc[valid_mask].reset_index(drop=True)

            retained_records: list[tuple[str, int, np.ndarray, object]] = []
            kept_counts = Counter()
            for row_idx in range(len(metadata)):
                row = metadata.iloc[row_idx]
                item = windows[int(row["window_index"])]
                sample = np_mod.asarray(item[0], dtype=np_mod.float32)
                sample = _resample_window(sample, n_samples, scipy_resample)
                if sample.shape != (n_channels, n_samples):
                    raise ValueError(
                        f"Unexpected sample shape for {release} row {row_idx}: "
                        f"expected {(n_channels, n_samples)}, got {sample.shape}"
                    )
                split = str(row["split"])
                retained_records.append((split, row_idx, sample, row))
                kept_counts[split] += 1

            counts = dict(kept_counts)
            for split in ("train", "valid", "test"):
                group = h5f[split]
                increment = int(counts.get(split, 0))
                if increment == 0:
                    continue
                start = group["targets"].shape[0]
                end = start + increment
                group["data"].resize((end, n_channels, n_samples))
                group["targets"].resize((end,))

            print(
                f"[release {release}] usable windows: {len(retained_records)} / {len(metadata)} | "
                f"train={counts.get('train', 0)} valid={counts.get('valid', 0)} test={counts.get('test', 0)}"
            )

            for split, row_idx, sample, row in retained_records:
                split_index = write_positions[split]
                h5f[split]["data"][split_index] = sample
                h5f[split]["targets"][split_index] = np_mod.float32(row["target_float"])
                write_positions[split] += 1

                metadata.at[row_idx, "split_index"] = split_index

            for row_idx in range(len(metadata)):
                row = metadata.iloc[row_idx]
                writer.writerow(
                    {
                        "split": metadata_field(row, "split"),
                        "split_index": int(row["split_index"]),
                        "release": release,
                        "subject": metadata_field(row, "subject"),
                        "session": metadata_field(row, "session"),
                        "run": metadata_field(row, "run"),
                        "target": metadata_field(row, "target"),
                        args.target_field: float(row["target_float"]),
                        "stimulus_onset": metadata_field(row, "stimulus_onset"),
                        "response_onset": metadata_field(row, "response_onset"),
                    }
                )

            release_summaries[release] = {
                "candidate_windows": int(len(metadata)),
                "kept_windows": int(len(retained_records)),
                "rejected_windows": 0,
                "kept_split_counts": dict(kept_counts),
                "rejected_reason_counts": {},
                "recordings_total": int(release_stats.get("recordings_total", 0)),
            }

    split_manifest["release_stats"] = release_summaries
    split_manifest["total_kept_samples"] = dict(write_positions)
    split_json.write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")

    print("\nFinished writing dataset.")
    print(f"  train: {write_positions['train']} samples")
    print(f"  valid: {write_positions['valid']} samples")
    print(f"  test:  {write_positions['test']} samples")
    print(f"  h5:    {output_h5}")
    print(f"  meta:  {metadata_csv}")
    print(f"  split: {split_json}")


if __name__ == "__main__":
    main()
