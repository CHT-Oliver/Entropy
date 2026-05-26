from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
RECALL_THRESHOLDS = (0.3, 0.5, 0.7)


def default_qwen25vl_modeling_src() -> Path:
    env_path = os.environ.get("QWEN25VL_MODELING_SRC")
    if env_path:
        return Path(env_path)

    script_dir = Path(__file__).resolve().parent
    experiment_dir_modeling = script_dir / "modeling_qwen2_5_vl.py"
    project_root_modeling = script_dir.parent / "modeling_qwen2_5_vl.py"
    legacy_project_modeling = (
        script_dir
        / "qwen25vl_patched_src"
        / "transformers"
        / "models"
        / "qwen2_5_vl"
        / "modeling_qwen2_5_vl.py"
    )
    if experiment_dir_modeling.exists():
        return experiment_dir_modeling
    if project_root_modeling.exists():
        return project_root_modeling
    return legacy_project_modeling


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare uniform and entropy/logit-diff adaptive Qwen2.5-VL sampling.")
    parser.add_argument("--samples-jsonl", type=Path, default=None)
    parser.add_argument("--ckpt-dir", default="/ssd/cht/checkpoints/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--transformers-src", default="")
    parser.add_argument("--qwen25vl-modeling-src", type=Path, default=default_qwen25vl_modeling_src())
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--mode", choices=["baseline", "adaptive", "both"], default="both")
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--total-frames", type=int, default=32)
    parser.add_argument("--min-pixels", type=int, default=3136)
    parser.add_argument("--max-pixels", type=int, default=50176)
    parser.add_argument("--final-square-size", type=int, default=112)
    parser.add_argument("--probe-nframes", type=int, default=4)
    parser.add_argument("--probe-max-pixels", type=int, default=12544)
    parser.add_argument("--probe-square-size", type=int, default=56)
    parser.add_argument("--segment-len", type=float, default=8.0)
    parser.add_argument("--target-segments", type=int, default=8)
    parser.add_argument("--segment-mode", choices=["partition", "sliding"], default="partition")
    parser.add_argument("--min-frames-per-segment", type=int, default=1)
    parser.add_argument("--max-frames-per-segment", type=int, default=12)
    parser.add_argument("--score-alpha", type=float, default=0.7, help="Weight for normalized logit_diff.")
    parser.add_argument("--score-beta", type=float, default=0.3, help="Weight for normalized entropy.")
    parser.add_argument("--entropy-direction", choices=["high", "low"], default="high")
    parser.add_argument("--allocation-mode", choices=["continuous", "quadrant", "logit_low_entropy"], default="continuous")
    parser.add_argument("--logit-threshold", type=float, default=None)
    parser.add_argument("--entropy-threshold", type=float, default=None)
    parser.add_argument("--top-logit-frac", type=float, default=0.5)
    parser.add_argument("--low-entropy-frac", type=float, default=0.5)
    parser.add_argument("--min-selected-segments", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--length-rho", type=float, default=0.0)
    parser.add_argument("--output-jsonl", type=Path, default=Path("outputs/qwen25vl_adaptive_temporal/results.jsonl"))
    parser.add_argument("--summary-json", type=Path, default=Path("outputs/qwen25vl_adaptive_temporal/summary.json"))
    parser.add_argument("--save-details", action="store_true", help="Write per-query JSONL details. Default: save summary only.")
    parser.add_argument("--keep-temp", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.total_frames <= 0:
        raise ValueError("--total-frames must be positive.")
    if args.target_segments <= 0:
        raise ValueError("--target-segments must be positive.")
    if args.min_frames_per_segment < 0:
        raise ValueError("--min-frames-per-segment cannot be negative.")
    if args.max_frames_per_segment < args.min_frames_per_segment:
        raise ValueError("--max-frames-per-segment must be >= --min-frames-per-segment.")
    if args.total_frames > args.target_segments * args.max_frames_per_segment:
        raise ValueError("The frame budget exceeds segment capacity; increase --max-frames-per-segment.")
    if args.allocation_mode in {"quadrant", "logit_low_entropy"}:
        minimum_budget = args.target_segments * args.min_frames_per_segment
        if args.total_frames <= minimum_budget:
            raise ValueError(
                "Adaptive selected segments need extra budget: require "
                "--total-frames > --target-segments * --min-frames-per-segment."
            )
        if args.min_selected_segments < 1:
            raise ValueError("--min-selected-segments must be at least 1.")
    if not 0.0 < args.top_logit_frac <= 1.0:
        raise ValueError("--top-logit-frac must be in (0, 1].")
    if not 0.0 < args.low_entropy_frac <= 1.0:
        raise ValueError("--low-entropy-frac must be in (0, 1].")


def load_project_qwen25vl_modeling(modeling_src: Path | None) -> Any | None:
    if modeling_src is None:
        return None

    modeling_src = Path(modeling_src)
    if not modeling_src.exists():
        return None

    module_name = "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl"
    spec = importlib.util.spec_from_file_location(module_name, modeling_src)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Qwen2.5-VL modeling source from {modeling_src}.")

    import transformers.models.qwen2_5_vl as qwen25vl_package

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    setattr(qwen25vl_package, "modeling_qwen2_5_vl", module)
    for exported_name in (
        "Qwen2_5_VLForConditionalGeneration",
        "Qwen2_5_VLModel",
        "Qwen2_5_VLPreTrainedModel",
    ):
        if hasattr(module, exported_name):
            setattr(qwen25vl_package, exported_name, getattr(module, exported_name))
    print(f"[INFO] Loaded project-local Qwen2.5-VL modeling source: {modeling_src}")
    return module


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_single_token_id(tokenizer: Any, text: str) -> int:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(f"Expected {text!r} to be a single token, got ids={token_ids}")
    return int(token_ids[0])


def ffprobe_duration(video_path: str) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_path,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def adaptive_stride_for_target(duration: float, segment_len: float, target_segments: int) -> float:
    if target_segments <= 1 or duration <= segment_len:
        return max(segment_len, 1e-3)
    return max((duration - segment_len) / max(target_segments - 1, 1), 1e-3)


def sliding_windows(duration: float, segment_len: float, target_segments: int) -> list[tuple[float, float]]:
    if duration <= segment_len:
        return [(0.0, max(duration, 0.01))]
    stride = adaptive_stride_for_target(duration, segment_len, target_segments)
    windows: list[tuple[float, float]] = []
    start = 0.0
    while start + segment_len <= duration + 1e-6 and len(windows) < target_segments:
        windows.append((round(start, 3), round(min(start + segment_len, duration), 3)))
        start += stride
    if not windows or windows[-1][1] < duration - 1e-3:
        windows.append((round(max(0.0, duration - segment_len), 3), round(duration, 3)))
    return windows[:target_segments]


def partition_windows(duration: float, target_segments: int) -> list[tuple[float, float]]:
    if target_segments <= 1:
        return [(0.0, max(duration, 0.01))]
    duration = max(duration, 0.01)
    return [
        (
            round(idx * duration / target_segments, 3),
            round((idx + 1) * duration / target_segments, 3),
        )
        for idx in range(target_segments)
    ]


def build_segment_windows(duration: float, args: argparse.Namespace) -> list[tuple[float, float]]:
    if args.segment_mode == "partition":
        return partition_windows(duration, args.target_segments)
    return sliding_windows(duration, args.segment_len, args.target_segments)


def sample_uniform_times(start: float, end: float, count: int) -> list[float]:
    if count <= 0:
        return []
    span = max(end - start, 1e-3)
    return [start + (idx + 0.5) * span / count for idx in range(count)]


def frame_seek_attempts(timestamp: float) -> list[float]:
    candidates = [timestamp, timestamp - 0.2, timestamp - 1.0, timestamp - 2.0, 0.0]
    attempts: list[float] = []
    for value in candidates:
        value = max(float(value), 0.0)
        if not any(math.isclose(value, seen, abs_tol=1e-3) for seen in attempts):
            attempts.append(value)
    return attempts


def extract_frames_ffmpeg(video_path: str, timestamps: list[float], work_dir: Path) -> list[Image.Image]:
    frames: list[Image.Image] = []
    work_dir.mkdir(parents=True, exist_ok=True)
    for idx, timestamp in enumerate(timestamps):
        frame_path = work_dir / f"frame_{idx:04d}.jpg"
        last_error = ""
        for seek_time in frame_seek_attempts(timestamp):
            if frame_path.exists():
                frame_path.unlink()
            proc = subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-ss",
                    f"{seek_time:.3f}",
                    "-i",
                    video_path,
                    "-frames:v",
                    "1",
                    "-q:v",
                    "2",
                    str(frame_path),
                ],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            last_error = proc.stderr.strip()
            if proc.returncode == 0 and frame_path.exists() and frame_path.stat().st_size > 0:
                break
        if not frame_path.exists() or frame_path.stat().st_size == 0:
            raise FileNotFoundError(
                f"ffmpeg did not produce a frame for {video_path} at {timestamp:.3f}s. {last_error}"
            )
        frames.append(Image.open(frame_path).convert("RGB"))
    return frames


def letterbox_square(frame: Image.Image, size: int) -> Image.Image:
    width, height = frame.size
    scale = min(size / max(width, 1), size / max(height, 1))
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    resized = frame.resize(new_size, Image.BICUBIC)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    offset = ((size - new_size[0]) // 2, (size - new_size[1]) // 2)
    canvas.paste(resized, offset)
    return canvas


def resize_frames(frames: list[Image.Image], max_pixels: int, square_size: int = 0) -> list[Image.Image]:
    resized: list[Image.Image] = []
    for frame in frames:
        if square_size > 0:
            resized.append(letterbox_square(frame, square_size))
            continue
        width, height = frame.size
        pixels = max(width * height, 1)
        if pixels <= max_pixels:
            resized.append(frame)
            continue
        scale = math.sqrt(max_pixels / pixels)
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        resized.append(frame.resize(new_size, Image.BICUBIC))
    return resized


def build_temporal_prompt(query: str, duration: float | None) -> str:
    duration_text = f"The video duration is {duration:.1f} seconds.\n" if duration is not None else ""
    return (
        "Temporal grounding task.\n"
        f"{duration_text}"
        "Given the video and the query, output the temporal segment in seconds.\n"
        "The predicted start and end must be within the video duration.\n"
        "Output format STRICT:\n"
        "start: <number>, end: <number>\n"
        f"Query: {query}"
    )


def build_probe_prompt(query: str) -> str:
    return (
        f"Query: {query}\n\n"
        "Question: Does the query overlap with this video segment?\n\n"
        "A. Yes\n"
        "B. No\n\n"
        "Output one letter only: A or B."
    )


def build_frame_inputs(
    processor: Any,
    prompt: str,
    frames: list[Image.Image],
    fps: float,
    min_pixels: int,
    max_pixels: int,
):
    messages = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return processor(
        text=[text],
        videos=[frames],
        fps=fps,
        min_pixels=min_pixels,
        max_pixels=max(max_pixels, min_pixels),
        padding=True,
        return_tensors="pt",
    )


def build_grid_timestamps(frame_timestamps: list[float], grid_t: int, temporal_patch_size: int) -> list[float]:
    if grid_t <= 0:
        return []
    timestamps = list(frame_timestamps) or [0.0]
    while len(timestamps) < grid_t * temporal_patch_size:
        timestamps.append(timestamps[-1])
    grid_ts = []
    for grid_idx in range(grid_t):
        chunk = timestamps[grid_idx * temporal_patch_size : (grid_idx + 1) * temporal_patch_size]
        grid_ts.append(float(sum(chunk) / max(len(chunk), 1)))
    return grid_ts


def get_temporal_patch_size(processor: Any) -> int:
    video_processor = getattr(processor, "video_processor", None)
    if video_processor is not None and hasattr(video_processor, "temporal_patch_size"):
        return int(video_processor.temporal_patch_size)
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is not None and hasattr(image_processor, "temporal_patch_size"):
        return int(image_processor.temporal_patch_size)
    return 2


def get_video_grid_thw(inputs: Any) -> list[list[int]]:
    grid = inputs.get("video_grid_thw", None)
    if grid is None:
        return []
    return [[int(value) for value in row] for row in grid.detach().cpu().tolist()]


def get_video_llm_token_count(inputs: Any, model: Any) -> int:
    spatial_merge_size = int(getattr(model.config.vision_config, "spatial_merge_size", 2))
    token_count = 0
    for t, h, w in get_video_grid_thw(inputs):
        token_count += int(t) * (int(h) // spatial_merge_size) * (int(w) // spatial_merge_size)
    return int(token_count)


def assert_count(label: str, actual: int, expected: int) -> None:
    if actual != expected:
        raise ValueError(f"{label} count mismatch: expected {expected}, got {actual}.")


def validate_pair_budget(baseline: dict[str, Any], adaptive: dict[str, Any]) -> None:
    if baseline["frame_count"] != adaptive["frame_count"]:
        raise ValueError(
            f"Frame budget mismatch: baseline={baseline['frame_count']} adaptive={adaptive['frame_count']}."
        )
    if baseline["video_grid_thw"] != adaptive["video_grid_thw"]:
        raise ValueError(
            f"video_grid_thw mismatch: baseline={baseline['video_grid_thw']} adaptive={adaptive['video_grid_thw']}."
        )
    if baseline["video_llm_tokens"] != adaptive["video_llm_tokens"]:
        raise ValueError(
            "Video token budget mismatch: "
            f"baseline={baseline['video_llm_tokens']} adaptive={adaptive['video_llm_tokens']}."
        )


def minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if math.isclose(lo, hi):
        return [0.0 for _ in values]
    return [(value - lo) / (hi - lo) for value in values]


def allocate_frame_counts(
    windows: list[tuple[float, float]],
    scores: list[float],
    total_frames: int,
    min_frames: int,
    max_frames: int,
    temperature: float,
    length_rho: float,
) -> list[int]:
    if not windows:
        return []
    n = len(windows)
    base = min_frames if total_frames >= n * min_frames else 0
    counts = [base for _ in windows]
    remaining = max(total_frames - sum(counts), 0)
    if remaining == 0:
        return counts

    score_arr = np.asarray(scores, dtype=np.float64)
    lengths = np.asarray([max(end - start, 1e-6) for start, end in windows], dtype=np.float64)
    logits = score_arr / max(temperature, 1e-6)
    if length_rho:
        logits = logits + length_rho * np.log(lengths)
    logits = logits - logits.max()
    weights = np.exp(logits)
    weights = weights / max(weights.sum(), 1e-12)

    raw_extra = weights * remaining
    extras = np.floor(raw_extra).astype(int)
    for idx, extra in enumerate(extras):
        counts[idx] += int(extra)
    leftover = total_frames - sum(counts)
    order = np.argsort(-(raw_extra - extras))
    for idx in order:
        if leftover <= 0:
            break
        if counts[idx] < max_frames:
            counts[idx] += 1
            leftover -= 1

    while sum(counts) > total_frames:
        candidates = [i for i in range(n) if counts[i] > base]
        if not candidates:
            break
        idx = min(candidates, key=lambda i: scores[i])
        counts[idx] -= 1
    while sum(counts) < total_frames:
        candidates = [i for i in range(n) if counts[i] < max_frames]
        if not candidates:
            break
        idx = max(candidates, key=lambda i: scores[i])
        counts[idx] += 1
    return counts


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.median(np.asarray(values, dtype=np.float64)))


def allocate_quadrant_frame_counts(
    windows: list[tuple[float, float]],
    scores: list[float],
    selected: list[bool],
    total_frames: int,
    min_frames: int,
    max_frames: int,
) -> tuple[list[int], int]:
    if not windows:
        return [], 0

    n = len(windows)
    base = min_frames if total_frames >= n * min_frames else 0
    counts = [base for _ in windows]
    remaining = max(total_frames - sum(counts), 0)
    selected_indices = [idx for idx, keep in enumerate(selected) if keep]
    if remaining == 0 or not selected_indices:
        return counts, remaining

    capacities = np.asarray([max(0, max_frames - counts[idx]) for idx in selected_indices], dtype=np.int64)
    allocatable = int(min(remaining, capacities.sum()))
    if allocatable <= 0:
        return counts, remaining

    selected_scores = np.asarray([max(scores[idx], 0.0) for idx in selected_indices], dtype=np.float64)
    if selected_scores.sum() <= 1e-12:
        weights = np.ones_like(selected_scores) / len(selected_scores)
    else:
        weights = selected_scores / selected_scores.sum()

    raw_extra = weights * allocatable
    extras = np.minimum(np.floor(raw_extra).astype(np.int64), capacities)
    for idx, extra in zip(selected_indices, extras):
        counts[idx] += int(extra)

    leftover = allocatable - int(extras.sum())
    order = np.argsort(-(raw_extra - extras))
    while leftover > 0:
        progressed = False
        for order_idx in order:
            idx = selected_indices[int(order_idx)]
            if counts[idx] < max_frames:
                counts[idx] += 1
                leftover -= 1
                progressed = True
                if leftover == 0:
                    break
        if not progressed:
            break

    # If selected segments hit max_frames before the global budget is exhausted,
    # fill the remaining budget across any segment with spare capacity. This
    # preserves "selected first" while keeping baseline/adaptive token budgets equal.
    unused_after_selected = total_frames - sum(counts)
    while unused_after_selected > 0:
        candidates = [idx for idx in range(n) if counts[idx] < max_frames]
        if not candidates:
            break
        idx = max(candidates, key=lambda i: (selected[i], scores[i]))
        counts[idx] += 1
        unused_after_selected -= 1

    unused = total_frames - sum(counts)
    return counts, max(unused, 0)


def count_by_fraction(total: int, frac: float, minimum: int) -> int:
    if total <= 0:
        return 0
    return max(1, min(total, max(minimum, int(math.ceil(total * frac)))))


def select_logit_low_entropy_segments(
    metrics: list[dict[str, float]],
    top_logit_frac: float,
    low_entropy_frac: float,
    min_selected_segments: int,
) -> tuple[list[bool], list[float], dict[str, Any]]:
    n = len(metrics)
    if n == 0:
        return [], [], {"top_logit_indices": [], "selected_indices": []}

    logit_values = [metric["logit_diff"] for metric in metrics]
    entropy_values = [metric["normalized_entropy"] for metric in metrics]
    logit_norm = minmax(logit_values)
    entropy_norm = minmax(entropy_values)

    top_k = count_by_fraction(n, top_logit_frac, min_selected_segments)
    top_logit_indices = sorted(range(n), key=lambda idx: logit_values[idx], reverse=True)[:top_k]

    low_k = count_by_fraction(len(top_logit_indices), low_entropy_frac, min_selected_segments)
    selected_indices = sorted(top_logit_indices, key=lambda idx: entropy_values[idx])[:low_k]
    selected = [idx in set(selected_indices) for idx in range(n)]
    scores = [
        (logit_norm[idx] + (1.0 - entropy_norm[idx])) if selected[idx] else 0.0
        for idx in range(n)
    ]
    debug = {
        "top_logit_indices": top_logit_indices,
        "selected_indices": selected_indices,
        "top_logit_frac": top_logit_frac,
        "low_entropy_frac": low_entropy_frac,
    }
    return selected, scores, debug


def probe_segment(
    model: Any,
    processor: Any,
    video_path: str,
    query: str,
    start: float,
    end: float,
    token_a_id: int,
    token_b_id: int,
    args: argparse.Namespace,
    work_dir: Path,
) -> dict[str, float]:
    timestamps = sample_uniform_times(start, end, args.probe_nframes)
    frames = extract_frames_ffmpeg(video_path, timestamps, work_dir)
    assert_count("probe frame", len(frames), args.probe_nframes)
    frames = resize_frames(frames, args.probe_max_pixels, args.probe_square_size)
    fps = max(args.probe_nframes / max(end - start, 1e-6), 1e-6)
    inputs = build_frame_inputs(
        processor,
        build_probe_prompt(query),
        frames,
        fps=fps,
        min_pixels=args.min_pixels,
        max_pixels=args.probe_max_pixels,
    ).to(model.device)

    with torch.inference_mode():
        outputs = model(**inputs)

    next_logits = outputs.logits[0, -1, :].detach().float()
    ab_logits = torch.stack([next_logits[token_a_id], next_logits[token_b_id]])
    ab_probs = torch.softmax(ab_logits, dim=0)
    vocab_probs = torch.softmax(next_logits, dim=0)
    entropy = float((-(vocab_probs * torch.log(vocab_probs.clamp_min(1e-12))).sum()).item())
    normalized_entropy = entropy / math.log(float(next_logits.numel()))
    return {
        "logit_A": float(ab_logits[0].item()),
        "logit_B": float(ab_logits[1].item()),
        "logit_diff": float((ab_logits[0] - ab_logits[1]).item()),
        "p_yes": float(ab_probs[0].item()),
        "entropy": entropy,
        "normalized_entropy": normalized_entropy,
    }


def score_segments(
    model: Any,
    processor: Any,
    sample: dict[str, Any],
    duration: float,
    token_a_id: int,
    token_b_id: int,
    args: argparse.Namespace,
    work_dir: Path,
) -> tuple[list[tuple[float, float]], list[dict[str, float]], list[float]]:
    windows = build_segment_windows(duration, args)
    metrics = []
    for seg_idx, (start, end) in enumerate(windows):
        metrics.append(
            probe_segment(
                model,
                processor,
                sample["video_path"],
                sample["query"],
                start,
                end,
                token_a_id,
                token_b_id,
                args,
                work_dir / f"probe_{seg_idx:03d}",
            )
        )

    logit_values = [m["logit_diff"] for m in metrics]
    entropy_values = [m["normalized_entropy"] for m in metrics]
    logit_norm = minmax(logit_values)
    entropy_norm_raw = minmax(entropy_values)
    logit_threshold = args.logit_threshold if args.logit_threshold is not None else median(logit_values)
    entropy_threshold = args.entropy_threshold if args.entropy_threshold is not None else median(entropy_values)

    selection_debug: dict[str, Any] = {}
    if args.allocation_mode == "quadrant":
        selected = [
            logit_value >= logit_threshold and entropy_value < entropy_threshold
            for logit_value, entropy_value in zip(logit_values, entropy_values)
        ]
        scores = [
            (logit_score + (1.0 - entropy_score)) if keep else 0.0
            for logit_score, entropy_score, keep in zip(logit_norm, entropy_norm_raw, selected)
        ]
    elif args.allocation_mode == "logit_low_entropy":
        selected, scores, selection_debug = select_logit_low_entropy_segments(
            metrics,
            args.top_logit_frac,
            args.low_entropy_frac,
            args.min_selected_segments,
        )
    else:
        entropy_norm = entropy_norm_raw
        if args.entropy_direction == "low":
            entropy_norm = [1.0 - value for value in entropy_norm]
        selected = [False for _ in metrics]
        scores = [
            args.score_alpha * logit_score + args.score_beta * entropy_score
            for logit_score, entropy_score in zip(logit_norm, entropy_norm)
        ]

    for metric, score, keep in zip(metrics, scores, selected):
        metric["adaptive_score"] = float(score)
        metric["quadrant_selected"] = bool(keep)
        metric["logit_low_entropy_selected"] = bool(keep) if args.allocation_mode == "logit_low_entropy" else False
        metric["logit_threshold"] = float(logit_threshold)
        metric["entropy_threshold"] = float(entropy_threshold)
        metric["selection_debug"] = selection_debug
    return windows, metrics, scores


def parse_pred_window(text: str) -> tuple[float, float] | None:
    text = (text or "").strip()
    if not text:
        return None
    m1 = re.search(r"start\s*[:=]\s*([-+]?\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    m2 = re.search(r"end\s*[:=]\s*([-+]?\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    if m1 and m2:
        return float(m1.group(1)), float(m2.group(1))
    nums = NUM_RE.findall(text)
    if len(nums) >= 2:
        return float(nums[0]), float(nums[1])
    return None


def sanitize_window(start: float, end: float, duration: float) -> tuple[float, float]:
    if start > end:
        start, end = end, start
    return max(0.0, min(start, duration)), max(0.0, min(end, duration))


def iou_1d(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(1e-12, (a1 - a0) + (b1 - b0) - inter)
    return inter / union


def generate_from_inputs(model: Any, processor: Any, inputs: Any, args: argparse.Namespace) -> str:
    inputs = inputs.to(model.device)
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            num_beams=1,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    gen_ids = output_ids[0][inputs.input_ids.shape[1] :]
    return processor.decode(gen_ids, skip_special_tokens=True)


def run_uniform(
    model: Any,
    processor: Any,
    sample: dict[str, Any],
    duration: float,
    args: argparse.Namespace,
    work_dir: Path,
) -> dict[str, Any]:
    timestamps = sample_uniform_times(0.0, duration, args.total_frames)
    assert_count("baseline timestamp", len(timestamps), args.total_frames)
    frames = extract_frames_ffmpeg(sample["video_path"], timestamps, work_dir / "baseline_frames")
    assert_count("baseline frame", len(frames), args.total_frames)
    frames = resize_frames(frames, args.max_pixels, args.final_square_size)
    fps = max(args.total_frames / max(duration, 1e-6), 1e-6)
    inputs = build_frame_inputs(
        processor,
        build_temporal_prompt(sample["query"], duration),
        frames,
        fps=fps,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    video_grid_thw = get_video_grid_thw(inputs)
    video_llm_tokens = get_video_llm_token_count(inputs, model)
    text = generate_from_inputs(model, processor, inputs, args)
    return {
        "text": text,
        "frame_timestamps": timestamps,
        "frame_count": len(frames),
        "video_grid_thw": video_grid_thw,
        "video_llm_tokens": video_llm_tokens,
    }


def run_adaptive(
    model: Any,
    processor: Any,
    sample: dict[str, Any],
    duration: float,
    token_a_id: int,
    token_b_id: int,
    args: argparse.Namespace,
    work_dir: Path,
) -> dict[str, Any]:
    windows, probe_metrics, scores = score_segments(
        model, processor, sample, duration, token_a_id, token_b_id, args, work_dir
    )
    if args.allocation_mode in {"quadrant", "logit_low_entropy"}:
        selected_key = "logit_low_entropy_selected" if args.allocation_mode == "logit_low_entropy" else "quadrant_selected"
        selected = [bool(metric.get(selected_key)) for metric in probe_metrics]
        counts, unused_frame_budget = allocate_quadrant_frame_counts(
            windows,
            scores,
            selected,
            args.total_frames,
            args.min_frames_per_segment,
            args.max_frames_per_segment,
        )
        required_selected = min(max(args.min_selected_segments, 1), len(windows))
        if sum(1 for keep in selected if keep) < required_selected:
            raise ValueError(f"Adaptive selection picked too few segments: selected={selected}.")
        if not any(keep and count > args.min_frames_per_segment for keep, count in zip(selected, counts)):
            raise ValueError(f"No selected segment received extra frame budget: selected={selected}, counts={counts}.")
    else:
        selected = [False for _ in probe_metrics]
        counts = allocate_frame_counts(
            windows,
            scores,
            args.total_frames,
            args.min_frames_per_segment,
            args.max_frames_per_segment,
            args.temperature,
            args.length_rho,
        )
        unused_frame_budget = max(args.total_frames - sum(counts), 0)
    if len(counts) != len(windows):
        raise ValueError(f"Frame-count/window mismatch: counts={len(counts)} windows={len(windows)}.")
    if sum(counts) != args.total_frames:
        raise ValueError(f"Adaptive frame budget not fully used: counts={counts}, total={sum(counts)}.")
    if unused_frame_budget != 0:
        raise ValueError(f"Adaptive frame budget has unused frames: {unused_frame_budget}.")
    timestamps: list[float] = []
    for (start, end), count in zip(windows, counts):
        timestamps.extend(sample_uniform_times(start, end, count))
    assert_count("adaptive timestamp", len(timestamps), args.total_frames)
    timestamps = sorted(timestamps)
    frames = extract_frames_ffmpeg(sample["video_path"], timestamps, work_dir / "adaptive_frames")
    assert_count("adaptive frame", len(frames), args.total_frames)
    frames = resize_frames(frames, args.max_pixels, args.final_square_size)
    fps = max(args.total_frames / max(duration, 1e-6), 1e-6)
    inputs = build_frame_inputs(
        processor,
        build_temporal_prompt(sample["query"], duration),
        frames,
        fps=fps,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )

    temporal_patch_size = get_temporal_patch_size(processor)
    grid_t = int(inputs["video_grid_thw"][0][0].item())
    grid_ts = build_grid_timestamps(timestamps, grid_t, temporal_patch_size)
    assert_count("adaptive video_time_grid_ts", len(grid_ts), grid_t)
    inputs["video_time_grid_ts"] = torch.tensor([grid_ts], dtype=torch.float32)
    video_grid_thw = get_video_grid_thw(inputs)
    video_llm_tokens = get_video_llm_token_count(inputs, model)
    text = generate_from_inputs(model, processor, inputs, args)
    return {
        "text": text,
        "windows": [[float(s), float(e)] for s, e in windows],
        "frame_counts": counts,
        "quadrant_selected": selected,
        "unused_frame_budget": unused_frame_budget,
        "frame_timestamps": timestamps,
        "grid_timestamps": grid_ts,
        "frame_count": len(frames),
        "video_grid_thw": video_grid_thw,
        "video_llm_tokens": video_llm_tokens,
        "probe_metrics": probe_metrics,
    }


def attach_metrics(result: dict[str, Any], prefix: str, duration: float, gt_start: float, gt_end: float) -> None:
    parsed = parse_pred_window(str(result.get(f"{prefix}_text", "")))
    result[f"{prefix}_pred_window"] = parsed
    if parsed is None:
        result[f"{prefix}_iou"] = None
        return
    pred_start, pred_end = sanitize_window(parsed[0], parsed[1], duration)
    result[f"{prefix}_pred_window"] = [pred_start, pred_end]
    result[f"{prefix}_iou"] = iou_1d(pred_start, pred_end, gt_start, gt_end)


def summarize(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    ious = [float(row[f"{prefix}_iou"]) for row in rows if row.get(f"{prefix}_iou") is not None]
    return {
        "valid_predictions": len(ious),
        "mean_iou": sum(ious) / len(ious) if ious else None,
        **{f"recall@{thr}": sum(iou >= thr for iou in ious) / len(ious) if ious else None for thr in RECALL_THRESHOLDS},
    }


def summarize_all_used(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    ious = [float(row.get(f"{prefix}_iou") or 0.0) for row in rows if "error" not in row]
    return {
        "samples": len(ious),
        "mean_iou": sum(ious) / len(ious) if ious else None,
        **{f"recall@{thr}": sum(iou >= thr for iou in ious) / len(ious) if ious else None for thr in RECALL_THRESHOLDS},
    }


def summarize_final_metrics(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    ious = [float(row.get(f"{prefix}_iou") or 0.0) for row in rows if "error" not in row]
    return {
        "samples": len(ious),
        "mIoU": sum(ious) / len(ious) if ious else None,
        **{f"Recall@1_IoU={thr}": sum(iou >= thr for iou in ious) / len(ious) if ious else None for thr in RECALL_THRESHOLDS},
    }


def summarize_paired_valid(rows: list[dict[str, Any]]) -> dict[str, Any]:
    paired_rows = [
        row for row in rows if row.get("baseline_iou") is not None and row.get("adaptive_iou") is not None
    ]

    def summarize_values(values: list[float]) -> dict[str, Any]:
        return {
            "samples": len(values),
            "mean_iou": sum(values) / len(values) if values else None,
            **{
                f"recall@{thr}": sum(value >= thr for value in values) / len(values) if values else None
                for thr in RECALL_THRESHOLDS
            },
        }

    return {
        "paired_samples": len(paired_rows),
        "baseline": summarize_values([float(row["baseline_iou"]) for row in paired_rows]),
        "adaptive": summarize_values([float(row["adaptive_iou"]) for row in paired_rows]),
    }


def int_hist(values: list[int]) -> dict[str, int]:
    hist: dict[str, int] = {}
    for value in values:
        key = str(int(value))
        hist[key] = hist.get(key, 0) + 1
    return dict(sorted(hist.items(), key=lambda item: int(item[0])))


def summarize_budget_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    baseline_stats = [row.get("baseline_stats") for row in rows if isinstance(row.get("baseline_stats"), dict)]
    adaptive_stats = [row.get("adaptive_stats") for row in rows if isinstance(row.get("adaptive_stats"), dict)]
    pair_rows = [
        row
        for row in rows
        if isinstance(row.get("baseline_stats"), dict) and isinstance(row.get("adaptive_stats"), dict)
    ]
    return {
        "baseline_video_llm_tokens_hist": int_hist([int(stat["video_llm_tokens"]) for stat in baseline_stats]),
        "adaptive_video_llm_tokens_hist": int_hist([int(stat["video_llm_tokens"]) for stat in adaptive_stats]),
        "adaptive_frame_sum_hist": int_hist([sum(int(v) for v in stat["frame_counts"]) for stat in adaptive_stats]),
        "adaptive_selected_count_hist": int_hist(
            [sum(1 for keep in stat["quadrant_selected"] if keep) for stat in adaptive_stats]
        ),
        "adaptive_unused_budget_hist": int_hist([int(stat["unused_frame_budget"]) for stat in adaptive_stats]),
        "pair_token_mismatch_count": sum(
            1
            for row in pair_rows
            if int(row["baseline_stats"]["video_llm_tokens"]) != int(row["adaptive_stats"]["video_llm_tokens"])
        ),
        "pair_grid_mismatch_count": sum(
            1
            for row in pair_rows
            if row["baseline_stats"]["video_grid_thw"] != row["adaptive_stats"]["video_grid_thw"]
        ),
        "zero_selected_count": sum(
            1 for stat in adaptive_stats if sum(1 for keep in stat["quadrant_selected"] if keep) == 0
        ),
        "unused_budget_count": sum(1 for stat in adaptive_stats if int(stat["unused_frame_budget"]) != 0),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_validation(
    args: argparse.Namespace,
    samples: list[dict[str, Any]] | None = None,
    sample_source: str | None = None,
) -> dict[str, Any]:
    validate_args(args)
    set_seed(args.seed)
    if args.transformers_src and os.path.isdir(args.transformers_src):
        sys.path.insert(0, args.transformers_src)

    qwen25vl_module = load_project_qwen25vl_modeling(args.qwen25vl_modeling_src)

    from transformers import AutoProcessor

    if qwen25vl_module is not None:
        Qwen2_5_VLForConditionalGeneration = qwen25vl_module.Qwen2_5_VLForConditionalGeneration
    else:
        from transformers import Qwen2_5_VLForConditionalGeneration

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.ckpt_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=args.attn_implementation,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(args.ckpt_dir)
    token_a_id = get_single_token_id(processor.tokenizer, "A")
    token_b_id = get_single_token_id(processor.tokenizer, "B")

    if samples is None:
        if args.samples_jsonl is None:
            raise ValueError("Either pass --samples-jsonl or call run_validation(..., samples=...).")
        samples = read_jsonl(args.samples_jsonl)
    sample_offset = max(0, int(getattr(args, "sample_offset", 0) or 0))
    if sample_offset:
        samples = samples[sample_offset:]
    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    rows: list[dict[str, Any]] = []
    failures = missing_video = used = 0
    temp_root = Path(tempfile.mkdtemp(prefix="qwen25vl_adaptive_temporal_"))
    progress_name = sample_source or (args.samples_jsonl.stem if args.samples_jsonl is not None else "samples")
    pbar = tqdm(samples, desc=progress_name, ncols=120)
    try:
        for idx, sample in enumerate(pbar):
            video_path = str(sample.get("video_path") or "")
            if not video_path or not os.path.exists(video_path):
                missing_video += 1
                continue
            sample_dir = temp_root / f"sample_{idx:04d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            result = {
                "sample_id": sample.get("sample_id"),
                "video_id": sample.get("video_id"),
                "video_path": video_path,
                "query": sample.get("query"),
                "gt_start": float(sample["gt_start"]),
                "gt_end": float(sample["gt_end"]),
            }
            try:
                duration = ffprobe_duration(video_path)
                result["duration"] = duration
                baseline: dict[str, Any] | None = None
                adaptive: dict[str, Any] | None = None
                if args.mode in {"baseline", "both"}:
                    baseline = run_uniform(model, processor, sample, duration, args, sample_dir)
                    result["baseline_text"] = baseline["text"]
                    result["baseline_frame_timestamps"] = baseline["frame_timestamps"]
                    result["baseline_stats"] = {k: v for k, v in baseline.items() if k != "text"}
                    attach_metrics(result, "baseline", duration, result["gt_start"], result["gt_end"])
                if args.mode in {"adaptive", "both"}:
                    adaptive = run_adaptive(
                        model,
                        processor,
                        sample,
                        duration,
                        token_a_id,
                        token_b_id,
                        args,
                        sample_dir,
                    )
                    result["adaptive_text"] = adaptive["text"]
                    result["adaptive_stats"] = {k: v for k, v in adaptive.items() if k != "text"}
                    attach_metrics(result, "adaptive", duration, result["gt_start"], result["gt_end"])
                if baseline is not None and adaptive is not None:
                    validate_pair_budget(baseline, adaptive)
            except Exception as exc:
                failures += 1
                result["error"] = repr(exc)
                print(f"[FAIL] sample={sample.get('sample_id')} err={exc!r}")
                rows.append(result)
                continue

            used += 1
            rows.append(result)
            postfix = {"used": used, "miss": missing_video, "fail": failures}
            if args.mode in {"baseline", "both"}:
                postfix["base_mIoU"] = f"{summarize(rows, 'baseline')['mean_iou'] or 0.0:.4f}"
            if args.mode in {"adaptive", "both"}:
                postfix["adapt_mIoU"] = f"{summarize(rows, 'adaptive')['mean_iou'] or 0.0:.4f}"
            pbar.set_postfix(postfix)
    finally:
        if args.keep_temp:
            print(f"[INFO] Kept temp dir: {temp_root}")
        else:
            shutil.rmtree(temp_root, ignore_errors=True)

    summary = {
        "sample_source": sample_source,
        "sample_offset": sample_offset,
        "used_samples": used,
        "missing_videos": missing_video,
        "failures": failures,
        "metric_denominator": "all successfully processed queries; unparsable predictions count as IoU 0",
        "baseline": summarize_final_metrics(rows, "baseline") if args.mode in {"baseline", "both"} else None,
        "adaptive": summarize_final_metrics(rows, "adaptive") if args.mode in {"adaptive", "both"} else None,
    }
    if args.save_details:
        write_jsonl(args.output_jsonl, rows)
        summary["details_jsonl"] = str(args.output_jsonl)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    args = parse_args()
    run_validation(args)


if __name__ == "__main__":
    main()
