from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

import validate as core
import validate_activitynet_captions as activitynet
import validate_charades_sta as charades


DEFAULT_OUTPUT_JSONL = Path("outputs/qwen25vl_gradient_vtg/gradient_vtg_details.jsonl")
DEFAULT_SUMMARY_JSON = Path("outputs/qwen25vl_gradient_vtg/gradient_vtg_summary.json")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Direct VTG from one full-video Qwen2.5-VL forward plus entropy/logit gradient backprop."
    )
    parser.add_argument("--dataset", choices=["charades_sta", "activitynet_captions"], default="charades_sta")
    parser.add_argument("--annotation", action="append", type=Path, default=None)
    parser.add_argument("--video-root", type=Path, default=None)
    parser.add_argument("--num-queries", type=int, default=10)
    parser.add_argument("--ckpt-dir", default="/ssd/cht/checkpoints/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--qwen25vl-modeling-src", type=Path, default=core.default_qwen25vl_modeling_src())
    parser.add_argument("--transformers-src", default="")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--query-offset", type=int, default=0)

    parser.add_argument("--total-frames", type=int, default=64)
    parser.add_argument("--min-pixels", type=int, default=3136)
    parser.add_argument("--max-pixels", type=int, default=12544)
    parser.add_argument("--final-square-size", type=int, default=112)

    parser.add_argument("--aggregation", choices=["mean", "max", "topk_mean"], default="topk_mean")
    parser.add_argument("--topk-frac", type=float, default=0.25)
    parser.add_argument("--smooth-radius", type=int, default=1)
    parser.add_argument(
        "--window-method",
        choices=["peak_fixed", "threshold_expand", "connected_component"],
        default="connected_component",
    )
    parser.add_argument("--threshold-mode", choices=["elbow", "mean", "median", "max_frac"], default="elbow")
    parser.add_argument("--threshold-frac", type=float, default=0.5)
    parser.add_argument("--candidate-diff-frac", type=float, default=0.95)
    parser.add_argument("--fixed-window-bins", type=int, default=8)
    parser.add_argument("--min-window-bins", type=int, default=1)
    parser.add_argument("--max-window-bins", type=int, default=0)

    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--save-details", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keep-temp", action="store_true")
    return parser.parse_args(argv)


def load_samples(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.dataset == "charades_sta":
        annotation = args.annotation[0] if args.annotation else charades.DEFAULT_ANNOTATION
        video_root = args.video_root or charades.DEFAULT_VIDEO_ROOT
        return charades.build_samples(annotation, video_root)

    annotations = args.annotation or activitynet.DEFAULT_ANNOTATIONS
    video_root = args.video_root or activitynet.DEFAULT_VIDEO_ROOT
    return activitynet.build_samples(annotations, video_root)


def load_model_and_processor(args: argparse.Namespace) -> tuple[Any, Any, int, int]:
    if args.transformers_src and os.path.isdir(args.transformers_src):
        sys.path.insert(0, args.transformers_src)

    qwen25vl_module = core.load_project_qwen25vl_modeling(args.qwen25vl_modeling_src)

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
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    processor = AutoProcessor.from_pretrained(args.ckpt_dir)
    token_a_id = core.get_single_token_id(processor.tokenizer, "A")
    token_b_id = core.get_single_token_id(processor.tokenizer, "B")
    return model, processor, token_a_id, token_b_id


def model_input_device(model: Any) -> torch.device:
    device = getattr(model, "device", None)
    if device is not None:
        return torch.device(device)
    return next(model.parameters()).device


def smooth_values(values: list[float], radius: int) -> list[float]:
    if radius <= 0 or len(values) <= 1:
        return [float(value) for value in values]
    arr = np.asarray(values, dtype=np.float64)
    smoothed = []
    for idx in range(len(arr)):
        left = max(0, idx - radius)
        right = min(len(arr), idx + radius + 1)
        smoothed.append(float(arr[left:right].mean()))
    return smoothed


def elbow_threshold(values: list[float]) -> float:
    arr = np.sort(np.asarray(values, dtype=np.float64))
    if arr.size == 0:
        return 0.0
    if arr.size < 3 or np.isclose(arr[0], arr[-1]):
        return float(arr.mean())
    x = np.linspace(0.0, 1.0, arr.size)
    y = (arr - arr[0]) / max(arr[-1] - arr[0], 1e-12)
    idx = int(np.argmax(np.abs(y - x)))
    return float(arr[idx])


def value_threshold(values: list[float], mode: str, frac: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    if mode == "elbow":
        return elbow_threshold(values)
    if mode == "mean":
        return float(arr.mean())
    if mode == "median":
        return float(np.median(arr))
    if mode == "max_frac":
        return float(arr.max() * frac)
    raise ValueError(f"Unknown threshold mode: {mode}")


def connected_components(mask: list[bool]) -> list[tuple[int, int]]:
    components: list[tuple[int, int]] = []
    start: int | None = None
    for idx, keep in enumerate(mask):
        if keep and start is None:
            start = idx
        elif not keep and start is not None:
            components.append((start, idx - 1))
            start = None
    if start is not None:
        components.append((start, len(mask) - 1))
    return components


def clamp_window_length(left: int, right: int, scores: list[float], min_len: int, max_len: int) -> tuple[int, int]:
    n = len(scores)
    left = max(0, min(left, n - 1))
    right = max(left, min(right, n - 1))
    min_len = max(1, min_len)

    while right - left + 1 < min_len:
        can_left = left > 0
        can_right = right < n - 1
        if not can_left and not can_right:
            break
        if can_left and (not can_right or scores[left - 1] >= scores[right + 1]):
            left -= 1
        else:
            right += 1

    if max_len > 0:
        while right - left + 1 > max_len:
            if scores[left] <= scores[right]:
                left += 1
            else:
                right -= 1
    return left, right


def fixed_peak_window(scores: list[float], fixed_bins: int) -> tuple[int, int]:
    n = len(scores)
    peak = int(np.argmax(np.asarray(scores, dtype=np.float64)))
    width = max(1, min(fixed_bins, n))
    left = peak - width // 2
    right = left + width - 1
    if left < 0:
        right -= left
        left = 0
    if right >= n:
        left -= right - n + 1
        right = n - 1
    return max(0, left), min(n - 1, right)


def threshold_expand_window(scores: list[float], threshold: float) -> tuple[int, int]:
    peak = int(np.argmax(np.asarray(scores, dtype=np.float64)))
    left = right = peak
    while left > 0 and scores[left - 1] >= threshold:
        left -= 1
    while right < len(scores) - 1 and scores[right + 1] >= threshold:
        right += 1
    return left, right


def connected_component_window(
    diff_scores: list[float],
    entropy_scores: list[float],
    threshold: float,
    candidate_diff_frac: float,
) -> tuple[int, int, dict[str, Any]]:
    mask = [score >= threshold for score in diff_scores]
    components = connected_components(mask)
    if not components:
        peak = int(np.argmax(np.asarray(diff_scores, dtype=np.float64)))
        return peak, peak, {"components": [], "fallback": "top_diff_peak"}

    stats = []
    for left, right in components:
        indices = list(range(left, right + 1))
        diff_sum = float(sum(diff_scores[idx] for idx in indices))
        entropy_sum = float(sum(entropy_scores[idx] for idx in indices))
        stats.append(
            {
                "left": left,
                "right": right,
                "length": right - left + 1,
                "diff_sum": diff_sum,
                "entropy_sum": entropy_sum,
            }
        )

    best_diff = max(item["diff_sum"] for item in stats)
    margin = max(0.0, min(candidate_diff_frac, 1.0))
    candidates = [item for item in stats if item["diff_sum"] >= best_diff * margin]
    chosen = max(candidates, key=lambda item: (item["entropy_sum"], item["diff_sum"], -item["length"]))
    return int(chosen["left"]), int(chosen["right"]), {
        "components": stats,
        "candidate_diff_frac": margin,
        "candidate_count": len(candidates),
        "chosen_component": chosen,
    }


def select_temporal_window(
    diff_scores: list[float],
    entropy_scores: list[float],
    args: argparse.Namespace,
) -> tuple[int, int, dict[str, Any]]:
    if not diff_scores:
        raise ValueError("No temporal gradient scores were produced.")
    if len(diff_scores) != len(entropy_scores):
        raise ValueError("diff_scores and entropy_scores must have the same length.")

    smoothed_diff = smooth_values(diff_scores, args.smooth_radius)
    smoothed_entropy = smooth_values(entropy_scores, args.smooth_radius)
    threshold = value_threshold(smoothed_diff, args.threshold_mode, args.threshold_frac)
    debug: dict[str, Any] = {
        "threshold": float(threshold),
        "threshold_mode": args.threshold_mode,
        "window_method": args.window_method,
    }

    if args.window_method == "peak_fixed":
        left, right = fixed_peak_window(smoothed_diff, args.fixed_window_bins)
    elif args.window_method == "threshold_expand":
        left, right = threshold_expand_window(smoothed_diff, threshold)
    elif args.window_method == "connected_component":
        left, right, component_debug = connected_component_window(
            smoothed_diff,
            smoothed_entropy,
            threshold,
            args.candidate_diff_frac,
        )
        debug.update(component_debug)
    else:
        raise ValueError(f"Unknown window method: {args.window_method}")

    left, right = clamp_window_length(left, right, smoothed_diff, args.min_window_bins, args.max_window_bins)
    debug.update(
        {
            "selected_indices": list(range(left, right + 1)),
            "selected_start_idx": int(left),
            "selected_end_idx": int(right),
            "smoothed_diff": [float(value) for value in smoothed_diff],
            "smoothed_entropy": [float(value) for value in smoothed_entropy],
        }
    )
    return left, right, debug


def temporal_windows_from_grid_ts(grid_ts: list[float], duration: float) -> list[tuple[float, float]]:
    if not grid_ts:
        return [(0.0, max(duration, 0.01))]
    n = len(grid_ts)
    if n == 1:
        return [(0.0, max(duration, 0.01))]

    centers = [max(0.0, min(float(ts), duration)) for ts in grid_ts]
    if any(centers[idx] < centers[idx - 1] for idx in range(1, n)):
        return core.partition_windows(duration, n)

    edges = [0.0]
    for idx in range(1, n):
        edges.append(max(edges[-1], 0.5 * (centers[idx - 1] + centers[idx])))
    edges.append(max(duration, edges[-1] + 1e-6))
    return [(float(edges[idx]), float(min(edges[idx + 1], duration))) for idx in range(n)]


def aggregate_temporal_importance(
    token_values: torch.Tensor,
    video_grid_thw: list[list[int]],
    spatial_merge_size: int,
    aggregation: str,
    topk_frac: float,
) -> tuple[list[float], dict[str, Any]]:
    if len(video_grid_thw) != 1:
        raise ValueError(f"Expected exactly one video grid, got {video_grid_thw}.")
    grid_t, grid_h, grid_w = [int(value) for value in video_grid_thw[0]]
    if grid_t <= 0:
        raise ValueError(f"Invalid video grid_t: {video_grid_thw}.")

    values = token_values.detach().float().cpu().numpy().reshape(-1)
    merged_h = max(1, grid_h // max(spatial_merge_size, 1))
    merged_w = max(1, grid_w // max(spatial_merge_size, 1))
    expected_tokens_per_t = merged_h * merged_w
    expected_total = grid_t * expected_tokens_per_t
    if values.size != expected_total:
        if values.size % grid_t != 0:
            raise ValueError(
                "Cannot map visual tokens to temporal bins: "
                f"token_count={values.size}, video_grid_thw={video_grid_thw}, spatial_merge_size={spatial_merge_size}."
            )
        tokens_per_t = values.size // grid_t
    else:
        tokens_per_t = expected_tokens_per_t

    grouped = values.reshape(grid_t, tokens_per_t)
    temporal: list[float] = []
    for row in grouped:
        if aggregation == "mean":
            temporal.append(float(row.mean()))
        elif aggregation == "max":
            temporal.append(float(row.max()))
        elif aggregation == "topk_mean":
            k = max(1, min(tokens_per_t, int(np.ceil(tokens_per_t * topk_frac))))
            temporal.append(float(np.partition(row, -k)[-k:].mean()))
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

    return temporal, {
        "grid_t": grid_t,
        "grid_h": grid_h,
        "grid_w": grid_w,
        "spatial_merge_size": spatial_merge_size,
        "tokens_per_t": int(tokens_per_t),
        "token_count": int(values.size),
        "aggregation": aggregation,
        "topk_frac": float(topk_frac),
    }


def prepare_full_video_inputs(
    processor: Any,
    sample: dict[str, Any],
    duration: float,
    args: argparse.Namespace,
    work_dir: Path,
) -> tuple[Any, list[float], list[float]]:
    timestamps = core.sample_uniform_times(0.0, duration, args.total_frames)
    core.assert_count("gradient timestamp", len(timestamps), args.total_frames)
    frames = core.extract_frames_ffmpeg(sample["video_path"], timestamps, work_dir / "gradient_frames")
    core.assert_count("gradient frame", len(frames), args.total_frames)
    frames = core.resize_frames(frames, args.max_pixels, args.final_square_size)
    fps = max(args.total_frames / max(duration, 1e-6), 1e-6)
    inputs = core.build_frame_inputs(
        processor,
        core.build_probe_prompt(sample["query"]),
        frames,
        fps=fps,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    if "video_grid_thw" not in inputs:
        raise ValueError("Processor did not return video_grid_thw for the gradient input.")

    temporal_patch_size = core.get_temporal_patch_size(processor)
    grid_t = int(inputs["video_grid_thw"][0][0].item())
    grid_ts = core.build_grid_timestamps(timestamps, grid_t, temporal_patch_size)
    core.assert_count("gradient video_time_grid_ts", len(grid_ts), grid_t)
    inputs["video_time_grid_ts"] = torch.tensor([grid_ts], dtype=torch.float32)
    return inputs, timestamps, grid_ts


def run_gradient_vtg_sample(
    model: Any,
    processor: Any,
    sample: dict[str, Any],
    duration: float,
    token_a_id: int,
    token_b_id: int,
    args: argparse.Namespace,
    work_dir: Path,
) -> dict[str, Any]:
    inputs, frame_timestamps, grid_ts = prepare_full_video_inputs(processor, sample, duration, args, work_dir)
    video_grid_thw = core.get_video_grid_thw(inputs)
    video_llm_tokens = core.get_video_llm_token_count(inputs, model)
    temporal_windows = temporal_windows_from_grid_ts(grid_ts, duration)

    device = model_input_device(model)
    inputs = inputs.to(device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    model._qwen25vl_capture_video_grads = True
    model._qwen25vl_last_video_embeds = None
    start_time = time.perf_counter()
    try:
        with torch.enable_grad():
            outputs = model(**inputs, use_cache=False)
            next_logits = outputs.logits[0, -1, :].float()
            ab_logits = torch.stack([next_logits[token_a_id], next_logits[token_b_id]])
            ab_probs = torch.softmax(ab_logits, dim=0)
            vocab_probs = torch.softmax(next_logits, dim=0)
            entropy = -(vocab_probs * torch.log(vocab_probs.clamp_min(1e-12))).sum()
            logit_diff = ab_logits[0] - ab_logits[1]

            video_embeds = getattr(model, "_qwen25vl_last_video_embeds", None)
            if video_embeds is None:
                raise RuntimeError("Qwen2.5-VL did not expose captured video embeddings for gradient VTG.")

            grad_diff = torch.autograd.grad(logit_diff, video_embeds, retain_graph=True)[0]
            grad_entropy = torch.autograd.grad(entropy, video_embeds, retain_graph=False)[0]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        runtime_sec = time.perf_counter() - start_time

        diff_token_importance = grad_diff.detach().float().norm(p=2, dim=-1)
        entropy_token_importance = grad_entropy.detach().float().norm(p=2, dim=-1)
        spatial_merge_size = int(getattr(model.config.vision_config, "spatial_merge_size", 2))
        diff_temporal, map_debug = aggregate_temporal_importance(
            diff_token_importance,
            video_grid_thw,
            spatial_merge_size,
            args.aggregation,
            args.topk_frac,
        )
        entropy_temporal, entropy_map_debug = aggregate_temporal_importance(
            entropy_token_importance,
            video_grid_thw,
            spatial_merge_size,
            args.aggregation,
            args.topk_frac,
        )
        if len(diff_temporal) != len(temporal_windows):
            raise ValueError(
                f"Temporal gradient/window mismatch: scores={len(diff_temporal)} windows={len(temporal_windows)}."
            )

        left, right, selection_debug = select_temporal_window(diff_temporal, entropy_temporal, args)
        pred_start = float(temporal_windows[left][0])
        pred_end = float(temporal_windows[right][1])
        pred_start, pred_end = core.sanitize_window(pred_start, pred_end, duration)
        peak_gpu_memory_mb = None
        if torch.cuda.is_available():
            peak_gpu_memory_mb = float(torch.cuda.max_memory_allocated() / (1024 * 1024))

        normalized_entropy = float(entropy.detach().item() / np.log(float(next_logits.numel())))
        return {
            "pred_start": pred_start,
            "pred_end": pred_end,
            "frame_timestamps": [float(value) for value in frame_timestamps],
            "grid_timestamps": [float(value) for value in grid_ts],
            "temporal_windows": [[float(start), float(end)] for start, end in temporal_windows],
            "diff_temporal": [float(value) for value in diff_temporal],
            "entropy_temporal": [float(value) for value in entropy_temporal],
            "logit_A": float(ab_logits[0].detach().item()),
            "logit_B": float(ab_logits[1].detach().item()),
            "logit_diff": float(logit_diff.detach().item()),
            "p_yes": float(ab_probs[0].detach().item()),
            "entropy": float(entropy.detach().item()),
            "normalized_entropy": normalized_entropy,
            "video_grid_thw": video_grid_thw,
            "video_llm_tokens": int(video_llm_tokens),
            "map_debug": map_debug,
            "entropy_map_debug": entropy_map_debug,
            "selection_debug": selection_debug,
            "runtime_sec": float(runtime_sec),
            "peak_gpu_memory_mb": peak_gpu_memory_mb,
        }
    finally:
        model._qwen25vl_capture_video_grads = False
        model._qwen25vl_last_video_embeds = None


def summarize_gradient_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if "error" not in row]
    ious = [float(row.get("gradient_vtg_iou") or 0.0) for row in valid]
    pred_fracs = []
    runtimes = []
    memories = []
    for row in valid:
        duration = float(row.get("duration") or 0.0)
        start = row.get("gradient_vtg_start")
        end = row.get("gradient_vtg_end")
        if duration > 0 and start is not None and end is not None:
            pred_fracs.append(max(0.0, float(end) - float(start)) / duration)
        stats = row.get("gradient_vtg_stats") or {}
        if stats.get("runtime_sec") is not None:
            runtimes.append(float(stats["runtime_sec"]))
        if stats.get("peak_gpu_memory_mb") is not None:
            memories.append(float(stats["peak_gpu_memory_mb"]))

    return {
        "samples": len(ious),
        "mIoU": sum(ious) / len(ious) if ious else None,
        **{
            f"Recall@1_IoU={thr}": sum(iou >= thr for iou in ious) / len(ious) if ious else None
            for thr in core.RECALL_THRESHOLDS
        },
        "avg_pred_frac": sum(pred_fracs) / len(pred_fracs) if pred_fracs else None,
        "avg_runtime_sec": sum(runtimes) / len(runtimes) if runtimes else None,
        "max_peak_gpu_memory_mb": max(memories) if memories else None,
    }


def run_gradient_vtg(args: argparse.Namespace) -> dict[str, Any]:
    if args.total_frames <= 0:
        raise ValueError("--total-frames must be positive.")
    if args.min_window_bins <= 0:
        raise ValueError("--min-window-bins must be positive.")
    if args.max_window_bins < 0:
        raise ValueError("--max-window-bins cannot be negative.")
    if not 0.0 < args.topk_frac <= 1.0:
        raise ValueError("--topk-frac must be in (0, 1].")
    if not 0.0 <= args.candidate_diff_frac <= 1.0:
        raise ValueError("--candidate-diff-frac must be in [0, 1].")

    if args.output_jsonl == DEFAULT_OUTPUT_JSONL:
        args.output_jsonl = DEFAULT_OUTPUT_JSONL.parent / f"{args.dataset}_gradient_vtg_details.jsonl"
    if args.summary_json == DEFAULT_SUMMARY_JSON:
        args.summary_json = DEFAULT_SUMMARY_JSON.parent / f"{args.dataset}_gradient_vtg_summary.json"

    core.set_seed(args.seed)
    model, processor, token_a_id, token_b_id = load_model_and_processor(args)

    samples = load_samples(args)
    query_offset = max(0, int(getattr(args, "query_offset", 0) or 0))
    if query_offset:
        samples = samples[query_offset:]
    if args.num_queries is not None:
        samples = samples[: args.num_queries]

    rows: list[dict[str, Any]] = []
    failures = 0
    missing_videos = 0
    used = 0
    temp_root = Path(tempfile.mkdtemp(prefix="qwen25vl_gradient_vtg_"))
    pbar = tqdm(samples, desc=f"gradient_vtg:{args.dataset}", ncols=120)
    try:
        for idx, sample in enumerate(pbar):
            video_path = str(sample.get("video_path") or "")
            if not video_path or not os.path.exists(video_path):
                missing_videos += 1
                continue

            result = {
                "sample_id": sample.get("sample_id"),
                "video_id": sample.get("video_id"),
                "video_path": video_path,
                "query": sample.get("query"),
                "gt_start": float(sample["gt_start"]),
                "gt_end": float(sample["gt_end"]),
            }
            try:
                duration = core.ffprobe_duration(video_path)
                result["duration"] = duration
                pred = run_gradient_vtg_sample(
                    model,
                    processor,
                    sample,
                    duration,
                    token_a_id,
                    token_b_id,
                    args,
                    temp_root / f"sample_{idx:04d}",
                )
                result["gradient_vtg_start"] = pred["pred_start"]
                result["gradient_vtg_end"] = pred["pred_end"]
                result["gradient_vtg_iou"] = core.iou_1d(
                    pred["pred_start"],
                    pred["pred_end"],
                    result["gt_start"],
                    result["gt_end"],
                )
                result["gradient_vtg_stats"] = pred
            except Exception as exc:
                failures += 1
                result["error"] = repr(exc)
                print(f"[FAIL] sample={sample.get('sample_id')} err={exc!r}")
                rows.append(result)
                continue

            used += 1
            rows.append(result)
            summary = summarize_gradient_rows(rows)
            pbar.set_postfix(
                {
                    "used": used,
                    "miss": missing_videos,
                    "fail": failures,
                    "mIoU": f"{summary['mIoU'] or 0.0:.4f}",
                }
            )
    finally:
        if args.keep_temp:
            print(f"[INFO] Kept temp dir: {temp_root}")
        else:
            shutil.rmtree(temp_root, ignore_errors=True)

    summary = {
        "sample_source": args.dataset,
        "query_offset": query_offset,
        "used_samples": used,
        "missing_videos": missing_videos,
        "failures": failures,
        "config": {
            "total_frames": args.total_frames,
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
            "final_square_size": args.final_square_size,
            "aggregation": args.aggregation,
            "topk_frac": args.topk_frac,
            "smooth_radius": args.smooth_radius,
            "window_method": args.window_method,
            "threshold_mode": args.threshold_mode,
            "threshold_frac": args.threshold_frac,
            "candidate_diff_frac": args.candidate_diff_frac,
            "fixed_window_bins": args.fixed_window_bins,
            "min_window_bins": args.min_window_bins,
            "max_window_bins": args.max_window_bins,
        },
        "gradient_vtg": summarize_gradient_rows(rows),
    }

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.save_details:
        core.write_jsonl(args.output_jsonl, rows)
        summary["details_jsonl"] = str(args.output_jsonl)
        args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_gradient_vtg(args)


if __name__ == "__main__":
    main(sys.argv[1:])
