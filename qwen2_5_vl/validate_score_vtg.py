from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

import validate as core
import validate_activitynet_captions as activitynet
import validate_charades_sta as charades


DEFAULT_OUTPUT_JSONL = Path("outputs/qwen25vl_score_vtg/score_vtg_details.jsonl")
DEFAULT_SUMMARY_JSON = Path("outputs/qwen25vl_score_vtg/score_vtg_summary.json")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Direct VTG from segment-level Qwen2.5-VL logit_diff/entropy scores."
    )
    parser.add_argument("--dataset", choices=["charades_sta", "activitynet_captions"], default="charades_sta")
    parser.add_argument("--annotation", action="append", type=Path, default=None)
    parser.add_argument("--video-root", type=Path, default=None)
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--ckpt-dir", default="/ssd/cht/checkpoints/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--qwen25vl-modeling-src", type=Path, default=core.default_qwen25vl_modeling_src())
    parser.add_argument("--transformers-src", default="")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--target-segments", type=int, default=32)
    parser.add_argument("--segment-mode", choices=["partition", "sliding"], default="partition")
    parser.add_argument("--segment-len", type=float, default=8.0)
    parser.add_argument("--probe-nframes", type=int, default=2)
    parser.add_argument("--min-pixels", type=int, default=3136)
    parser.add_argument("--probe-max-pixels", type=int, default=3136)
    parser.add_argument("--probe-square-size", type=int, default=56)

    parser.add_argument(
        "--score-mode",
        choices=["logit_diff", "sigmoid_logit", "logit_confidence", "norm_logit_low_entropy"],
        default="logit_confidence",
    )
    parser.add_argument("--window-method", choices=["top1", "threshold_expand", "max_subarray"], default="max_subarray")
    parser.add_argument("--threshold-mode", choices=["mean", "median", "max_frac"], default="mean")
    parser.add_argument("--threshold-frac", type=float, default=0.5)
    parser.add_argument("--min-window-segments", type=int, default=1)
    parser.add_argument("--max-window-segments", type=int, default=0)
    parser.add_argument("--smooth-radius", type=int, default=1)

    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--save-details", action="store_true")
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
    processor = AutoProcessor.from_pretrained(args.ckpt_dir)
    token_a_id = core.get_single_token_id(processor.tokenizer, "A")
    token_b_id = core.get_single_token_id(processor.tokenizer, "B")
    return model, processor, token_a_id, token_b_id


def sigmoid(values: list[float]) -> list[float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = np.clip(arr, -60.0, 60.0)
    return (1.0 / (1.0 + np.exp(-arr))).tolist()


def smooth_scores(scores: list[float], radius: int) -> list[float]:
    if radius <= 0 or len(scores) <= 1:
        return [float(value) for value in scores]
    smoothed = []
    for idx in range(len(scores)):
        left = max(0, idx - radius)
        right = min(len(scores), idx + radius + 1)
        smoothed.append(float(np.mean(scores[left:right])))
    return smoothed


def build_segment_scores(metrics: list[dict[str, float]], mode: str, smooth_radius: int) -> list[float]:
    logit_values = [float(metric["logit_diff"]) for metric in metrics]
    entropy_values = [float(metric["normalized_entropy"]) for metric in metrics]

    if mode == "logit_diff":
        scores = logit_values
    elif mode == "sigmoid_logit":
        scores = sigmoid(logit_values)
    elif mode == "logit_confidence":
        probs = sigmoid(logit_values)
        scores = [prob * max(0.0, 1.0 - entropy) for prob, entropy in zip(probs, entropy_values)]
    elif mode == "norm_logit_low_entropy":
        logit_norm = core.minmax(logit_values)
        entropy_norm = core.minmax(entropy_values)
        scores = [logit + 0.3 * (1.0 - entropy) for logit, entropy in zip(logit_norm, entropy_norm)]
    else:
        raise ValueError(f"Unknown score mode: {mode}")

    return smooth_scores(scores, smooth_radius)


def score_threshold(scores: list[float], mode: str, frac: float) -> float:
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    if mode == "mean":
        return float(arr.mean())
    if mode == "median":
        return float(np.median(arr))
    if mode == "max_frac":
        return float(arr.max() * frac)
    raise ValueError(f"Unknown threshold mode: {mode}")


def clamp_window_length(left: int, right: int, scores: list[float], min_len: int, max_len: int) -> tuple[int, int]:
    n = len(scores)
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


def top1_window(scores: list[float], min_len: int, max_len: int) -> tuple[int, int]:
    idx = int(np.argmax(np.asarray(scores, dtype=np.float64)))
    return clamp_window_length(idx, idx, scores, min_len, max_len)


def threshold_expand_window(scores: list[float], threshold: float, min_len: int, max_len: int) -> tuple[int, int]:
    peak = int(np.argmax(np.asarray(scores, dtype=np.float64)))
    left = right = peak
    while left > 0 and scores[left - 1] >= threshold:
        left -= 1
    while right < len(scores) - 1 and scores[right + 1] >= threshold:
        right += 1
    return clamp_window_length(left, right, scores, min_len, max_len)


def max_subarray_window(scores: list[float], threshold: float, min_len: int, max_len: int) -> tuple[int, int]:
    adjusted = [float(score - threshold) for score in scores]
    best_sum = float("-inf")
    best_left = best_right = 0
    cur_sum = 0.0
    cur_left = 0
    for idx, value in enumerate(adjusted):
        if cur_sum <= 0:
            cur_sum = value
            cur_left = idx
        else:
            cur_sum += value
        if cur_sum > best_sum:
            best_sum = cur_sum
            best_left = cur_left
            best_right = idx

    if best_sum <= 0:
        return top1_window(scores, min_len, max_len)
    return clamp_window_length(best_left, best_right, scores, min_len, max_len)


def scores_to_window(
    windows: list[tuple[float, float]],
    scores: list[float],
    method: str,
    threshold_mode: str,
    threshold_frac: float,
    min_window_segments: int,
    max_window_segments: int,
) -> tuple[float, float, dict[str, Any]]:
    if not windows:
        return 0.0, 0.0, {"selected_indices": []}

    threshold = score_threshold(scores, threshold_mode, threshold_frac)
    if method == "top1":
        left, right = top1_window(scores, min_window_segments, max_window_segments)
    elif method == "threshold_expand":
        left, right = threshold_expand_window(scores, threshold, min_window_segments, max_window_segments)
    elif method == "max_subarray":
        left, right = max_subarray_window(scores, threshold, min_window_segments, max_window_segments)
    else:
        raise ValueError(f"Unknown window method: {method}")

    return (
        float(windows[left][0]),
        float(windows[right][1]),
        {
            "threshold": float(threshold),
            "selected_indices": list(range(left, right + 1)),
            "selected_start_idx": int(left),
            "selected_end_idx": int(right),
        },
    )


def run_score_vtg_sample(
    model: Any,
    processor: Any,
    sample: dict[str, Any],
    duration: float,
    token_a_id: int,
    token_b_id: int,
    args: argparse.Namespace,
    work_dir: Path,
) -> dict[str, Any]:
    windows = core.build_segment_windows(duration, args)
    metrics = []
    for seg_idx, (start, end) in enumerate(windows):
        metric = core.probe_segment(
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
        metrics.append(metric)

    scores = build_segment_scores(metrics, args.score_mode, args.smooth_radius)
    pred_start, pred_end, debug = scores_to_window(
        windows,
        scores,
        args.window_method,
        args.threshold_mode,
        args.threshold_frac,
        args.min_window_segments,
        args.max_window_segments,
    )
    pred_start, pred_end = core.sanitize_window(pred_start, pred_end, duration)
    return {
        "pred_start": pred_start,
        "pred_end": pred_end,
        "windows": [[float(start), float(end)] for start, end in windows],
        "segment_scores": [float(score) for score in scores],
        "probe_metrics": metrics,
        "selection_debug": debug,
    }


def summarize_score_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if "error" not in row]
    ious = [float(row.get("score_vtg_iou") or 0.0) for row in valid]
    return {
        "samples": len(ious),
        "mIoU": sum(ious) / len(ious) if ious else None,
        **{
            f"Recall@1_IoU={thr}": sum(iou >= thr for iou in ious) / len(ious) if ious else None
            for thr in core.RECALL_THRESHOLDS
        },
    }


def run_score_vtg(args: argparse.Namespace) -> dict[str, Any]:
    if args.target_segments <= 0:
        raise ValueError("--target-segments must be positive.")
    if args.probe_nframes <= 0:
        raise ValueError("--probe-nframes must be positive.")
    if args.min_window_segments <= 0:
        raise ValueError("--min-window-segments must be positive.")
    if args.max_window_segments < 0:
        raise ValueError("--max-window-segments cannot be negative.")

    core.set_seed(args.seed)
    model, processor, token_a_id, token_b_id = load_model_and_processor(args)

    samples = load_samples(args)
    if args.num_queries is not None:
        samples = samples[: args.num_queries]

    rows: list[dict[str, Any]] = []
    failures = 0
    missing_videos = 0
    used = 0
    temp_root = Path(tempfile.mkdtemp(prefix="qwen25vl_score_vtg_"))
    pbar = tqdm(samples, desc=f"score_vtg:{args.dataset}", ncols=120)
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
                pred = run_score_vtg_sample(
                    model,
                    processor,
                    sample,
                    duration,
                    token_a_id,
                    token_b_id,
                    args,
                    temp_root / f"sample_{idx:04d}",
                )
                result["score_vtg_start"] = pred["pred_start"]
                result["score_vtg_end"] = pred["pred_end"]
                result["score_vtg_iou"] = core.iou_1d(
                    pred["pred_start"],
                    pred["pred_end"],
                    result["gt_start"],
                    result["gt_end"],
                )
                result["score_vtg_stats"] = pred
            except Exception as exc:
                failures += 1
                result["error"] = repr(exc)
                print(f"[FAIL] sample={sample.get('sample_id')} err={exc!r}")
                rows.append(result)
                continue

            used += 1
            rows.append(result)
            summary = summarize_score_rows(rows)
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
        "used_samples": used,
        "missing_videos": missing_videos,
        "failures": failures,
        "config": {
            "target_segments": args.target_segments,
            "segment_mode": args.segment_mode,
            "probe_nframes": args.probe_nframes,
            "score_mode": args.score_mode,
            "window_method": args.window_method,
            "threshold_mode": args.threshold_mode,
            "threshold_frac": args.threshold_frac,
            "min_window_segments": args.min_window_segments,
            "max_window_segments": args.max_window_segments,
            "smooth_radius": args.smooth_radius,
        },
        "score_vtg": summarize_score_rows(rows),
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
    run_score_vtg(args)


if __name__ == "__main__":
    main(sys.argv[1:])
