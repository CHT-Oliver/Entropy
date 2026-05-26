from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import validate as core

DEFAULT_ANNOTATIONS = [
    Path("/ssd/cht/datasets/ActivityNet_Captions/activitynet_captions_val1.json"),
    Path("/ssd/cht/datasets/ActivityNet_Captions/activitynet_captions_val2.json"),
]
DEFAULT_VIDEO_ROOT = Path("/ssd/cht/datasets/ActivityNet_Captions/Activity_Videos")
DEFAULT_OUTPUT_JSONL = Path("outputs/qwen25vl_adaptive_temporal/activitynet_captions_results.jsonl")  # Written only when --save-details is used.
DEFAULT_SUMMARY_JSON = Path("outputs/qwen25vl_adaptive_temporal/activitynet_captions_summary.json")  # Default summary with final mIoU and Recall@1 only.
DEFAULT_NUM_QUERIES: int | None = 1000  # None means validate all available queries.
DEFAULT_CONFIG = {
    "mode": "both",  # baseline only, adaptive only, or both for a fair paired comparison.
    "total_frames": 64,  # Final QA frame budget per video; baseline and adaptive use the same value.
    "target_segments": 8,  # Use coarser segments for ActivityNet to avoid over-fragmenting long events.
    "min_frames_per_segment": 4,  # Keep enough background coverage for long temporal intervals.
    "max_frames_per_segment": 32,  # Cap frames per segment; high-score segments can still receive extra budget.
    "segment_mode": "partition",  # partition = non-overlap uniform segments; sliding = sliding windows.
    "probe_nframes": 4,  # Frames used to probe each segment for entropy/logit_diff.
    "min_pixels": 3136,  # Processor parameter; effective size is controlled by square_size below.
    "max_pixels": 12544,  # Processor max pixels for the final QA pass.
    "final_square_size": 112,  # Letterbox final QA frames to 112x112 for equal token counts.
    "probe_max_pixels": 3136,  # Processor max pixels for segment probing.
    "probe_square_size": 56,  # Letterbox probe frames to 56x56 to keep probing cheap.
    "allocation_mode": "continuous",  # Smoothly allocate frames by logit_diff plus low-entropy confidence.
    "entropy_direction": "low",  # Lower entropy means the probe is more confident.
    "score_alpha": 0.7,  # Weight for normalized logit_diff in continuous allocation.
    "score_beta": 0.3,  # Weight for normalized low-entropy confidence in continuous allocation.
    "temperature": 0.7,  # Allocation softness; lower is sharper, higher is closer to uniform.
}

def config_to_args(config: dict[str, object]) -> list[str]:
    args: list[str] = []
    for key, value in config.items():
        args.extend([f"--{key.replace('_', '-')}", str(value)])
    return args


def parse_dataset_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--annotation", action="append", type=Path, default=None)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--query-offset", type=int, default=0)
    parser.add_argument("--num-queries", type=int, default=None)
    return parser.parse_known_args(argv)


def build_samples(annotations: list[Path], video_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for annotation in annotations:
        split_name = annotation.stem.replace("activitynet_captions_", "")
        ann = json.loads(annotation.read_text(encoding="utf-8"))
        for item in ann:
            video_id = str(item["video_id"])
            video_name = item.get("video") or f"{video_id}.mp4"
            video_path = video_root / video_name
            if not video_path.exists():
                continue
            timestamps = item.get("timestamps", [])
            sentences = item.get("sentences", [])
            for idx, (ts, sent) in enumerate(zip(timestamps, sentences)):
                rows.append(
                    {
                        "sample_id": f"activitynet_{split_name}_{len(rows):04d}_{video_id}_{idx}",
                        "split": split_name,
                        "video_id": video_id,
                        "video_path": str(video_path),
                        "query": sent,
                        "gt_start": float(ts[0]),
                        "gt_end": float(ts[1]),
                    }
                )
    return rows


def main(argv: list[str] | None = None) -> None:
    dataset_args, remaining = parse_dataset_args(argv)
    args = core.parse_args(config_to_args(DEFAULT_CONFIG) + remaining)
    args.samples_jsonl = None
    args.sample_offset = max(0, int(dataset_args.query_offset or 0))
    if dataset_args.num_queries is not None:
        args.max_samples = dataset_args.num_queries
    elif args.max_samples is None:
        args.max_samples = DEFAULT_NUM_QUERIES
    if args.output_jsonl == Path("outputs/qwen25vl_adaptive_temporal/results.jsonl"):
        args.output_jsonl = DEFAULT_OUTPUT_JSONL
    if args.summary_json == Path("outputs/qwen25vl_adaptive_temporal/summary.json"):
        args.summary_json = DEFAULT_SUMMARY_JSON

    annotations = dataset_args.annotation or DEFAULT_ANNOTATIONS
    samples = build_samples(annotations, dataset_args.video_root)
    core.run_validation(args, samples=samples, sample_source="activitynet_captions")


if __name__ == "__main__":
    main(sys.argv[1:])
