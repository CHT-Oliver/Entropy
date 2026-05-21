from __future__ import annotations

import argparse 
import json
import sys
from pathlib import Path
from typing import Any

import validate as core

DEFAULT_ANNOTATION = Path("/ssd/cht/datasets/charades-sta/charades_sta_test.json")
DEFAULT_VIDEO_ROOT = Path("/ssd/cht/datasets/charades-sta/Charades_v1_480")
DEFAULT_OUTPUT_JSONL = Path("outputs/qwen25vl_adaptive_temporal/charades_sta_results.jsonl")  # Written only when --save-details is used.
DEFAULT_SUMMARY_JSON = Path("outputs/qwen25vl_adaptive_temporal/charades_sta_summary.json")  # Default summary with final mIoU and Recall@1 only.
DEFAULT_NUM_QUERIES: int | None = 1000  # None means validate all available queries.
DEFAULT_CONFIG = {
    "mode": "both",  # baseline only, adaptive only, or both for a fair paired comparison.
    "total_frames": 64,  # Final QA frame budget per video; baseline and adaptive use the same value.
    "target_segments": 8,  # Split the video into this many temporal segments for probing and allocation.
    "min_frames_per_segment": 6,  # Floor frames per segment; prevents adaptive from starving non-selected segments.
    "max_frames_per_segment": 32,  # Cap frames per segment; selected key segments can still receive extra budget.
    "segment_mode": "partition",  # partition = non-overlap uniform segments; sliding = sliding windows.
    "probe_nframes": 2,  # Frames used to probe each segment for entropy/logit_diff.
    "min_pixels": 3136,  # Processor parameter; effective size is controlled by square_size below.
    "max_pixels": 12544,  # Processor max pixels for the final QA pass.
    "final_square_size": 112,  # Letterbox final QA frames to 112x112 for equal token counts.
    "probe_max_pixels": 3136,  # Processor max pixels for segment probing.
    "probe_square_size": 56,  # Letterbox probe frames to 56x56 to keep probing cheap.
    "allocation_mode": "logit_low_entropy",  # Boost segments with high logit_diff and low entropy.
    "top_logit_frac": 0.5,  # Keep the top 50% segments by logit_diff as candidates.
    "low_entropy_frac": 0.5,  # From those candidates, keep the lowest-entropy 50% for extra budget.
    "min_selected_segments": 1,  # Select at least one segment for every video.
}

def config_to_args(config: dict[str, object]) -> list[str]:
    args: list[str] = []
    for key, value in config.items():
        args.extend([f"--{key.replace('_', '-')}", str(value)])
    return args


def parse_dataset_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--annotation", type=Path, default=DEFAULT_ANNOTATION)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--num-queries", type=int, default=None)
    return parser.parse_known_args(argv)


def build_samples(annotation: Path, video_root: Path) -> list[dict[str, Any]]:
    ann = json.loads(annotation.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for item in ann:
        video_id = str(item["video_id"])
        video_path = video_root / f"{video_id}.mp4"
        if not video_path.exists():
            continue
        rows.append(
            {
                "sample_id": f"charades_{len(rows):04d}_{video_id}",
                "video_id": video_id,
                "video_path": str(video_path),
                "query": item["query"],
                "gt_start": float(item["start"]),
                "gt_end": float(item["end"]),
            }
        )
    return rows


def main(argv: list[str] | None = None) -> None:
    dataset_args, remaining = parse_dataset_args(argv)
    args = core.parse_args(config_to_args(DEFAULT_CONFIG) + remaining)
    args.samples_jsonl = None
    if dataset_args.num_queries is not None:
        args.max_samples = dataset_args.num_queries
    elif args.max_samples is None:
        args.max_samples = DEFAULT_NUM_QUERIES
    if args.output_jsonl == Path("outputs/qwen25vl_adaptive_temporal/results.jsonl"):
        args.output_jsonl = DEFAULT_OUTPUT_JSONL
    if args.summary_json == Path("outputs/qwen25vl_adaptive_temporal/summary.json"):
        args.summary_json = DEFAULT_SUMMARY_JSON

    samples = build_samples(dataset_args.annotation, dataset_args.video_root)
    core.run_validation(args, samples=samples, sample_source="charades_sta")


if __name__ == "__main__":
    main(sys.argv[1:])
