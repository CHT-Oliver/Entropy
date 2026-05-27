"""PoC v3 — visualize uncertainty gradient on Qwen2.5-VL for video.

Per segment:
  1) extract `frames_per_segment` uniformly inside [s, e]
  2) build inputs with the **Tempo official inference prompt 1**
     (system+user, copied verbatim from reference/Tempo/tempo/mm_datautils.py:266-271)
  3) ONE forward → TWO backwards via torch.autograd.grad:
       grad_diff    = ∂(logit_Yes − logit_No) / ∂V    (Tempo router signal)
       grad_entropy = ∂H / ∂V                          (EGG full-vocab entropy signal)
  4) saliency_diff_3d, saliency_entropy_3d : (T_seg, H_llm, W_llm)

Visualization: per-segment T_seg time slots fully expanded (no averaging),
two gradient rows (logit_diff vs entropy) stacked.

Reference cross-checks:
  - EGG entropy:  utils.py::calc_grad — `(probs * log probs).sum() * -1`  ✓ same
  - Tempo router: qwen3vl_encoder.py:85-95 — sigmoid(logit_Yes − logit_No)  ✓ same
  - Tempo prompt: mm_datautils.py:266-271 — system+user, "answer exactly Yes/No"  ✓ verbatim
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- Project paths ----
PROJECT_ROOT = Path(__file__).resolve().parents[2]
QWEN_DIR = PROJECT_ROOT / "qwen2_5_vl"
sys.path.insert(0, str(QWEN_DIR))

import validate as core  # noqa: E402

# ---- Defaults ----
CKPT = "/data/home/plumliu/myspace_cq/weights/Qwen2.5-VL-7B-Instruct"
CHARADES_VIDEO_ROOT = Path(
    "/apdcephfs_cq8/share_1367250/alfiechen/video_data/video_data_collection/Charades/Charades_v1_480"
)
CHARADES_STA_TEST = Path(
    "/apdcephfs_sh3/share_302139670/alfiechen/data/video_data/Charades_STA/annotations/charades_sta_test.txt"
)
ANNOTATION = CHARADES_STA_TEST
VIDEO_ROOT = CHARADES_VIDEO_ROOT
OUTPUT_DIR = Path(__file__).parent / "outputs"

DEFAULT_NUM_SAMPLES = 1
DEFAULT_SEGMENT_LEN = 5.0
DEFAULT_FRAMES_PER_SEGMENT = 8
FRAME_SIZE = 112


# =============================================================================
#  Tempo official prompt (verbatim from reference/Tempo/tempo/mm_datautils.py:266-275)
# =============================================================================
TEMPO_SYSTEM_INFERENCE = (
    "You are a query-conditioned visual compressor. "
    "Store in the provided memory tokens the minimal visual information needed to answer the Query. "
    "Ignore irrelevant details. "
    "Now, before compressing, answer exactly 'Yes' or 'No': is this segment relevant to the Query?"
)


def tempo_messages(query: str) -> list[dict[str, Any]]:
    """Tempo official inference prompt 1: system + user with video."""
    return [
        {"role": "system", "content": [{"type": "text", "text": TEMPO_SYSTEM_INFERENCE}]},
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": f"\nQuery:\n{query}"},
            ],
        },
    ]


# =============================================================================
#  Sample picking
# =============================================================================

def pick_samples(
    annotation: Path,
    video_root: Path,
    n: int,
    min_gt_duration: float = 2.0,
    dedupe_video: bool = True,
) -> list[dict[str, Any]]:
    picked: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in annotation.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or "##" not in line:
            continue
        head, query = line.split("##", 1)
        parts = head.split()
        if len(parts) != 3:
            continue
        video_id, s, e = parts
        try:
            start, end = float(s), float(e)
        except ValueError:
            continue
        if end - start < min_gt_duration:
            continue
        if dedupe_video and video_id in seen:
            continue
        video_path = video_root / f"{video_id}.mp4"
        if not video_path.exists():
            continue
        picked.append(
            {
                "video_id": video_id,
                "video_path": str(video_path),
                "query": query.strip().rstrip(".") + ".",
                "gt_start": float(start),
                "gt_end": float(end),
            }
        )
        seen.add(video_id)
        if len(picked) >= n:
            break
    return picked


# =============================================================================
#  Frame extraction
# =============================================================================

def get_video_meta(video_path: str) -> tuple[float, float, int]:
    import decord

    decord.bridge.set_bridge("native")
    vr = decord.VideoReader(video_path, num_threads=1)
    fps = float(vr.get_avg_fps())
    total = len(vr)
    duration = total / max(fps, 1e-6)
    return fps, duration, total


def extract_frames_in_window(
    video_path: str, start: float, end: float, n: int
) -> tuple[list[Image.Image], list[float]]:
    import decord

    decord.bridge.set_bridge("native")
    vr = decord.VideoReader(video_path, num_threads=1)
    fps = float(vr.get_avg_fps())
    total = len(vr)
    f_start = max(0, int(start * fps))
    f_end = min(total - 1, int(end * fps))
    if f_end <= f_start:
        f_end = min(total - 1, f_start + 1)
    indices = np.linspace(f_start, f_end, n, dtype=int)
    indices = np.clip(indices, 0, total - 1)
    frames_np = vr.get_batch(list(indices)).asnumpy()
    timestamps = [float(idx) / max(fps, 1e-6) for idx in indices]
    frames = [Image.fromarray(arr) for arr in frames_np]
    return frames, timestamps


def build_segment_windows(duration: float, segment_len: float) -> list[tuple[float, float]]:
    if duration <= 0:
        return [(0.0, 0.01)]
    n = max(1, int(round(duration / segment_len)))
    return [
        (round(i * duration / n, 3), round((i + 1) * duration / n, 3))
        for i in range(n)
    ]


def letterbox_square(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    scale = min(size / max(w, 1), size / max(h, 1))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = img.resize((new_w, new_h), Image.BICUBIC)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(resized, ((size - new_w) // 2, (size - new_h) // 2))
    return canvas


# =============================================================================
#  Model loading
# =============================================================================

def load_model_and_processor():
    qwen_module = core.load_project_qwen25vl_modeling(core.default_qwen25vl_modeling_src())

    from transformers import AutoProcessor

    if qwen_module is not None:
        Qwen2_5_VLForConditionalGeneration = qwen_module.Qwen2_5_VLForConditionalGeneration
    else:
        from transformers import Qwen2_5_VLForConditionalGeneration

    print(f"[INFO] Loading model from {CKPT} ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        CKPT,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    processor = AutoProcessor.from_pretrained(CKPT)
    return model, processor


def get_token_id(tokenizer: Any, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"Expected single token for {text!r}, got {ids}")
    return int(ids[0])


# =============================================================================
#  Core: hooked forward + dual grad backward
# =============================================================================

def build_inputs(
    processor: Any, query: str, frames: list[Image.Image], segment_duration: float
):
    boxed = [letterbox_square(f, FRAME_SIZE) for f in frames]
    messages = tempo_messages(query)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    fps = max(len(boxed) / max(segment_duration, 1e-6), 1e-6)
    return processor(
        text=[text],
        videos=[boxed],
        fps=fps,
        min_pixels=3136,
        max_pixels=12544,
        padding=True,
        return_tensors="pt",
    )


def run_dual_grad_probe(
    model: Any,
    processor: Any,
    query: str,
    frames: list[Image.Image],
    segment_duration: float,
    token_yes: int,
    token_no: int,
) -> dict[str, Any]:
    """One forward → two backwards via torch.autograd.grad. Mirrors validate_gradient_vtg.py."""
    inputs = build_inputs(processor, query, frames, segment_duration).to(model.device)

    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _args, output):
        new_out = output.detach().requires_grad_(True)
        captured["video_embeds"] = new_out
        return new_out

    handle = model.visual.register_forward_hook(hook)
    try:
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values_videos=inputs["pixel_values_videos"],
            video_grid_thw=inputs["video_grid_thw"],
            return_dict=True,
            use_cache=False,
        )
    finally:
        handle.remove()
    if "video_embeds" not in captured:
        raise RuntimeError("Visual hook did not fire.")

    next_logits = outputs.logits[0, -1, :].float()
    log_probs = F.log_softmax(next_logits, dim=-1)
    probs = log_probs.exp()

    logit_yes = next_logits[token_yes]
    logit_no = next_logits[token_no]
    logit_diff = logit_yes - logit_no
    entropy = -(probs * log_probs).sum()

    video_embeds = captured["video_embeds"]
    grad_diff = torch.autograd.grad(logit_diff, video_embeds, retain_graph=True)[0]
    grad_entropy = torch.autograd.grad(entropy, video_embeds, retain_graph=False)[0]

    sal_diff = grad_diff.float().norm(p=2, dim=-1).detach().cpu().numpy()
    sal_entropy = grad_entropy.float().norm(p=2, dim=-1).detach().cpu().numpy()

    grid_thw = inputs["video_grid_thw"][0].tolist()
    spatial_merge = int(model.config.vision_config.spatial_merge_size)
    T = int(grid_thw[0])
    H_llm = int(grid_thw[1]) // spatial_merge
    W_llm = int(grid_thw[2]) // spatial_merge
    expected = T * H_llm * W_llm
    if sal_diff.size != expected or sal_entropy.size != expected:
        raise RuntimeError(
            f"Saliency size mismatch: got diff={sal_diff.size} ent={sal_entropy.size}, "
            f"expected {expected}."
        )
    sal_diff_3d = sal_diff.reshape(T, H_llm, W_llm)
    sal_entropy_3d = sal_entropy.reshape(T, H_llm, W_llm)

    p_yes = float(torch.sigmoid(logit_diff).detach().item())

    return {
        "sal_diff_3d": sal_diff_3d,
        "sal_entropy_3d": sal_entropy_3d,
        "logit_yes": float(logit_yes.detach().item()),
        "logit_no": float(logit_no.detach().item()),
        "logit_diff": float(logit_diff.detach().item()),
        "p_yes": p_yes,
        "entropy": float(entropy.detach().item()),
        "grid_thw": grid_thw,
        "T_seg": T,
    }


# =============================================================================
#  Visualization
# =============================================================================

def overlay_heatmap(
    frame: Image.Image, sal: np.ndarray, alpha: float = 0.45
) -> Image.Image:
    sal_n = sal - sal.min()
    if sal_n.max() > 1e-12:
        sal_n = sal_n / sal_n.max()
    sal_pil = Image.fromarray((sal_n * 255).astype(np.uint8)).resize(
        frame.size, Image.BILINEAR
    )
    sal_arr = np.array(sal_pil) / 255.0
    cmap = matplotlib.colormaps.get_cmap("jet")
    heat_rgb = (cmap(sal_arr)[..., :3] * 255).astype(np.uint8)
    heat = Image.fromarray(heat_rgb)
    return Image.blend(frame.convert("RGB"), heat, alpha)


def visualize_segments_expanded(
    sample: dict[str, Any],
    duration: float,
    segments: list[dict[str, Any]],
    out_path: Path,
) -> None:
    """Layout: 2 grad rows × (n_seg × T_seg) cells, plus 2 plot rows."""
    n_seg = len(segments)
    if n_seg == 0:
        return
    T_seg = segments[0]["probe"]["T_seg"]
    total_cols = n_seg * T_seg

    fig = plt.figure(figsize=(max(0.9 * total_cols, 9.0), 9.5))
    gs = fig.add_gridspec(
        4, total_cols, height_ratios=[1.0, 1.0, 1.0, 1.0], hspace=0.6, wspace=0.05
    )

    # ---- Heatmap rows ----
    for s_idx, seg in enumerate(segments):
        probe = seg["probe"]
        s_start, s_end = seg["window"]
        n_frames = len(seg["frames"])
        for t in range(T_seg):
            f_lo = int(t * n_frames / T_seg)
            f_hi = int((t + 1) * n_frames / T_seg)
            f_pick = (f_lo + f_hi) // 2
            f_pick = min(max(f_pick, 0), n_frames - 1)
            rep_frame = seg["frames"][f_pick]

            col = s_idx * T_seg + t

            ax_diff = fig.add_subplot(gs[0, col])
            ax_diff.imshow(overlay_heatmap(rep_frame, probe["sal_diff_3d"][t]))
            ax_diff.set_xticks([])
            ax_diff.set_yticks([])
            if t == 0:
                ax_diff.set_title(f"seg{s_idx} [{s_start:.1f}-{s_end:.1f}]", fontsize=8)

            ax_ent = fig.add_subplot(gs[1, col])
            ax_ent.imshow(overlay_heatmap(rep_frame, probe["sal_entropy_3d"][t]))
            ax_ent.set_xticks([])
            ax_ent.set_yticks([])
            ax_ent.set_xlabel(f"t{t}", fontsize=7)

            if col == 0:
                ax_diff.set_ylabel("∂(Δlogit)/∂V\n(Tempo)", fontsize=8)
                ax_ent.set_ylabel("∂H/∂V\n(EGG)", fontsize=8)

    # ---- Bar chart of per-segment p(yes) ----
    seg_centers = [(seg["window"][0] + seg["window"][1]) / 2 for seg in segments]
    seg_widths = [seg["window"][1] - seg["window"][0] for seg in segments]
    p_yes_arr = np.array([seg["probe"]["p_yes"] for seg in segments])
    diff_arr = np.array([seg["probe"]["logit_diff"] for seg in segments])

    ax2 = fig.add_subplot(gs[2, :])
    bar_colors = [
        "tab:green" if (sample["gt_start"] <= c <= sample["gt_end"]) else "tab:gray"
        for c in seg_centers
    ]
    ax2.bar(seg_centers, p_yes_arr, width=[w * 0.9 for w in seg_widths], color=bar_colors, alpha=0.85)
    ax2.axhline(0.5, color="red", linestyle="--", linewidth=0.8, label="p=0.5")
    ax2.axvspan(
        sample["gt_start"], sample["gt_end"], color="green", alpha=0.15,
        label=f"GT [{sample['gt_start']:.1f}, {sample['gt_end']:.1f}]"
    )
    ax2.set_xlim(0, duration)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("p(yes) = σ(Δlogit)")
    ax2.set_title("Tempo router signal per segment", fontsize=9)
    ax2.legend(loc="upper right", fontsize=7)
    ax2.grid(True, alpha=0.3)
    for c, p, d in zip(seg_centers, p_yes_arr, diff_arr):
        ax2.text(c, p + 0.02, f"{p:.2f}\n(Δ={d:+.2f})", ha="center", fontsize=6)

    # ---- Curve of total saliency magnitudes ----
    sum_diff = np.array([seg["probe"]["sal_diff_3d"].sum() for seg in segments])
    sum_ent = np.array([seg["probe"]["sal_entropy_3d"].sum() for seg in segments])
    sum_diff_n = sum_diff / max(sum_diff.max(), 1e-12)
    sum_ent_n = sum_ent / max(sum_ent.max(), 1e-12)

    ax3 = fig.add_subplot(gs[3, :])
    ax3.plot(seg_centers, sum_diff_n, marker="o", color="C0",
             label="‖∂(Δlogit)/∂V‖ Σ (norm.)")
    ax3.plot(seg_centers, sum_ent_n, marker="s", color="C1",
             label="‖∂H/∂V‖ Σ (norm.)")
    ax3.axvspan(sample["gt_start"], sample["gt_end"], color="green", alpha=0.15)
    ax3.set_xlim(0, duration)
    ax3.set_ylim(0, 1.05)
    ax3.set_xlabel("time (s)")
    ax3.set_ylabel("normalized\ntotal saliency")
    ax3.set_title("Per-segment gradient magnitude", fontsize=9)
    ax3.legend(loc="upper right", fontsize=7)
    ax3.grid(True, alpha=0.3)

    fig.suptitle(
        f"{sample['video_id']}  ·  '{sample['query']}'  ·  duration={duration:.1f}s  ·  "
        f"GT=[{sample['gt_start']:.1f}, {sample['gt_end']:.1f}]  ·  "
        f"{n_seg} segments × T_seg={T_seg}  ·  Tempo official prompt 1",
        fontsize=11,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
#  Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--segment-len", type=float, default=DEFAULT_SEGMENT_LEN)
    parser.add_argument("--frames-per-segment", type=int, default=DEFAULT_FRAMES_PER_SEGMENT)
    parser.add_argument("--annotation", type=Path, default=ANNOTATION)
    parser.add_argument("--video-root", type=Path, default=VIDEO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    samples = pick_samples(args.annotation, args.video_root, args.num_samples)
    print(f"[INFO] Picked {len(samples)} sample(s).")
    if not samples:
        print("[ERROR] No samples found.")
        return

    model, processor = load_model_and_processor()
    token_yes = get_token_id(processor.tokenizer, "Yes")
    token_no = get_token_id(processor.tokenizer, "No")
    print(f"[INFO] token_id(Yes)={token_yes}, token_id(No)={token_no}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "log.jsonl"

    with log_path.open("w", encoding="utf-8") as logf:
        for s_idx, sample in enumerate(samples):
            print(f"\n[SAMPLE {s_idx}] {sample['video_id']}  ·  {sample['query']!r}")
            try:
                _, duration, _ = get_video_meta(sample["video_path"])
            except Exception as exc:
                print(f"  [ERROR] meta extraction failed: {exc}")
                continue

            windows = build_segment_windows(duration, args.segment_len)
            print(f"  duration={duration:.1f}s  →  {len(windows)} segments  "
                  f"(≈{args.segment_len}s each, {args.frames_per_segment} frames/seg)")

            seg_results: list[dict[str, Any]] = []
            for seg_idx, (s, e) in enumerate(windows):
                seg_dur = e - s
                try:
                    frames, ts = extract_frames_in_window(
                        sample["video_path"], s, e, args.frames_per_segment
                    )
                except Exception as exc:
                    print(f"  [seg {seg_idx}] frame extract failed: {exc}")
                    continue

                probe = run_dual_grad_probe(
                    model, processor, sample["query"], frames, seg_dur, token_yes, token_no
                )
                torch.cuda.empty_cache()

                seg_results.append(
                    {
                        "window": (s, e),
                        "frames": frames,
                        "frame_timestamps": ts,
                        "probe": probe,
                    }
                )
                print(
                    f"  [seg {seg_idx}] [{s:5.1f}-{e:5.1f}]  "
                    f"p(yes)={probe['p_yes']:.3f}  "
                    f"logit_diff={probe['logit_diff']:+.3f}  "
                    f"H={probe['entropy']:.3f}  "
                    f"T_seg={probe['T_seg']}"
                )

            log_entry = {
                "video_id": sample["video_id"],
                "query": sample["query"],
                "gt_start": sample["gt_start"],
                "gt_end": sample["gt_end"],
                "duration": duration,
                "prompt": "tempo_official_inference_prompt_1",
                "segments": [
                    {
                        "window": list(seg["window"]),
                        "p_yes": seg["probe"]["p_yes"],
                        "logit_diff": seg["probe"]["logit_diff"],
                        "logit_yes": seg["probe"]["logit_yes"],
                        "logit_no": seg["probe"]["logit_no"],
                        "entropy": seg["probe"]["entropy"],
                        "sum_diff_saliency": float(seg["probe"]["sal_diff_3d"].sum()),
                        "sum_entropy_saliency": float(seg["probe"]["sal_entropy_3d"].sum()),
                        "T_seg": seg["probe"]["T_seg"],
                        "grid_thw": seg["probe"]["grid_thw"],
                    }
                    for seg in seg_results
                ],
            }
            logf.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            logf.flush()

            out_png = args.output_dir / f"sample_{s_idx:02d}_{sample['video_id']}.png"
            visualize_segments_expanded(sample, duration, seg_results, out_png)
            print(f"  saved: {out_png}")
            torch.cuda.empty_cache()

    print(f"\n[DONE] Outputs at {args.output_dir}")


if __name__ == "__main__":
    main()
