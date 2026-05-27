# PoC · Uncertainty Gradient Visualization on Qwen2.5-VL Video

**目的**：手动验证 ∂H/∂V 在 Qwen2.5-VL 视频侧是否能产生**有语义**的时空 saliency。这是整个研究路线 (Training-free Spatio-Temporal Evidence Localization via Uncertainty Gradients) 的可行性 gating step。

如果 saliency **能在 GT 时间区间附近聚集** → Path 2（提取范式）路径可行，进入正式 method 阶段。
如果 saliency **完全 noise** → 退回 forward-only 段筛 (师弟 path 1)。

---

## 两种 probe 形式

脚本对每个样本独立跑两次 forward+backward：

| Probe | Prompt | 第一 token 信号 | 用途 |
|---|---|---|---|
| **EGG-style** | `build_temporal_prompt` (自由 format `start: …, end: …`) | entropy → ∂H/∂V | 测试无格式约束下的纯 saliency |
| **Tempo-style** | `build_probe_prompt` (A/B 二元) | entropy → ∂H/∂V **+** logit_diff(A,B) | 一个 forward 拿 saliency + relevance 双信号 |

Tempo-style 的设计核心：**一次 forward+backward，两个信号免费拿到**。这是后续主 pipeline 想要的形态。

---

## 运行

```bash
cd /data/home/plumliu/myspace_cq/research/Entropy/experiments/poc_gradient_viz
python poc_visualize.py --num-samples 3 --num-frames 16
```

参数：

```
--num-samples N    抽前 N 个 Charades-STA 样本（默认 3）
--num-frames N     每个视频抽多少帧（默认 16，T = 16 / temporal_patch_size = 8）
--annotation PATH  指向 charades_sta_test.json
--video-root PATH  指向 Charades_v1_480/
--output-dir PATH  输出目录（默认 ./outputs）
```

数据集和 ckpt 的默认路径写在脚本顶部，按需修改。

---

## 输出

```
outputs/
├── log.jsonl                          # 每样本数值指标
├── sample_00_<video_id>.png           # 可视化
├── sample_01_<video_id>.png
└── ...
```

每张 PNG 三行布局：

```
┌─────────────────────────────────────────────────────────────┐
│  [EGG-style heatmaps]    t=1s  t=3s  ... t=N s              │
│  [Tempo-style heatmaps]  t=1s  t=3s  ... t=N s              │
│  [Temporal saliency curves + GT interval shaded green]      │
└─────────────────────────────────────────────────────────────┘
```

`log.jsonl` 每行一个样本：

```json
{
  "video_id": "...",
  "query": "...",
  "gt_start": 5.0, "gt_end": 12.4, "duration": 30.0,
  "egg":   {"entropy": ..., "top_token": "...", "temporal_saliency": [...], "temporal_concentration": ...},
  "tempo": {"entropy": ..., "p_yes": ..., "logit_diff": ...,
            "top_token": "...", "temporal_saliency": [...], "temporal_concentration": ...}
}
```

`temporal_concentration` 是归一化时间熵：**0 = 单峰聚焦，1 = 完全均匀**。我们希望 Tempo-style 的 conc 显著低于 EGG-style，且峰位与 GT 区间对齐。

---

## 验证 checklist

跑完后看几件事：

1. **Saliency 峰值在 GT 区间附近吗？**
   绿色阴影是 GT，曲线峰应该落在阴影内或紧邻阴影。
2. **Tempo-style 比 EGG-style 更 sharp 吗？**
   `temporal_concentration` 应该是 Tempo < EGG（更小更好）。
3. **Tempo-style 的 p(yes) 和 logit_diff 与样本相关性匹配吗？**
   query 与视频明显相关 → p(yes) 高、logit_diff 正大。
4. **空间 heatmap 落在 query 提到的物体/动作上吗？**
   query="person opens a door" → 期望 saliency 在门/手附近。

如果以上 4 点至少 3 点过线，Path 2 可行；否则需要换 prompt 或改 hook 位置再试。

---

## 实现要点

1. `model.visual` forward hook → `output.detach().requires_grad_(True)` → 让视觉 token 成为新的 leaf，LLM 反传时 grad 落在它上。
2. 模型参数全部 `requires_grad_(False)`，只对输入 grad，省显存。
3. Logits 在 fp32 下做 softmax 算 entropy，避免 bf16 数值问题。
4. Saliency 取 `‖∂H/∂V_i‖₂`（per-token L2 norm），照搬 EGG 的 `calc_grad`。
5. Reshape：`saliency.reshape(T, H_grid//spatial_merge, W_grid//spatial_merge)` —— Qwen2.5-VL 的视觉 token 经过 spatial-merge 才进 LLM。
