# Qwen2.5-VL 自适应时间 Token 分配

这个仓库只保留代码和运行说明，不上传具体实验结果。

项目实现了一个 Qwen2.5-VL 视频时间定位实验：baseline 使用均匀抽帧，adaptive 会先对视频 segment 做 probe，根据 `logit_diff` 和 `entropy` 给不同 segment 分配不同数量的视觉帧/token。为了让模型知道非均匀采样后的真实时间位置，`modeling_qwen2_5_vl.py` 增加了 `video_time_grid_ts`，把每个 temporal grid 的真实时间戳送入 temporal RoPE。

## 目录

```text
qwen2_5_vl/
├── modeling_qwen2_5_vl.py
├── validate.py
├── validate_charades_sta.py
└── validate_activitynet_captions.py
```

文件作用：

```text
modeling_qwen2_5_vl.py          Qwen2.5-VL modeling patch，支持 video_time_grid_ts
validate.py                     通用验证逻辑：probe、分配预算、推理、评测
validate_charades_sta.py         Charades-STA 入口
validate_activitynet_captions.py ActivityNet-Captions 入口，默认读 val1 + val2
```

运行时应看到：

```text
[INFO] Loaded project-local Qwen2.5-VL modeling source: .../qwen2_5_vl/modeling_qwen2_5_vl.py
```

如果没有这行，说明没有用到本项目的 modeling patch。

## 环境

当前验证环境：

```text
python 3.10
torch 2.5.1+cu121
transformers 4.49.0
accelerate
qwen-vl-utils
decord
opencv-python
pillow
numpy
tqdm
ffmpeg / ffprobe
```

推荐安装：

```bash
conda create -n qwen25vl python=3.10
conda activate qwen25vl
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.49.0 accelerate qwen-vl-utils decord opencv-python pillow numpy tqdm
```

## 修改路径

如果换机器，先改两个入口脚本顶部的路径。

Charades-STA：

```python
DEFAULT_ANNOTATION = Path("...")
DEFAULT_VIDEO_ROOT = Path("...")
```

ActivityNet-Captions：

```python
DEFAULT_ANNOTATIONS = [
    Path("...val1.json"),
    Path("...val2.json"),
]
DEFAULT_VIDEO_ROOT = Path("...")
```

模型路径在 `validate.py` 的参数默认值里：

```python
--ckpt-dir
```

也可以运行时传入：

```bash
--ckpt-dir /path/to/Qwen2.5-VL-7B-Instruct
```

## 修改参数

主要改两个入口脚本里的：

```python
DEFAULT_NUM_QUERIES
DEFAULT_CONFIG
```

常用参数说明：

```text
DEFAULT_NUM_QUERIES
默认验证多少条 query。None 表示全量。

mode
baseline：只跑均匀抽帧
adaptive：只跑自适应抽帧
both：同一批 query 同时跑 baseline 和 adaptive

total_frames
正式问答阶段每条视频总帧数。baseline 和 adaptive 必须一样。

target_segments
先把视频切成多少个 segment 做 probe 和预算分配。

probe_nframes
每个 segment probe 时抽几帧。

min_frames_per_segment
adaptive 中每个 segment 至少保留多少帧，避免非 selected segment 被饿死。

max_frames_per_segment
adaptive 中单个 segment 最多能拿到多少帧。

final_square_size
正式问答阶段每帧 resize 到多大。增大它会显著增加视觉 token 和显存。

probe_square_size
probe 阶段每帧 resize 到多大。通常可以比 final 小。

allocation_mode
logit_low_entropy：先选高 logit_diff，再选低 entropy 的 segment 加预算。
continuous：根据连续分数给所有 segment 平滑分配预算。

top_logit_frac / low_entropy_frac
logit_low_entropy 模式下使用，控制选多少 segment。

entropy_direction / score_alpha / score_beta / temperature
continuous 模式下使用，控制 logit_diff 和 entropy 如何组合，以及分配有多尖锐。
```

当前建议配置：

Charades-STA：

```python
"total_frames": 64,
"target_segments": 8,
"min_frames_per_segment": 6,
"max_frames_per_segment": 32,
"allocation_mode": "logit_low_entropy",
```

ActivityNet-Captions：

```python
"total_frames": 64,
"target_segments": 8,
"min_frames_per_segment": 4,
"max_frames_per_segment": 32,
"allocation_mode": "continuous",
"entropy_direction": "low",
"score_alpha": 0.7,
"score_beta": 0.3,
"temperature": 0.7,
```

## 运行

进入项目目录：

```bash
cd /ssd/cht/projects/entropy_exp/qwen2_5_vl
```

先看空闲 GPU：

```bash
nvidia-smi
```

小样本 smoke test：

```bash
CUDA_VISIBLE_DEVICES=0 python validate_charades_sta.py \
  --num-queries 5 \
  --save-details

CUDA_VISIBLE_DEVICES=0 python validate_activitynet_captions.py \
  --num-queries 5 \
  --save-details
```

跑指定数量 query：

```bash
CUDA_VISIBLE_DEVICES=0 python validate_charades_sta.py \
  --num-queries 300 \
  --summary-json outputs/qwen25vl_adaptive_temporal/charades_summary.json

CUDA_VISIBLE_DEVICES=0 python validate_activitynet_captions.py \
  --num-queries 300 \
  --summary-json outputs/qwen25vl_adaptive_temporal/activitynet_summary.json
```

如果想保存每条 query 的详细信息，加：

```bash
--save-details \
--output-jsonl outputs/qwen25vl_adaptive_temporal/details.jsonl
```

后台运行示例：

```bash
nohup bash -lc 'CUDA_VISIBLE_DEVICES=0 python validate_charades_sta.py --num-queries 300 --save-details' \
  > charades.log 2>&1 &
```

## 输出

默认保存 summary，包含：

```text
mIoU
Recall@1_IoU=0.3
Recall@1_IoU=0.5
Recall@1_IoU=0.7
```

加 `--save-details` 后会额外保存逐 query JSONL，常用来检查：

```text
frame_counts 是否加和为 total_frames
unused_frame_budget 是否为 0
baseline/adaptive video_llm_tokens 是否一致
baseline/adaptive video_grid_thw 是否一致
adaptive grid_timestamps 长度是否等于 grid_t
```

## 注意事项

1. baseline 和 adaptive 对同一条视频必须使用相同 `total_frames`。
2. adaptive 必须传入 `video_time_grid_ts`，否则非均匀 token 的时间位置会错。
3. 增大 `total_frames` 会增加时间 token；增大 `final_square_size` 会更明显增加显存。
4. Charades 更适合较保守的 hard selection；ActivityNet 长区间多，continuous allocation 更稳定。
5. GitHub 不上传 checkpoint、原始数据集、outputs 结果、日志和 pycache。
