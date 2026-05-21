# Qwen2.5-VL 自适应时间 Token 分配

本项目用于验证 Qwen2.5-VL 视频时间定位中的自适应视觉 token 分配。

baseline 使用均匀抽帧；adaptive 会先把视频切成多个 segment，对每个 segment 计算 `logit_diff` 和 `entropy`，再给不同 segment 分配不同数量的帧/token。

为了让模型知道非均匀采样后的真实时间位置，`modeling_qwen2_5_vl.py` 增加了 `video_time_grid_ts`，把每个 temporal grid 的真实时间戳送入 Qwen2.5-VL 的 temporal RoPE。

## 文件说明

```text
qwen2_5_vl/
├── modeling_qwen2_5_vl.py
├── validate.py
├── validate_charades_sta.py
└── validate_activitynet_captions.py
```

```text
modeling_qwen2_5_vl.py          修改后的 Qwen2.5-VL modeling 文件
validate.py                     通用验证逻辑
validate_charades_sta.py         Charades-STA 验证入口
validate_activitynet_captions.py ActivityNet-Captions 验证入口
```

运行时如果看到下面日志，说明已经成功使用项目里的 modeling 文件：

```text
[INFO] Loaded project-local Qwen2.5-VL modeling source: .../qwen2_5_vl/modeling_qwen2_5_vl.py
```

## 环境

当前服务器环境：

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

## 修改数据和模型路径

Charades-STA 路径在：

```text
qwen2_5_vl/validate_charades_sta.py
```

修改文件顶部：

```python
DEFAULT_ANNOTATION = Path(...)
DEFAULT_VIDEO_ROOT = Path(...)
```

ActivityNet-Captions 路径在：

```text
qwen2_5_vl/validate_activitynet_captions.py
```

修改文件顶部：

```python
DEFAULT_ANNOTATIONS = [...]
DEFAULT_VIDEO_ROOT = Path(...)
```

模型 checkpoint 路径在：

```text
qwen2_5_vl/validate.py
```

搜索并修改：

```python
--ckpt-dir
```

## 修改验证数量

Charades-STA：

```text
qwen2_5_vl/validate_charades_sta.py
```

ActivityNet-Captions：

```text
qwen2_5_vl/validate_activitynet_captions.py
```

在对应文件顶部修改：

```python
DEFAULT_NUM_QUERIES = 1000
```

如果想跑全量，改成：

```python
DEFAULT_NUM_QUERIES = None
```

## 修改自适应分配参数

所有主要实验参数都在两个入口脚本顶部的 `DEFAULT_CONFIG` 里。

Charades-STA：

```text
qwen2_5_vl/validate_charades_sta.py
```

当前建议配置：

```python
DEFAULT_CONFIG = {
    "mode": "both",
    "total_frames": 64,
    "target_segments": 8,
    "min_frames_per_segment": 6,
    "max_frames_per_segment": 32,
    "segment_mode": "partition",
    "probe_nframes": 2,
    "final_square_size": 112,
    "probe_square_size": 56,
    "allocation_mode": "logit_low_entropy",
    "top_logit_frac": 0.5,
    "low_entropy_frac": 0.5,
    "min_selected_segments": 1,
}
```

ActivityNet-Captions：

```text
qwen2_5_vl/validate_activitynet_captions.py
```

当前建议配置：

```python
DEFAULT_CONFIG = {
    "mode": "both",
    "total_frames": 64,
    "target_segments": 8,
    "min_frames_per_segment": 4,
    "max_frames_per_segment": 32,
    "segment_mode": "partition",
    "probe_nframes": 4,
    "final_square_size": 112,
    "probe_square_size": 56,
    "allocation_mode": "continuous",
    "entropy_direction": "low",
    "score_alpha": 0.7,
    "score_beta": 0.3,
    "temperature": 0.7,
}
```

常改参数：

```text
total_frames              每条视频正式问答阶段的总帧数
target_segments           视频切成多少个 segment
probe_nframes             每个 segment probe 用几帧
min_frames_per_segment    每个 segment 至少保留几帧
max_frames_per_segment    每个 segment 最多分到几帧
allocation_mode           adaptive 分配方式
final_square_size         正式问答阶段每帧尺寸
probe_square_size         probe 阶段每帧尺寸
```

## 运行

进入代码目录：

```bash
cd /ssd/cht/projects/entropy_exp/qwen2_5_vl
```

运行 Charades-STA：

```bash
python validate_charades_sta.py
```

运行 ActivityNet-Captions：

```bash
python validate_activitynet_captions.py
```

输出默认保存在：

```text
qwen2_5_vl/outputs/qwen25vl_adaptive_temporal/
```

## 注意事项

1. baseline 和 adaptive 对同一条视频必须使用相同 `total_frames`。
2. adaptive 必须传入 `video_time_grid_ts`，否则非均匀 token 的时间位置会错。
3. 增大 `total_frames` 会增加时间 token。
4. 增大 `final_square_size` 会明显增加视觉 token 和显存。
5. Charades 当前更适合 `logit_low_entropy`。
6. ActivityNet 当前更适合 `continuous`。
7. GitHub 不上传 checkpoint、原始数据、outputs 结果、日志和 pycache。
