# 实验记录

每版 PoC 的核心机制 / 流程 / 代码 / 输出 / 关键结论，简洁记录。

---

## v1 — 整段视频一次 forward + EGG-style 全 vocab entropy 梯度

**日期**:2026-05-22
**核心机制**:对完整视频做一次 forward，对最后一 token 的 Shannon entropy 反传到视觉 token，得到时空 saliency。
**流程**：
```
全视频 → 16 帧 → letterbox 112² → processor
                          │
                          ▼
                model.visual (forward hook 注入 detach + requires_grad)
                          │
                          ▼
              LLM forward → logits[最后 token]
                          │
                          ▼
              H = −Σ p log p  (full vocab)
              H.backward()
                          │
                          ▼
              video_embeds.grad → ‖·‖₂ → reshape (T, H, W)
```
两种 prompt 各跑一次：
- EGG-style：`build_temporal_prompt`（强制 `start: x, end: y` 格式）
- Tempo-style：`build_probe_prompt`（A/B 二元）

**代码**:`experiments/poc_gradient_viz/poc_visualize.py`（旧版本）
**输出**:`experiments/poc_gradient_viz/outputs/sample_00_3MSZA.png`（已被 v2 覆盖）

**关键发现**：
1. 反传机制能跑通，∂H/∂V 在 GT 内的某一帧上确实定位到合理空间区域（门口/开关）
2. **时间分辨率太粗**：T=4 时间槽（16 帧 / temporal_patch=2 / 1 段）→ 曲线只 4 个点，看不出峰
3. 第一个时间槽出现 boundary artifact（saliency 值显著偏高）
4. EGG-style 第一 token entropy 极低（H≈0.18，模型几乎确定输出 "start"）→ 梯度信号被高 confidence 淹没

---

## v2 — 分段 probe + EGG 自由 prompt + Tempo A/B token

**日期**:2026-05-22
**核心机制**:把视频按 5s 切段，每段独立 probe，但仍然分两个 prompt 各跑一次 forward+backward。Tempo signal 用了 "A/B" multiple-choice 字母作为 token。
**流程**：
```
视频 → segment_len 切段（默认 5s）
每段:
  └─ extract_frames_in_window(s, e, frames_per_segment=8)
  └─ Tempo prompt (A/B)  → forward + backward(entropy) → saliency 立方体
  └─ EGG free-form prompt → forward + backward(entropy) → saliency 立方体
```

**代码**:`experiments/poc_gradient_viz/poc_visualize.py`（v2）
**输出**:`experiments/poc_gradient_viz/outputs/sample_00_3MSZA.png`（v2 版本，已被 v3 覆盖）

**关键发现**：
1. ✅ 分段后 per-segment 信号变干净，p_yes 在不同段上能区分（最大 0.41 vs 最小 0.16）
2. ❌ **Tempo 排序部分有效但绝对值全错**:模型对所有段都说 "No"（p_yes < 0.5），且 GT 段 (seg5) 反而最低。说明 zero-shot Qwen2.5-VL 没被 Tempo 那样训练，A/B prompt 校准有问题
3. ❌ **EGG free-form prompt 给的梯度退化**:6 段热图几乎相同 → free-form 第一 token 不依赖 query，梯度只反映通用视觉偏置
4. ❌ **可视化 bug**:每段 T_seg=4 个时间槽被 `mean(axis=0)` 平均成 1 张图，丢了段内时间结构
5. ❌ **prompt 不对齐 Tempo paper**:用 A/B 和自编的 "Does X happen?" 问句，跟 Tempo 官方 system+user prompt 完全不同

---

## v3 — Tempo 官方 prompt 对齐 + 单 forward 双梯度 + 时间槽展开

**日期**:2026-05-27
**核心机制**:每段一次 forward，**用 `torch.autograd.grad` 同时抽两个梯度**:`∂(logit_Yes − logit_No)/∂V`（Tempo router 信号）和 `∂H/∂V`（EGG 全 vocab entropy 信号）。**Prompt 严格对齐 Tempo `mm_datautils.py:266-271` 官方 inference prompt 1**（system role + user role）。

**流程**：
```
视频 → segment_len 切段
每段:
  └─ extract_frames_in_window(s, e, frames_per_segment=8)
  └─ build_inputs(prompt = Tempo official inference prompt 1)
  └─ forward hook 在 model.visual 上注入 detach + requires_grad
  └─ outputs = model(**inputs)                    # 一次 forward
  └─ logits = outputs.logits[0, -1, :]
     logit_diff = logits[Yes] - logits[No]
     entropy   = -Σ softmax(logits) · log softmax(logits)
  └─ grad_diff    = torch.autograd.grad(logit_diff, V, retain_graph=True)[0]   # Tempo signal
  └─ grad_entropy = torch.autograd.grad(entropy,    V, retain_graph=False)[0]  # EGG signal
  └─ saliency_diff_3d, saliency_entropy_3d = each ‖·‖₂ → reshape (T_seg, H, W)
```

**Prompt 严格对齐**（chat template messages 结构）：
```
[system] "You are a query-conditioned visual compressor. Store in the provided memory tokens
          the minimal visual information needed to answer the Query. Ignore irrelevant details.
          Now, before compressing, answer exactly 'Yes' or 'No': is this segment relevant to the Query?"
[user]   <video> + "\nQuery:\n{query}"
```
（注:Tempo 训练版 prompt 0 不带 Yes/No 问句；assistant 前缀变种 `Scanning for target features...` 暂未启用。）

**可视化布局**（解决 v2 的图像数量问题）：
```
                seg0 [0-5s]              seg1 [5-10s]    ...    seg5 [25-31s]
                t1  t2  t3  t4         t1  t2  t3  t4          t1  t2  t3  t4
∂(logit_diff)/∂V [□][□][□][□]  ||    [□][□][□][□]    ||  ...   [□][□][□][□]
∂H/∂V            [□][□][□][□]  ||    [□][□][□][□]    ||  ...   [□][□][□][□]
[per-seg p(yes) bar chart, 全宽, GT 区间绿色阴影]
[per-seg ∂H/∂V 总和 + ∂(logit_diff)/∂V 总和 曲线]
```

**代码**:`experiments/poc_gradient_viz/poc_visualize.py`
**输出**:`experiments/poc_gradient_viz/outputs/`

**预期 vs v2**：
- forward 数减半（每段 1 次 forward → 2 次 grad 调用）
- 每段 T_seg=4 个时间槽完全展开 → 能看到段内空间证据如何随时间移动
- prompt 对齐 paper → p_yes 校准应改善，至少排序信号应更稳定
- 同时拿到 logit-grad 和 entropy-grad 两个 saliency → paper 主消融轴

**实测结果**（sample 3MSZA，GT [24.3, 30.4]）：
```
seg0 [ 0.0,  5.2]  p(yes)=0.119  Δ=-2.000  H=0.395
seg1 [ 5.2, 10.3]  p(yes)=0.165  Δ=-1.625  H=0.479
seg2 [10.3, 15.5]  p(yes)=0.165  Δ=-1.625  H=0.482
seg3 [15.5, 20.7]  p(yes)=0.148  Δ=-1.750  H=0.454
seg4 [20.7, 25.8]  p(yes)=0.245  Δ=-1.125  H=0.599   ← 部分 GT, 排序最高 ✓
seg5 [25.8, 31.0]  p(yes)=0.042  Δ=-3.125  H=0.193   ← 主要 GT, 排序最低 ✗
```

**关键发现**：
1. ✅ 可视化布局正确，6 段 × T_seg=4 时间槽全部展开，能看到段内空间证据演化
2. ✅ 单 forward 双 grad 跑通，**两个 saliency map 在视觉上明显不同**（Tempo grad 更稀疏聚焦，EGG grad 更稳定弥散）
3. ✅ `‖∂(Δlogit)/∂V‖` 和 `‖∂H/∂V‖` 总量曲线**形状完全不同** — 前者在 seg5 (GT) 升到最大，后者在 seg5 反而最低 → **两个梯度信号确实正交**
4. ❌ Tempo 官方 prompt 反而让 p(yes) 整体更低（最高 0.245 < v2 的 0.41）→ Qwen2.5-VL zero-shot 在 Tempo paper 的 prompt 下更倾向说 No
5. ⚠️ **关键反直觉现象**:seg5（GT 主体）的 p(yes)=0.04 最低、H=0.19 最低（极度自信 No），但 ‖∂(Δlogit)/∂V‖ 总量最高 → 模型"自信地答错"，但视觉证据扰动会强烈改变这个决策。这是**梯度信号比 logit 标量更可靠**的强证据
6. ⚠️ ∂H/∂V 在 GT 段反而最弱：低 entropy → 弱 grad（meaningful gradient signal needs uncertainty）。说明 entropy gradient 在模型"过度自信"时失效

**v3 暴露的下一个问题**：
- Qwen2.5-VL 7B zero-shot 没经过 Tempo 那种 router 训练，logit 标量本身校准很差（系统性偏向 No）
- **梯度信号 vs logit 标量信号:看 ‖∂(Δlogit)/∂V‖ 比 σ(Δlogit) 更靠谱**——这恰好是我们 paper 的核心 claim
- 需要扩到 3-5 个样本验证 grad 信号的排序稳定性，而不是依赖 logit 排序

---

## v3 扩展 — 5 样本验证（同代码，更多样本）

**日期**:2026-05-27
**目的**:验证 v3 在单样本上观察到的"梯度信号比 logit 更可靠"现象是否能泛化到更多样本。
**代码**:同 v3，仅改 `--num-samples 5`
**输出**:`experiments/poc_gradient_viz/outputs/sample_00..04_*.png`

**5 样本 argmax 命中表（GT 范围内为命中）**：

| Sample | Query | GT | p_yes argmax | ‖∂(Δlogit)/∂V‖ argmax | ‖∂H/∂V‖ argmax |
|---|---|---|---|---|---|
| 0 | 3MSZA — turn light on | [24.3, 30.4] | seg4 部分✓ | seg5 ✓ | seg0 ✗ |
| 1 | AMT7R — picture wall | [4.3, 12.5] | seg3 ✗（seg1/2 也高） | seg0 ✗ | seg1 ✓ |
| 2 | YVKIV — puts bag | [4.4, 9.2] | seg5 ✗ | seg5 ✗ | seg5 ✗ |
| 3 | VXJS4 — walks doorway | [0.0, 3.4] | **seg0 ✓** | seg5 ✗ | seg5 ✗ |
| 4 | GBD1Y — closes door | [26.2, 31.3] | **seg6 ✓** | seg0 ✗ | seg0 ✗ |
| | | **Strict 命中** | **3/5** | **1/5** | **1/5** |

**关键反转结论**：
1. ❌ **v3 单样本时认为"grad 比 logit 更可靠"是被 coincidence 骗了**。Sample 0 的 ‖∂(Δlogit)/∂V‖ 在 seg5 (GT) 取得峰值，看起来是"grad 知道 GT 在哪"，但实际上：
   - Sample 2、3 的 grad 也在 seg5 取得峰值（GT 不在那）
   - Sample 4 的 GT 在末尾，grad 反而 peak 在 seg0
   - → Grad 总量信号有**系统性 boundary artifact**:倾向 seg0 或最后一段，与 GT 无关
2. ✅ **logit 标量 p_yes 命中率反而最高**:3/5 strict + 1/5 partial（v3 单样本时被低估）
3. ❌ **‖∂(Δlogit)/∂V‖ 和 ‖∂H/∂V‖ 都只 1/5 命中**:作为段级 ranking 信号都失效
4. ⚠️ **boundary artifact 来源猜测**:
   - 每段独立 forward → 不是跨段的累积效应
   - 可能是 vision encoder 内部 temporal patch 边界处理（第一/最后 patch 的 padding/wrap）
   - 或是 letterbox padding（112² 黑边）在边界 token 上累积梯度
5. ⚠️ **段级总量 ≠ 全部信号**:可视化里**段内时间槽（T_seg=4 子格）展开的 spatial saliency 仍然看起来合理**,只是当我们 reduce 成段级标量（sum over space and time slots）时,boundary artifact 主导

**可能的下一步**（待选,等用户决定）：
- **A. 换聚合方式**:不用 sum,用 max-over-time-slots,或用 spatial concentration（spatial entropy of saliency）
- **B. 显式去 boundary**:丢掉 t0 和最后一个 t_slot 再 sum,看是否能修
- **C. 换 prompt**:试 Tempo training prompt 0（无 Yes/No 问句）或加 assistant 前缀 `Scanning for target features...`,看 logit 校准会不会改善
- **D. 拉长视频粒度**:这 5 样本都是 30s 短视频,可能 boundary 占比过大;到 ActivityNet 长视频上再看
- **E. 绕过段级标量,直接用空间 saliency**:每段 4 个 t-slot × spatial pattern 合成 query relevance,绕开段级 sum 的 artifact

**我的判断**:
本研究路径 (∂H/∂V on video) 在**段级标量**上当前不优于 logit 标量,但**段内时空 pattern** 仍有信息（看 PNG 第二行就能看到红色聚焦点）。**应该转向探索方向 E**:不要把每段 reduce 成 1 个标量做 ranking,而是把整段 saliency 拼成全视频时空 volume,做 EGG 那种 connected-component / elbow threshold 后处理直接读出 (start, end)。这才匹配 paper 立意:"saliency 几何就是答案"。
