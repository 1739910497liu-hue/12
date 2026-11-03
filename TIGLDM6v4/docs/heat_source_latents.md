# 论文空间 Gate：不同潜在空间尺寸训练

通过改变 VQGAN 下采样倍率 f 比较潜在空间的空间分辨率，输入图像保持 256x256，
潜在通道数 d=4，码本容量 K=16384。使用论文解码器公式：

```text
g_l = sigmoid(alpha_l * raw_prior_l + bias_l)
f_mix = g_l * f_cond + (1 - g_l) * f_dec
```

所有组保留未归一化热源先验作为 Gate 输入、按层可学习参数以及现有扩散分支。
旧 `configs/heat_source_revision/f4.json` 等采用的门控不是本次论文空间 Gate。

| 实验 | f | 潜在张量 CxHxW | 每张图的码索引位置数 | 编码/解码尺度数 |
| --- | ---: | --- | ---: | ---: |
| paper_gate_f4 | 4 | 4x64x64 | 4096 | 3 |
| paper_gate_f8 | 8 | 4x32x32 | 1024 | 4 |
| paper_gate_f16 | 16 | 4x16x16 | 256 | 5 |
| paper_gate_f32 | 32 | 4x8x8 | 64 | 6 |

这里的 4 是潜在通道数，不是码本容量，也不是下采样倍率。
每个位置从同一组 16384 个码向量中选择一个，不同位置可以重复选择同一个码。

## 对比协议

f=4/8/16 除 `factor`、实验名和独立的第一阶段分组名外，配置与
`configs/heat_source_paper_gate/paper_gate_spatial.json` 完全相同。
f=32 还将 `unet_mult` 从 `[1,2,2,4,4]` 改为 `[1,2,2,4]`，避免 8x8 潜在空间
经过四次下采样变为零尺寸。此调整仅适用于 f=32，其余三组的配置与权重兼容性不变。
第一阶段 50000 个优化器更新，第二阶段 100000 个；batch_size=2、accumulate=4。
固定同一数据划分、损失权重、学习率、训练预算、采样协议和训练种子集合。
先使用 seed=42 检查，再以 42、123、2026 重复，报告实测均值和标准差。

现有模型通过以下通道倍率列表实现不同下采样倍率：

```text
f=4:  [1, 2, 4]
f=8:  [1, 2, 4, 4]
f=16: [1, 2, 4, 4, 4]
f=32: [1, 2, 4, 4, 4, 4]
```

因此尺度数、残差块总数、门控层数和总参数量会随 f 改变。这是现有实现下的
“下采样倍率/潜在空间分辨率消融”，不能声称网络深度和参数量完全相同。
f=4/8/16 的 Diffusion U-Net 保持五个尺度，对应瓶颈空间大小为 4x4、2x2、1x1。
f=32 使用四个尺度，空间路径为 8→4→2→1；第二阶段参数量也随该适配改变。
本组保持 Gate 公式和训练超参数一致，不宣称四组第二阶段网络深度和参数量相同。
模型构造时会检查潜在边长是否满足 U-Net 的下采样要求，并明确拒绝不兼容配置。
改变 f 也改变各个注意力层对应的绝对空间分辨率。
所有组使用相同 seed 不代表形状不同的模型共享参数逐元素初始化相同。

64x64 潜在空间通常需要更高的计算和显存开销，不能根据 32x32 的速度直接给出
精确耗时。代码不会为了某一组自动改变 batch 或累积步数；如需调整有效 batch，
应为所有组采用一致的新协议并使用新的输出目录。

## 部署

将 `paper_gate_latent_f32_update_20260928.zip` 上传到服务器项目根目录，在已经安装好
依赖的虚拟环境下运行：

```bash
cd /root/autodl-tmp/TIGLDM6v3-3
unzip -o paper_gate_latent_f32_update_20260928.zip
```

包中包含四组配置和入口、支持 factor=32 的 models.py，以及论文空间 Gate 依赖文件。
必须同步更新模型代码，不能只复制 factor=32 的 JSON 到旧版代码中。
不包含数据或训练权重，也不会复用不同 f 的检查点。

## 训练命令

只训练第一阶段，依次完成四组，每组 50000 步，不需要 CLIP：

```bash
python scripts/heat_source_latents.py stage1 --factors 4 8 16 32 --seeds 42 --precision bf16
```

仅训练 64x64 潜在空间：

```bash
python scripts/heat_source_latents.py stage1 --factors 4 --seeds 42 --precision bf16
```

如果已有 f=4/8/16，只补充新增的 8x8 潜在空间：

```bash
python scripts/heat_source_latents.py stage1 --factors 32 --seeds 42 --precision bf16
```

匹配的第一阶段完成后训练第二阶段，每组 100000 步：

```bash
python scripts/heat_source_latents.py stage2 --factors 4 8 16 32 --seeds 42 --precision bf16 --clip-path model/clip/clip-vit-base-patch32
```

连续训练两阶段，每组完成 stage1 后再训练本组 stage2：

```bash
python scripts/heat_source_latents.py both --factors 4 8 16 32 --seeds 42 --precision bf16 --clip-path model/clip/clip-vit-base-patch32
```

正式三种子实验把 `--seeds 42` 改为 `--seeds 42 123 2026`。
单种子四组第一阶段合计 200000 步，第二阶段合计 400000 步；三种子再乘 3。
多个模型依次训练，不会自动并行占用同一 GPU。

默认数据和输出路径为：

```text
--data-root datasets/ss304
--manifests datasets/ss304_manifests
--out outputs/ss304_paper_gate_latents
```

可通过以上参数覆盖路径，添加 `--dry-run` 只打印命令，不执行训练。
已存在阶段 `last.pt` 时自动恢复，precision、workers、配置和数据必须与检查点一致。
无检查点的非空阶段目录会被拒绝，防止意外混合旧运行。

输出示例：

```text
outputs/ss304_paper_gate_latents/full/paper_gate_f4/seed_42/stage1/last.pt
outputs/ss304_paper_gate_latents/full/paper_gate_f4/seed_42/stage1/best.pt
outputs/ss304_paper_gate_latents/full/paper_gate_f4/seed_42/stage2/last.pt
outputs/ss304_paper_gate_latents/full/paper_gate_f4/seed_42/stage2/best.pt
```

第二阶段必须使用同 f、同种子、同配置和同数据清单、完成预算的 `stage1/last.pt`。
不要在训练第二阶段后替换其第一阶段文件，也不要使用 `--smoke` 做正式对比；
原入口会用它改写训练预算、码本容量等设置。

## 统计与评价

```bash
python scripts/heat_source_latents.py status --factors 4 8 16 32 --seeds 42 --csv outputs/ss304_paper_gate_latents/latent_status.csv
```

状态入口读取 run.json 和 metrics.jsonl，核对完整实验配置（允许 precision/workers
的运行时覆盖）。缺失 metadata、混入旧 Gate 或不同 f 的日志会报错。
报告实际步数、最近及最佳验证误差、潜在尺寸、每张图的码位置数量，以及整个验证集
的码本活跃数量、利用率、困惑度。只容忍训练进程尚未写完的末行，不虚构缺失结果。
run.json 有记录时还输出实际模型参数量；日志有记录时输出当前训练会话的用时和
峰值显存。会话用时不是跨断点恢复累计的总训练时长，不能把它误写成完整训练成本。
`budget_logged` 表示日志到达预算，不代表测试通过或模型质量已验证。

不同 f 的验证图像数相同，但总码位置数不同（4096/1024/256/64 每张），比较码本
使用率时要同时报告这一差异。不能把利用率升高直接视为重建或生成效果变好。

第一阶段评价 PSNR、SSIM、LPIPS、重建 MAE、码本利用率和困惑度。
若论证最终生成质量，每个 f 都需第二阶段，评价 FID、KID，以及运行时间和显存。
只论证第一阶段重建则无需第二阶段。当前大损失/码本使用集中问题不会仅因新增这组
配置而自动解决，需要分项损失和实测评价；不预设哪种潜在尺寸更优。

第一阶段全测试集重建示例（训练完成后执行，所有组统一使用 last.pt）：

```bash
(
set -euo pipefail
cd /root/autodl-tmp/TIGLDM6v3-3
LATENT_RUN=outputs/ss304_paper_gate_latents/full/paper_gate_f4/seed_42
test -f "$LATENT_RUN/stage1/last.pt"
LATENT_EVAL=$(mktemp -d outputs/latent_f4_reconstruction_XXXXXX)
cp "$LATENT_RUN/stage1/last.pt" "$LATENT_EVAL/stage1_last.pt"
python scripts/heat_source.py sample --checkpoint "$LATENT_EVAL/stage1_last.pt" --data-root datasets/ss304 --manifests datasets/ss304_manifests --device cuda --sample-seed 2026 --out "$LATENT_EVAL/stage1"
python scripts/evaluate_heat_source.py "$LATENT_EVAL/stage1" --device cuda
echo "Evaluation output: $LATENT_EVAL"
)
```

第二阶段全测试集生成示例（同一规则选择验证噪声误差最小的 best.pt）：

```bash
(
set -euo pipefail
cd /root/autodl-tmp/TIGLDM6v3-3
LATENT_RUN=outputs/ss304_paper_gate_latents/full/paper_gate_f4/seed_42
test -f "$LATENT_RUN/stage1/last.pt"
test -f "$LATENT_RUN/stage2/best.pt"
LATENT_EVAL=$(mktemp -d outputs/latent_f4_generation_XXXXXX)
cp "$LATENT_RUN/stage2/best.pt" "$LATENT_EVAL/stage2_best.pt"
python scripts/heat_source.py sample --checkpoint "$LATENT_EVAL/stage2_best.pt" --first-stage "$LATENT_RUN/stage1/last.pt" --data-root datasets/ss304 --manifests datasets/ss304_manifests --clip-path model/clip/clip-vit-base-patch32 --device cuda --sample-seed 2026 --out "$LATENT_EVAL/stage2"
python scripts/evaluate_heat_source.py "$LATENT_EVAL/stage2" --device cuda
echo "Evaluation output: $LATENT_EVAL"
)
```

不传 --limit 时遍历整个测试集；不同组固定采样种子 2026、DDIM 步数 50。
第二阶段不报告随机生成图片与单张真实图片配对的重建误差。
现有评价脚本将实际参数量、训练时间、峰值显存与图像指标一同保存到 evaluation.json。
