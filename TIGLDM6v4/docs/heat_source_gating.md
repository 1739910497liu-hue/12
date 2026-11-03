# 解码器门控机制消融

本实验固定热源模板、扰动、物理特征投影、注入位置和主干训练协议，只改变解码器的门值。`configs/heat_source_revision/full.json` 仍使用原有的二值门控；本次没有将其改为 `sigmoid(alpha*M+b)`。

共同融合式为 `h_l' = h_l + g_l * P_l(T_l, M_l)`。G1 是半强度的残差先验注入，不能写成两路特征的平均融合 `(h_l + P_l)/2`。

| 组别 | 门值 | 配置 | 门值新增可训练参数（factor=8） |
| --- | --- | --- | --- |
| G1 | 恒定 `g_l=0.5`，全空间相同 | `configs/heat_source_gating/g1_fixed.json` | 0 |
| G2 | `g_l=sigmoid(beta_l)`，每个分辨率层级一个标量，初始为 0.5 | `configs/heat_source_gating/g2_scalar.json` | 4 |
| G3 | 原有 `g_l=M_l`，非可学习二值空间掩码 | `configs/heat_source_revision/full.json` | 0 |

256 像素输入、8 倍压缩时，四个注入位置分别为解码器的 32、64、128、256 分辨率层级，均在该层级残差块之前。模板 `[T,M]` 在三个组中保持相同，物理投影均为 `Conv3x3 -> SiLU -> Conv1x1`。G1/G2 使用相同随机种子时，共享网络参数初始化相同，初始门值也相同。G2 的四个新增标量不消耗随机初始化序列。

本消融改变的只有解码器门控。第二阶段扩散输入处的门控仍为原有 `M`，三组一致。不能据此声称已经消融了扩散模型的所有门控位置。

## 先完成第一阶段

在服务器项目根目录运行：

```bash
conda activate tigldm-blackwell
bash scripts/heat_source_gating_commands.sh stage1
```

这条命令依次训练 G1、G2 的第一阶段，各 50,000 个优化器步骤，默认 seed=42。它不启动第二阶段，也不需要 CLIP。数据默认使用 `datasets/ss304` 和 `datasets/ss304_manifests`，与当前 SS304 训练相同。有断点时自动恢复，没有断点时从头训练。

多种子训练：

```bash
HEAT_SEEDS="42 123 2026" bash scripts/heat_source_gating_commands.sh stage1
```

默认 G1/G2 输出目录：

```text
outputs/ss304_gate_ablation/full/g1_fixed/seed_42/stage1/last.pt
outputs/ss304_gate_ablation/full/g2_scalar/seed_42/stage1/last.pt
```

G3 默认复用正在训练的原模型第一阶段：

```text
outputs/ss304_category_numeric/full/full/seed_42/stage1/last.pt
```

只有当 G3 第一阶段已完成、数据清单/随机种子/模型/精度/训练预算和其他训练超参数一致时，才可作为同协议对照。每个新增种子都需要对应种子的 G3。未完成的第 50 步检查点不能用于正式消融表。原 `suite` 命令仍会自动训练两个阶段；它的行为没有改变。

## 重建指标与图像

三组第一阶段均完成后运行：

```bash
bash scripts/heat_source_gating_commands.sh evaluate
```

评估使用相同测试集计算 PSNR、SSIM、LPIPS，并输出原图、重建图、绝对误差图以及每个分辨率层级的真实门控图。误差和门值可视化使用固定标度，G1/G2 的常数图不会逐图归一化成看似空间变化的热图。当前二值 G3 的门图也是其实际二值掩码。

默认使用完成训练预算的 `last.pt`，避免混用不同早停步数。若主动更改评估协议，应对所有组一致执行并记录。图像展示不替代完整测试集指标，多种子均值和标准差也不能由单次运行代替。

自定义目录通过环境变量传入：

```bash
HEAT_DATA_ROOT=/root/autodl-tmp/TIGLDM6v3-3/datasets/ss304 \
HEAT_MANIFESTS=datasets/ss304_manifests \
HEAT_G3_RUNS=outputs/ss304_category_numeric \
HEAT_GATE_EVAL=outputs/ss304_gate_evaluation \
bash scripts/heat_source_gating_commands.sh evaluate
```

## 可选的最终生成实验

第一阶段的重建指标只能支持重建质量结论。若需要比较最终生成效果，再训练 G1/G2 各自匹配的第二阶段：

```bash
HEAT_CLIP_PATH=/root/autodl-tmp/TIGLDM6v3-3/model/clip/clip-vit-base-patch32 \
bash scripts/heat_source_gating_commands.sh stage2
```

这会为每组加载其自己的已完成第一阶段，冻结 VQ/解码器，训练 100,000 步条件扩散。代码会拒绝把 G3 第一阶段权重混用于 G1/G2。随后可沿用 `scripts/evaluate_heat_source_suite.py` 对第二阶段生成结果计算 FID/KID；G3 同样需要匹配的第二阶段、种子、数据清单和采样协议。第一阶段门控消融本身不构成生成 FID 已改善的证据。
