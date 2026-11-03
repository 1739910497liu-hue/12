# 热源模板条件生成与五组消融实验

> 审稿意见修订后的主实验矩阵与命令请使用 [heat_source_revision.md](heat_source_revision.md)。本文件保留旧版方案及环境、数据准备说明；旧版结果不应与修订版混合汇总。

本方案用于 **从随机潜变量、热源模板及可选文本/数值条件生成焊接图像**。代码位于 `ldm/heat_source/`，训练入口为 `scripts/heat_source.py`。原有训练配置和模型文件保持原样，新的实验均从头训练。

目前交付的是可运行训练流程、实验配置和验证结果，尚未在 AutoDL 完成正式训练，不能据此报告模型性能提升。

## 1. 当前代码与新实验的关系

原目录的 SS304 配置采用 `SymmetricVQModel`、普通 `LatentDiffusion` 和 `Cf_r: 1.0`，没有实际接入热源掩码。原 `my_decoder_large2.py` 中存在图像修复特征融合 `h*m + h_condition*(1-m)`，其中条件特征来自遮挡后的原图；它不适合作为无需原图的生成分支直接使用。

新流程复用本仓库 Encoder、Decoder 骨干和 UNet，在第一阶段解码器的各尺度、第二阶段 UNet 输入端接入物理模板。模板包含归一化热源强度 `T` 与阈值区域 `M` 两个通道，阈值为 `0.1`。

- 普通融合：`h_out = h + P(T, M)`。
- 掩码门控融合：`h_out = h + M * P(T, M)`。

两组使用同样的模板、投影层、输入通道和参数量，只有空间门控不同。这里的门控是固定物理掩码控制的残差融合，不是可学习的 sigmoid 门控，也不是声称原检查点已采用该结构。生成阶段解码器只接收潜变量和物理模板，不接收目标图像或目标图像特征。

模板保留 `imagenet.py` 原实现的两个完整椭圆高斯分布叠加，默认尺度为 `(48,24)` 和 `(72,32)`，在 256 像素图像中心。原实现未按前后半平面分段；不能将其表述为严格的 Goldak 双椭球热传导求解器。强度归一化也会消除公共幅值 Q 的作用。电流、电压、焊速通过单独的数值条件分支进入，不用于伪造模板的绝对温度或热通量。

## 2. 实验矩阵

输入固定为 256×256，潜变量通道为 4，基准压缩因子 `f=8`，码本 `K=16384`。前四组固定为仅物理模板条件，没有文本和数值输入。

| 组别 | 设置与配置名 | 其余固定项 | 主要回答的问题 |
| --- | --- | --- | --- |
| 门控机制 | `fusion_plain`、`baseline` | 相同模板与拟定基准扰动；同 f/K 和参数量 | 增益是否来自空间门控？ |
| 模板扰动 | `perturb_none`、`perturb_small`、`baseline`、`perturb_large` | 门控、f=8、K=16384 | 增强幅度与质量/形态的关系 |
| 潜在压缩 | `compression_f4`、`baseline`、`compression_f16` | 门控、基准扰动、K=16384 | 重建细节、生成质量与计算开销的平衡 |
| 码本容量 | `codebook_8192`、`baseline`、`codebook_32768` | 门控、基准扰动、f=8 | 容量、利用率与重建/生成性能的关系 |
| 条件贡献 | `perturb_none`、`condition_text`、`condition_numeric`、`condition_both` | 相同无扰动模板、门控、f=8、K=16384；同一个冻结的第一阶段 | 文本、数值、联合输入分别带来多少增益？ |

压缩因子是每个空间轴的缩小倍数，三个潜网格分别为 `64×64`、`32×32`、`16×16`，对应 4096、1024、256 个位置。调整 f 会改变编解码层数，因此同步记录参数量、耗时与显存。所有 K/f 变体必须重新训练第一阶段及其匹配的第二阶段，不能复用不兼容检查点。

合并重复基准后共 **12 个独立实验设置**。每个种子需要 9 次第一阶段训练、12 次第二阶段训练。使用 `42、123、2026` 三个种子，共 **27 次第一阶段 + 36 次第二阶段 = 63 个训练任务**。

## 3. 模板扰动范围

原配置没有启用扰动，以下“基准”是本次拟定的实验起点，尚未验证最优。

| 档位 | 旋转角度 | 共同等比例缩放 | 中心平移（各轴占图像边长） |
| --- | --- | --- | --- |
| 无 | 0° | 1.00 | 0 |
| 小 | ±15° | 0.95–1.05 | ±2.5% |
| 拟定基准 | ±45° | 0.90–1.10 | ±5% |
| 大 | ±75° | 0.80–1.20 | ±10% |

对两个高斯分布施加相同变换，保持它们的相对形态。只扰动条件模板，不旋转目标图像。由“种子、训练轮次、样本 ID”决定扰动，同组同种子可复现。验证和测试统一使用固定的无扰动模板，避免把测试难度变化混入训练增强的比较。大扰动可能造成模板与图像热源位置不一致，这正是该组要验证的因素。

预览：`outputs/heat_source_design/template_perturbations.png`。重新生成：

```bash
python scripts/heat_source.py preview --out outputs/heat_source_design
```

## 4. 数据与条件标注

推荐数据根目录：

```text
datasets/ss304/
  train/<焊接过程或视频ID>/<图像文件>
  valid/<焊接过程或视频ID>/<图像文件>
  test/<焊接过程或视频ID>/<图像文件>
```

按焊接过程/视频分组划分，避免相邻帧同时出现在训练和测试中。数据准备脚本检查跨集合重复 group 和相同文件内容，不移动或删除原始图片。如果只有 train/valid，应先从未用于训练的独立焊接过程建立固定 test 集，再进行这批实验。不要把 valid 同时当成调参集和最终测试集。

准备 `datasets/ss304/metadata.csv`，所有实验使用同一套样本清单。字段为：

```csv
image,group,caption,current_A,voltage_V,speed_mm_s
```

- `image`：相对于数据根目录的完整路径，包含 train/valid/test 前缀。
- `group`：真实焊接过程/视频 ID；缺省时使用直接父目录名。不同类别下的同一焊接过程应使用同一 group。
- `caption`：该样本对应的报告文本。缺值保持空白，不能用统一默认句子代替。
- `current_A`、`voltage_V`、`speed_mm_s`：真实电流 A、电压 V、焊速 mm/s，均须为正值；`cm/min ÷ 6 = mm/s`。设备速度档位不是实际速度，必须有校准关系才能转换。

为了区分语义文本与数值支路的独立贡献，准备脚本将文本中的显式“数值+电流/电压/速度单位”替换为占位符，原文保留为 `caption_raw`。请审核无单位数字、文件名中的数字和特殊单位写法；它们可能绕过规则。若选择保留完整参数文本，则应将实验解释为“额外数值编码的边际作用”，不能声称数值信息只存在于数值分支。

数值均值和标准差只从训练集计算，验证/测试直接复用。三维数值分别映射为三个条件 token；文本使用冻结的 CLIP ViT-B/32，77 个 token。所有条件设置均保留 80 个 context 位置与同一 UNet 结构，禁用的分支替换为可学习的公共空条件。第一阶段在四组条件实验中完全共享。

本地检查发现 `H:/数据集/JSON/text/valid_numeric.json`、`valid_numeric.csv` 和 `train_text.json` 等候选资料，但其中存在缺值、路径表达差异和不同数据集混杂。它们没有被自动当作完整可信的三字段标注导入。本地 `H:/数据集/ss304` 当前只有 train/valid。请依据 AutoDL 上最终确认的数据和标注建立上述清单。

如果先准备图像清单，可暂不传 `--metadata`；脚本会输出 `metadata_inventory.csv`，保留缺失标注为空。前四组能够只用图像运行，第五组会在训练前检查标注。不要对四种条件设置分别丢弃不同样本。

## 5. AutoDL 环境与数据准备

在 AutoDL 上放置本项目，以下以 `/root/autodl-tmp/TIGLDM6v3` 为例。Windows 数据目录中的 junction 不会变成 Linux 上有效数据，需要将实际图像和标注上传到服务器对应目录。

```bash
cd /root/autodl-tmp/TIGLDM6v3
conda create -n heatabl python=3.10 -y
conda activate heatabl
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-heat-source.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

python scripts/prepare_heat_source.py \
  --data-root datasets/ss304 \
  --train-dir train --val-dir valid --test-dir test \
  --metadata datasets/ss304/metadata.csv \
  --out datasets/heat_source_manifests
```

PyTorch/CUDA 12.1 的版本组合来自 [PyTorch 官方版本表](https://docs.pytorch.org/get-started/previous-versions/)。LPIPS 和评估用 Inception 首次运行可能下载权重。CLIP 要提前放在本地目录；训练不会自动联网寻找它：

```bash
python -c "from transformers import CLIPTokenizer,CLIPTextModel; p='models/clip/clip-vit-base-patch32'; CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32').save_pretrained(p); CLIPTextModel.from_pretrained('openai/clip-vit-base-patch32').save_pretrained(p)"
```

如果服务器无法访问模型站点，可上传已经下载好的相同 CLIP 模型目录。默认要求其 hidden size=512，不能直接换为 768 维文本模型。

先做完整预检与小规模验证：

```bash
python scripts/heat_source.py preflight --stage 2 \
  --config configs/heat_source/condition_both.json \
  --data-root datasets/ss304 \
  --clip-path models/clip/clip-vit-base-patch32 --verify-images

python -m unittest discover -s tests -p test_heat_source.py -v

python scripts/heat_source.py suite \
  --experiments baseline --seeds 42 --smoke \
  --data-root datasets/ss304 --out outputs/heat_source \
  --device cuda --precision bf16 --workers 0
```

`--smoke` 使用小模型、小码本和两步训练，只验证链路。输出在 `smoke/`，正式结果汇总脚本拒绝使用这些检查点。不要将其损失、图片或指标作为论文结果。

## 6. 正式训练命令

五组分别运行的简短入口（同一基准自动复用）：

```bash
bash scripts/heat_source_experiment_commands.sh fusion
bash scripts/heat_source_experiment_commands.sh perturbation
bash scripts/heat_source_experiment_commands.sh compression
bash scripts/heat_source_experiment_commands.sh codebook
bash scripts/heat_source_experiment_commands.sh condition
```

全部运行用 `bash scripts/heat_source_experiment_commands.sh all`。默认路径为 `datasets/ss304`、`datasets/heat_source_manifests` 和 `models/clip/clip-vit-base-patch32`，可通过 `HEAT_DATA_ROOT`、`HEAT_MANIFESTS`、`HEAT_CLIP_PATH` 环境变量覆盖。

一条命令串行运行全部 12 个设置、3 个种子，自动共享第一阶段；支持中断后以同一命令续训：

```bash
bash scripts/run_heat_source_autodl.sh \
  datasets/ss304 \
  datasets/heat_source_manifests \
  models/clip/clip-vit-base-patch32
```

若暂时没有完整文本/数值标注，只运行前四组的 9 个唯一设置：

```bash
bash scripts/run_heat_source_autodl.sh \
  datasets/ss304 datasets/heat_source_manifests - \
  baseline fusion_plain perturb_none perturb_small perturb_large \
  compression_f4 compression_f16 codebook_8192 codebook_32768
```

补齐标注后，应在新输出目录或新实验批次中统一运行四种条件设置；如果 manifests 改变，续训检查会拒绝复用旧批次。仅补充缺失标注也会改变清单哈希，不会静默混用。

只运行一组门控消融：

```bash
python scripts/heat_source.py suite \
  --experiments fusion_plain baseline --seeds 42 123 2026 \
  --data-root datasets/ss304 --device cuda --precision bf16 \
  --out outputs/heat_source --resume
```

只运行条件贡献实验：

```bash
python scripts/heat_source.py suite \
  --experiments perturb_none condition_text condition_numeric condition_both \
  --seeds 42 123 2026 --data-root datasets/ss304 \
  --clip-path models/clip/clip-vit-base-patch32 \
  --device cuda --precision bf16 --out outputs/heat_source --resume
```

默认第一阶段 50,000 个优化器更新、第二阶段 100,000 个更新；micro batch=2，梯度累积=4，有效 batch=8。比较固定优化器更新数，不提前停止个别设置。学习率分别为 `4.5e-6`、`5e-5`，前 1000 步 warmup。第一阶段使用 L1+LPIPS+VQ 损失，10,000 步后增加固定权重 0.1 的对抗损失；第二阶段使用 epsilon 预测 L1 损失。训练参数均在各 JSON 中记录。

这是新的受控实验训练配方，不是原有对称 VQGAN 检查点的无改动复现。第一阶段和第二阶段都从头训练，避免旧权重带来的预训练差异。若需研究迁移微调，应另设所有组一致的初始化协议。

`last.pt` 是固定预算的正式比较检查点；`best.pt` 根据验证集留存以便诊断。后续阶段使用 `last.pt`，不挑选不同训练步数的最优模型混入主表。续训保存模型、优化器、调度器、数据顺序、随机状态、潜变量尺度和数据哈希。

若 4090 显存不足，应对所有设置统一降低 micro batch、相应增加累积，保持有效 batch=8，并使用新的输出目录重新开始这批比较。记录完整硬件与软件环境。

## 7. 生成、评价和结果汇总

对全部已完成的固定预算检查点生成与评价：

```bash
python scripts/evaluate_heat_source_suite.py \
  --runs outputs/heat_source/full \
  --out outputs/heat_source_evaluation \
  --data-root datasets/ss304 --manifests datasets/heat_source_manifests \
  --clip-path models/clip/clip-vit-base-patch32 --device cuda
```

正式采样从随机噪声开始，DDIM 50 步、eta=0、不使用 CFG。所有组使用相同测试 ID 和逐样本固定采样种子，测试模板不扰动。默认每个测试条件生成一张图，数量与测试集相同；若研究多样性，应为所有组统一增加采样种子并单独汇总。

- 第一阶段重建：PSNR↑、SSIM↑、LPIPS↓、MAE↓；验证日志另记录码本利用率和 perplexity。
- 第二阶段生成：FID↓、KID↓，由 [torch-fidelity 0.3.0](https://torch-fidelity.readthedocs.io/en/v0.3.0/api.html) 统一计算；报告测试样本数。
- 计算成本：参数量、训练耗时、每步耗时、峰值显存、生成一张图耗时。条件实验同时报告总参数和需优化的参数，文本编码器固定。
- 图像形态：输出固定亮度阈值下的亮区面积、重心和平均亮度，作为描述性检查。它们不等于热通量或温度误差；需物理标定或独立区域标注才能声称物理精度。

随机生成图像与某张真实图像不具有逐像素配对关系，因此第二阶段不会计算配对 PSNR/SSIM。不要仅凭较低 FID 断言条件遵循更准确；应在相同噪声和模板下改变报告/数值，结合独立检测器、形态标注或专家盲评评价条件一致性。

汇总默认输出：

```text
outputs/heat_source_evaluation/summary/results.csv
outputs/heat_source_evaluation/summary/results.json
outputs/heat_source_evaluation/summary/condition_contributions.json
```

也可以再次汇总：

```bash
python scripts/summarize_heat_source.py outputs/heat_source_evaluation \
  --out outputs/heat_source_evaluation/summary
```

主表报告三个训练种子的均值±样本标准差，不用 KID 内部子集的标准差代替跨训练种子的标准差。三个种子是起步设置，不能单凭这一样本量宣称统计显著。

对于越小越好的误差 E，文本增益=`E_none−E_text`，数值增益=`E_none−E_numeric`，联合增益=`E_none−E_both`；描述性交互量=`E_text+E_numeric−E_both−E_none`。正值表示联合改善超过单分支改善之和，负值表示收益重叠或相互抑制。这个量依赖所选指标，不等同于物理机理的因果证明。

## 8. 本地验证范围

自动测试覆盖模板可复现性、二值区域、门控行为、量化最近邻与梯度、f=4/8/16 尺寸、物理分支梯度、四种条件输入行为、固定测试模板、参数量一致、两阶段小模型训练、对抗更新、采样、续训一致性及检查点不匹配拒绝。文本测试使用本地随机小型 CLIP 组件，不下载真实预训练权重。

本地验证使用可用的 PyTorch 1.13.1 环境；AutoDL 的目标环境为上面锁定的 PyTorch 2.5.1。完整规模的 LPIPS 权重加载、预训练 CLIP、FID/KID 权重下载和 4090 BF16 训练需在 AutoDL 预检/试跑中验证。正式消融结果尚未产生。
