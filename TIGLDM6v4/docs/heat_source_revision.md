# 审稿意见对应的实验修订：reviewer_v2

本文件取代旧实验说明中的主实验矩阵；旧配置 `configs/heat_source/` 和旧输出不被覆盖。修订配置位于 `configs/heat_source_revision/`，输出默认写入 `outputs/heat_source_revision/`。

本轮已实现训练/评价入口和软件验证，正式 AutoDL 实验尚未运行。当前仅审阅了用户提供的审稿意见和相关代码，没有收到稿件正文、补充材料或作者回复，不能据此声称已核对论文全部公式。本文中的 full 是本次代码实现的完整模型，不自动等同于论文中既有的 MPLDM 检查点。

## 一、主要修改

| 审稿要求 | 此次修改 | 需要补入论文的实际证据 |
| --- | --- | --- |
| 关键组件消融 | 以完整模型为共同基准；新增无先验对照；保持报告和数值条件不变后逐项修改 | 三种种子的质量、重建、码本利用率与成本表 |
| 复杂度与运行时间 | 10/20/50/100 步；batch=1；5 次预热、30 次计时；同步 CUDA；单独测推理显存 | 延迟均值/中位数/P95、吞吐率、推理峰值显存、采样质量曲线 |
| 失败模式 | 模板旋转/平移、大扰动、解码器先验与潜变量冲突、空报告、错误报告、数值输入误差 | 固定噪声对照图、输入敏感性指标、专家盲评和有证据的局限性分析 |
| 扩展对比 | 无先验条件 LDM、普通物理融合、空间拼接条件 LDM、真实仿真场+LDM；支持外部模型结果统一评价 | 明确同任务、自实现/原文复现、预训练数据、训练预算及仿真成本 |
| 文献覆盖 | 核对五篇 DOI，区分条件生成、信息融合、工业监控与焊接视觉背景 | 相关工作中的适用范围，不把水印任务误列为焊接生成基线 |

## 二、修订后的五组消融

完整配置 `full`：门控物理模板、拟定基准扰动、f=8、K=16384、语义报告+电流/电压/焊速。默认图像 256×256，潜变量 4 通道。

| 组 | 配置 | 控制变量与解释 |
| --- | --- | --- |
| 门控 | `no_prior` / `plain` / `full` | 相同文本/数值与 f/K；无先验分支完全旁路；后两者同模板与投影参数量，只改门控 |
| 扰动 | `perturb_none` / `perturb_small` / `full` / `perturb_large` | 其余完整模型设置一致；测试模板统一固定 |
| 压缩 | `f4` / `full` / `f16` | 潜网格为 64×64 / 32×32 / 16×16；各自重训匹配的第一阶段 |
| 码本 | `k8192` / `full` / `k32768` | f=8；各自重训码本与匹配的第二阶段 |
| 文本/数值 | `condition_none` / `condition_text` / `condition_numeric` / `full` | 相同物理模板分布、同一个冻结的 full 第一阶段；仅改变条件支路 |

门控为 `h + M*P(T,M)`，普通融合为 `h + P(T,M)`，无先验为 `h`。关闭先验后，解码器和扩散模型均不读取模板特征。未使用的投影参数保留在模型结构中但被冻结并旁路，所以报告“总参数”和“可训练参数”时应区分。plain/full 参数量相同。该实现仍是确定性的掩码残差门控，尚未定义为可学习 sigmoid 门控；论文若使用另一门控公式，必须以论文公式进一步对齐，不能直接替换术语。

扰动仍为以下联合范围；“基准”是待验证设定，不是已证明最优：

| 档位 | 旋转 | 共同缩放 | 各轴中心平移 |
| --- | --- | --- | --- |
| none | 0° | 1 | 0 |
| small | ±15° | 0.95–1.05 | ±2.5% 边长 |
| full | ±45° | 0.90–1.10 | ±5% 边长 |
| large | ±75° | 0.80–1.20 | ±10% 边长 |

这组实验评价联合增强范围，不能单独归因于角度、缩放或平移中的某一个因素。第一阶段与第二阶段均使用同档训练扰动；验证/测试统一无扰动，只有失败模式实验主动改变测试条件。

五组去重后为 **13 个设置，每个种子 10 次第一阶段+13 次第二阶段，三个种子共69个训练任务**。训练预算保持第一阶段50,000步、第二阶段100,000步，有效batch=8，固定预算 last.pt 用于正式比较。

## 三、报告条件的补充实验

数值字段仍为电流 A、电压 V、焊速 mm/s。`caption` 使用已去除显式三项参数数值的语义文本，`caption_raw` 保留原始报告。条件贡献四组始终使用相同样本集合，归一化统计只来自训练集。

附加 `report` 组：`condition_numeric` / `full` / `report_raw`。

- `condition_numeric` → `full`：固定数值分支后，语义报告的增量贡献。
- `full` → `report_raw`：完整参数文本相对去数值文本的效果；数值输入在文本中出现后存在冗余，不能解释成新的独立信息来源。
- `condition_none` → `condition_text`：在物理模板已经存在时，纯报告语义分支的贡献。

去数值规则只自动识别带单位的常见写法；仍需审核无单位数字和文件名内容。报告条件应是实际部署时可获得的信息。若报告描述的缺陷标签来源于测试图像人工判读，应明确任务定义，不能把这类标签当作在线可用输入。

## 四、AutoDL 命令

环境和主数据清单格式沿用 `docs/heat_source_ablation.md`。先同步更新后的代码，准备 train/valid/test、真实报告与三项数值标注、512维的本地 CLIP。新主实验都保留完整条件，因此不能再用无标注数据运行前四组完整模型消融。

```bash
cd /root/autodl-tmp/TIGLDM6v3
conda activate heatabl
python scripts/heat_source_revision.py configs

# 五组消融，可逐组运行；自动复用已完成的相同基准并断点续训。
bash scripts/heat_source_revision_commands.sh fusion
bash scripts/heat_source_revision_commands.sh perturbation
bash scripts/heat_source_revision_commands.sh compression
bash scripts/heat_source_revision_commands.sh codebook
bash scripts/heat_source_revision_commands.sh condition

# 或一次运行五组消融的13个唯一设置。
bash scripts/heat_source_revision_commands.sh core

# 报告文本内容的补充对照。
bash scripts/heat_source_revision_commands.sh report

# 新增条件扩散/物理融合对照，不需要仿真场。
bash scripts/heat_source_revision_commands.sh comparison
```

默认种子42/123/2026，RTX4090上使用BF16。改变设备、种子或做小规模检查可追加参数：

```bash
bash scripts/heat_source_revision_commands.sh fusion --seeds 42 --smoke --precision fp32
```

这是软件检查，不能进入正式主表。旧版输出不迁移到新主表。

路径通过环境变量设置：

```bash
export HEAT_DATA_ROOT=/root/autodl-tmp/datasets/ss304
export HEAT_MANIFESTS=/root/autodl-tmp/TIGLDM6v3/datasets/heat_source_manifests
export HEAT_CLIP_PATH=/root/autodl-tmp/TIGLDM6v3/models/clip/clip-vit-base-patch32
```

训练后统一评价：

```bash
python scripts/evaluate_heat_source_suite.py \
  --runs outputs/heat_source_revision/full \
  --out outputs/heat_source_revision_evaluation \
  --data-root "$HEAT_DATA_ROOT" --manifests "$HEAT_MANIFESTS" \
  --clip-path "$HEAT_CLIP_PATH" --device cuda
```

第一阶段报告重建PSNR/SSIM/LPIPS及码本利用率；第二阶段报告FID/KID。随机生成与任意真实图像没有逐像素配对关系，因此第二阶段不使用配对PSNR作为生成精度。

## 五、复杂度与采样步数

训练完成后对完整模型执行：

```bash
# 默认 seed=42，自动找到full的第一/第二阶段last.pt。
bash scripts/heat_source_revision_commands.sh runtime \
  --steps 10 20 50 100 --warmup 5 --repeats 30 --profile-flops

# 使用相同测试条件和初始噪声测量对应步数下的FID/KID。
bash scripts/heat_source_revision_commands.sh quality --steps 10 20 50 100
```

对其他种子也运行，输出目录按种子隔离：

```bash
for seed in 42 123 2026; do
  bash scripts/heat_source_revision_commands.sh runtime --seed "$seed"
  bash scripts/heat_source_revision_commands.sh quality --seed "$seed"
done
```

默认不能向非空诊断目录再次写入；更换设置时用 `--out` 指定新目录。测试某个其他模型时明确传入 `--checkpoint`、`--first-stage` 与独立输出目录。

输出 `runtime.json` / `runtime.csv` 包含硬件、Python/PyTorch、精度、常驻参数、预热次数、采样步数、延迟均值/中位数/P95、单样本吞吐率、推理阶段 peak allocated / peak reserved 和额外临时显存。显存是 PyTorch 分配器计数，不等于整张卡的所有驱动占用。

计时包含输入传输、报告分词/CLIP、扩散采样、重新量化和解码；排除模型加载、图像磁盘I/O、上游报告获取和数值模拟求解。batch固定为1，CUDA前后同步。可选 profiler FLOPs 仅覆盖其支持的矩阵乘法/卷积等操作，输出字段明确标注 partial，不能把它当作完整精确计算量。

真实部署的时限由应用定义。例如仅当应用确认生成环节允许100ms时，可额外传 `--deadline-ms 100 --out outputs/runtime_budget100ms`。程序比较P95与这个生成环节预算；默认不设置时限，也不自动宣称“实时”。论文应把模型延迟和整条监控链路延迟分开，并判断它适合离线数据增强、近实时辅助还是在线闭环。

## 六、失败模式与局限性

```bash
bash scripts/heat_source_revision_commands.sh failures --limit 32 --failure-steps 50
```

使用相同检查点、样本ID和初始噪声生成以下组：

| 干预 | 具体设置 | 可回答的问题 |
| --- | --- | --- |
| clean | 正确条件 | 同噪声参考 |
| prior_rot90 | 模板旋转90°，扩散和解码均改变 | 对方向失配的敏感性 |
| prior_shift25 | 模板向右平移25%边长，空白补零 | 对位置失配的敏感性，无周期回绕 |
| prior_large | 固定种子的大扰动模板 | 超出基准增强范围时的形态变化 |
| decoder_prior_conflict | 扩散阶段保持正确模板，仅解码模板旋转90° | 固定扩散潜变量后，解码先验失配的影响 |
| empty_report | 输入空字符串 | 报告缺失时的影响 |
| wrong_report | 换成另一个测试焊接过程的不同报告，保留原数值 | 错误报告输入的敏感性 |
| numeric_error10/30 | 电流、电压、焊速各自相对±10%/±30%偏移，固定符号种子 | 数值测量/录入偏差的影响 |

未启用对应支路的模型不会伪造该项结果，会在 skipped 中记录原因。没有可用不同焊接过程报告时，wrong_report也会跳过并说明。变更数值可能超出训练分布，应在论文中标注这些工况；图像变化本身不意味着失败，也可能是模型合理服从改变后的条件。

输出包括每个条件的图像、与clean图的变化量、亮区面积/重心/纵横比、输出裁剪比例、各尺度物理分支与潜在特征的余弦相似度/范数比。亮区阈值0.8为未标定图像代理指标，不能声称是熔池真实边界或热物理误差。负余弦仅为特征诊断，不直接证明物理矛盾。

`blind/` 是随机编号图像，`blind_review.csv` 提供真实报告和数值用于人工评价，`review_key.json` 保存编号与干预对应关系，应由统计人员保管。建议至少两名领域人员在不知道干预名称时判断形态退化与真实工况一致性，按焊接过程报告失败比例与分歧。错误报告诱导的输出偏离真实工况，应同时区分“遵循错误输入”和“模型本身形态崩溃”。

可写入稿件的局限性要点：模板是简化形态先验，不能替代热传导求解；先验方向/位置失配可能影响解码；报告真实性和数值标定决定条件有效性；RGB形态指标不等于温度精度；扩散采样成本需与应用时限比较。实际失败方向与幅度应以运行结果和人工判读为依据。

## 七、扩展对比与真实仿真

`comparison` 组是本代码内的受控替代模型：

- `no_prior`：相同报告/数值条件下的无物理先验LDM。
- `plain`：相同解析物理模板的普通残差融合。
- `concat_condition`：第一阶段保持full，第二阶段将模板直接与带噪潜变量拼接输入UNet，作为空间条件输入的替代结构。
- `full`：门控融合。

concat_condition 不是ControlNet复现。以上对照不能替代已有文献方法的独立复现，只能回答本实现内的结构选择问题。

混合仿真-ML的 `hybrid` 组比较 `perturb_none` 与 `simulation_hybrid`。两者都固定先验、保留完整报告/数值，区别仅为解析模板或真实仿真场；仿真场模型需要自己的第一阶段。

在 metadata.csv 增加可选列 `simulation_field,simulation_provenance`，再用原数据准备脚本重新生成一套清单：

```bash
python scripts/prepare_heat_source.py --revision-configs \
  --data-root datasets/ss304 --metadata datasets/ss304/metadata_with_simulation.csv \
  --out datasets/heat_source_manifests_with_simulation

HEAT_MANIFESTS=datasets/heat_source_manifests_with_simulation \
  bash scripts/heat_source_revision_commands.sh hybrid --runs outputs/heat_source_revision_simulation
```

simulation_field 是数据根目录内的`.npy`二维非负有限数组，可表示经过明确定义的热通量或相对环境的温升；需要与图像视场坐标对齐，不能用目标RGB图像构造。provenance填写求解器/版本、材料与边界条件、单位、仿真工况ID及网格投影方式。代码按最大值归一化后插值为256×256，并以0.1阈值得到区域通道；绝对物理幅值不保留。温度场若带有环境基线，应先依据实际物理定义处理，不能把绝对开尔文背景直接误当成热源区域。

程序只加载和检查真实仿真输出，没有虚构有限元求解器、边界条件或仿真结果。缺少仿真场时该组不可运行。应另外报告仿真预计算时间和存储成本。为了公平，解析模板组与仿真组必须使用相同清单和图像子集；新的清单不能静默复用旧权重。

附加report/concat/simulation后共16个设置，每个种子11次第一阶段+16次第二阶段，三种种子共81个训练任务。`all` 会预检并运行全部16个设置，包含真实仿真场要求：

```bash
bash scripts/heat_source_revision_commands.sh all
```

对于论文已有GAN/扩散及额外公开实现，提供统一的外部结果评价入口。每个测试ID恰好一张256×256 RGB PNG；填入 `docs/reviewer_references/external_method.example.json` 中的真实代码来源、commit、预训练权重、条件输入、预算和清单哈希，执行：

```bash
python scripts/evaluate_heat_source_external.py \
  --samples outputs/external_method/samples \
  --metadata outputs/external_method/method.json \
  --data-root datasets/ss304 --manifests datasets/heat_source_manifests \
  --out outputs/external_method/evaluation --device cuda
```

该入口只评价已有真实生成结果，不训练或冒称复现外部模型。外部方法的训练、参数调优和计时仍需按其实现完成；有大规模预训练的方法应在表中另列预训练条件。

## 八、文献与回复材料

见 `docs/reviewer_references/README.md`、`references.bib` 和核对后的 `crossref_metadata.json`。审稿意见1–5对应的交付状态见 `docs/reviewer_response_map.md`。

本地测试覆盖原流程与修订模块，包括真实小型模型的优化/采样、关闭先验不变性、拼接条件梯度、仿真场必填校验、运行时统计和失败模式干预。测试使用合成图像或随机小型CLIP，结果不能代替正式焊接实验。
