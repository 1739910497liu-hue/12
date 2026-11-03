# 第二阶段 BF16 梯度检查点修复

适用于当前服务器的 PyTorch 2.9 / CUDA BF16 环境。

第二阶段的 `BasicTransformerBlock` 默认启用梯度检查点。旧的 `CheckpointFunction.backward` 在重计算时只开启梯度，没有恢复前向的 autocast 状态，因此可能将 BF16 激活与 FP32 参数直接送入 LayerNorm 或 Linear，出现类型不一致。`HeatDiffusion` 中的 `use_checkpoint=False` 仅传递到部分 UNet 模块，不会关闭这些 Transformer 检查点。

本次运行代码只修改 `ldm/modules/diffusionmodules/util.py`：保留四参数调用接口，内部使用 `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`，交由 PyTorch 保存和恢复 autocast 与随机数状态；未启用检查点或处于 `no_grad` 时直接调用前向。模型参数和权重文件格式不变，第一阶段无需重训。

上传 `stage2_bf16_fix_20260923.zip` 至服务器项目根目录后：

```bash
conda activate tigldm-blackwell
cd /root/autodl-tmp/TIGLDM6v3-3
python -m zipfile -e stage2_bf16_fix_20260923.zip .
python -m unittest discover -s tests -p "test_heat_source_checkpoint.py" -v
```

测试不会下载权重或读取数据集。在支持 BF16 的 CUDA 设备上会自动执行 Transformer GPU 反向传播测试；本地无 CUDA 时该项跳过。

然后重新执行与第一阶段对应的第二阶段命令。

门控实验：

```bash
bash scripts/heat_source_gating_commands.sh stage2
```

相加/拼接融合实验：

```bash
bash scripts/heat_source_fusion_commands.sh stage2
```

两个脚本分别使用各自输出目录；已有 `stage2/last.pt` 时自动续训，否则从第二阶段起点开始。各组仍要求对应的第一阶段已完成 50,000 步。保留 BF16、数据清单和原实验配置。

本地验证：CPU BF16 回归 7 项通过，CUDA BF16 测试因无 GPU 跳过；另通过小型两阶段 FP32 训练、恢复及生成测试。旧实现的 BF16 Linear 前向可通过、反向失败，新实现通过前后对照。服务器 GPU 完整训练结果需在部署后确认。

PyTorch 原生实现参考：[2.9 checkpoint 文档](https://docs.pytorch.org/docs/2.9/checkpoint.html)、[2.9 实现源码](https://github.com/pytorch/pytorch/blob/v2.9.0/torch/utils/checkpoint.py)。
