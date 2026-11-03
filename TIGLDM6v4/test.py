import torch
from ldm.util import instantiate_from_config
import yaml

# 加载配置
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 实例化模型
model = instantiate_from_config(config.model)

# 检查CLIP维度
clip = model.cond_stage_model
print(f"CLIP embed_dim: {clip.embed_dim}")

# 检查UNet的context_dim
unet = model.model.diffusion_model
print(f"UNet context_dim: {unet.context_dim}")

# 如果维度不匹配，会报错
assert clip.embed_dim == unet.context_dim, f"维度不匹配: {clip.embed_dim} vs {unet.context_dim}"