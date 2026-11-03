# 文献核对与使用边界

已根据出版方提交到Crossref的记录核对题名、作者、年份、卷页与DOI，原始字段保存在crossref_metadata.json。元数据核对不等于完整阅读所有论文。

| 文献 | 核对后的出版信息 | 可讨论的关系 |
| --- | --- | --- |
| DiffW | IEEE TMM 28, 837–852, 2026；[DOI](https://doi.org/10.1109/TMM.2025.3632631) | 条件扩散与多编码器融合；任务为图像水印 |
| WFormer | IEEE TETCI 8, 4179–4196, 2024；[DOI](https://doi.org/10.1109/TETCI.2024.3386916) | Transformer软融合；任务为图像水印 |
| CAISFormer | Neurocomputing 603, 128295, 2024；[出版方页面](https://www.sciencedirect.com/science/article/pii/S092523122401066X) | 通道注意力和跨特征交互；任务为图像隐写 |
| Hybrid variable monitoring | Automatica 147, 110670, 2023；[出版方页面](https://www.sciencedirect.com/science/article/pii/S0005109822005349) | 连续/二值变量的工业过程监控背景；其hybrid不是“有限元+图像生成” |
| Seam extraction enhancement | IJAMT 141, 647–660, 2025；[出版方页面](https://link.springer.com/article/10.1007/s00170-025-16672-8) | 多通道融合与焊接机器人视觉；任务为焊缝提取 |

这些工作可用于定位相关方法思想，但其任务、输入输出和评价指标与焊接图像生成不同。是否适合作定量生成基线是需要额外论证的判断，不能仅因审稿人建议引用就把它们列为同任务模型。正文应使用“相关的条件建模/特征融合/工业视觉研究”等准确表述。

更直接相关的扩展方向：

- [ControlNet原论文](https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_Adding_Conditional_Control_to_Text-to-Image_Diffusion_Models_ICCV_2023_paper.html)与[作者实现](https://github.com/lllyasviel/ControlNet)：使用空间条件控制预训练文生图模型。若加入正式对比，需要说明大规模预训练的差异。本代码的concat_condition不等于ControlNet。
- [Physics-Informed Diffusion Models原论文](https://arxiv.org/abs/2403.14404)与[作者实现](https://github.com/jhbastek/PhysicsInformedDiffusionModels)：将物理约束纳入扩散训练目标。要迁移到焊接任务，需要定义可计算的控制方程、变量和边界条件；仅有未标定RGB图像与椭圆模板不足以声称复现该方法。

五篇审稿人指定文献的BibTeX由所核对元数据生成，见references.bib。
