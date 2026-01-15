本系统基于Unity3D引擎与NicheFormer空间转录组基础模型，构建了一个数字孪生平台，可以在虚拟空间中重建细胞间的空间位置关系，捕捉静态图表无法呈现的局部细节与空间连续性。

主要功能 (Features)
本系统集成了五大核心分析模块：
交互式 3D 视图: 在三维空间中实时渲染数万个细胞，直观展示组织结构。
基因表达与插补 (Gene Imputation): 实时查询特定基因表达水平，并利用 AI 模型对稀疏数据进行插补，还原真实表达模式。
自动细胞类型注释 (Auto Annotation): 基于 Nicheformer 模型自动预测细胞类型，并生成可视化图例。
组织区域语义分割 (Tissue Segmentation): 自动识别并分割不同的组织功能区域（如皮层、髓质等），支持单区域独立展示。
零样本聚类 (Zero-Shot Clustering): 用户自定义聚类数目，无需重新训练即可即时发现潜在的细胞亚群。


快速开始 (Getting Started)
1. 环境准备
后端 (Python): 确保已安装 Python 环境及必要的依赖库。

Bash

cd backend
pip install -r requirements.txt
# 确保已下载 Nicheformer 权重文件 (nicheformer_weights.pth) 并放置在正确路径
前端 (Unity):

安装 Unity Hub 和 Unity Editor (建议 2020.3 或更高版本)。

克隆本项目到本地。

2. 运行步骤
启动后端服务:

Bash

python server.py
# 默认运行在 http://127.0.0.1:8000
启动 Unity 客户端:

在 Unity Editor 中打开项目。

打开主场景 (MainScene)。

点击顶部 Play 按钮。

📖 使用指南 (User Guide)
1. 基因表达分析
在输入框输入基因名称（如 TP53）。

点击 Search 更新视图。

点击 Imputation 按钮，系统将通过 AI 补全缺失信号，增强可视化效果。

点击 Save 保存当前分析结果。

2. 细胞类型注释
点击 Cell Type Annotation 按钮。

右侧将自动生成细胞类型图例。

点击图例中的特定类型可高亮显示该类细胞。

3. 区域组织分割
点击 Tissue Segmentation 按钮。

系统将根据空间特征将组织划分为不同区域。

使用下拉菜单或图例选择特定区域（如 Region 1）进行单独查看。

提示：点击 "Return" 按钮可重置显示所有细胞。

4. 零样本聚类
在聚类输入框中输入期望的聚类数目（例如 10）。

点击 Zero-Shot Clustering。

系统将实时计算并在 3D 空间中更新聚类颜色分布。
