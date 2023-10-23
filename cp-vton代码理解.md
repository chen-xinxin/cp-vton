`data_download.py`：下载数据并解压、放到对应的文件夹（运行报错，手动下载）

`cp_dataset.py`：定义了一个数据集类`CPDataset`（处理CP-VTON数据集）和一个数据加载器类`CPDataLoader`（加载CP-VTON数据集）

`convert_data.m`：处理原始数据的（不用管，下载的数据已经处理好了）

`network.py`：

##### GMM：

- `weights_init_normal(m)`：用正态分布初始化方法初始化网络层m
- `weights_init_xavier(m)`：用Xavier初始化方法初始化网络层m
- `weights_init_kaiming(m)`：用Kaiming初始化方法初始化网络层m
- `init_weights(net, init_type='normal')`：根据初始化方法类型`init_type`初始化网络模型`net`
- `class FeatureExtraction(nn.Module)`：定义了一个名为`FeatureExtraction`的神经网络模型类。该模型用于提取特征，并由多个卷积层和归一化层组成。
- `class FeatureL2Norm(torch.nn.Module)`：对输入的特征进行L2归一化（张量的每个元素除以张量的L2范数）
- `class FeatureCorrelation(nn.Module)`：计算两个特征张量之间的相关性（组合两个特征为一个张量），相关性张量可以用于匹配关键点、估算光流
- `class FeatureRegression(nn.Module)`：回归出TPS变形参数
- `class AffineGridGen(nn.Module)`：生成仿射变换的网格
- `class TpsGridGen(nn.Module)`：TPS变形模块

##### TOM：

`class UnetGenerator(nn.Module)`：它通过堆叠`UnetSkipConnectionBlock`模块（嵌套结构）来构建U-Net生成器

`class UnetSkipConnectionBlock(nn.Module)`：实现U-Net中的跳跃连接块