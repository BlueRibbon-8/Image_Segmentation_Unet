""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F
from .unet_parts import *  


class UNetMini(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        """
        初始化 UNetMini 模型，一个轻量级的 U-Net 变体，用于图像分割任务。

        参数:
            n_channels (int): 输入图像的通道数（例如 RGB 图像为 3）
            n_classes (int): 输出类别的数量（例如分割任务中的类别数）
            bilinear (bool): 是否使用双线性插值进行上采样（若为 False，则使用转置卷积）
        """
        super(UNetMini, self).__init__()
        self.n_channels = n_channels  # 保存输入通道数
        self.n_classes = n_classes    # 保存输出类别数
        self.bilinear = bilinear      # 保存上采样方式的选择

        # U-Net 的编码器部分（下采样路径）
        self.inc = DoubleConv(n_channels, 8)  # 输入层：初始双卷积块，通道数从 n_channels 变为 8
        self.down1 = Down(8, 16)              # 下采样：8 -> 16 通道
        self.down2 = Down(16, 32)             # 下采样：16 -> 32 通道
        self.down3 = Down(32, 64)             # 下采样：32 -> 64 通道
        factor = 2 if bilinear else 1         # 根据上采样方式调整通道数缩放因子
        self.down4 = Down(64, 128 // factor)  # 最深层下采样：64 -> 128/factor 通道

        # U-Net 的解码器部分（上采样路径）
        self.up1 = Up(128, 64 // factor, bilinear)  # 上采样：连接 x5 和 x4，通道数调整为 64/factor
        self.up2 = Up(64, 32 // factor, bilinear)   # 上采样：连接上一层和 x3，通道数调整为 32/factor
        self.up3 = Up(32, 16 // factor, bilinear)   # 上采样：连接上一层和 x2，通道数调整为 16/factor
        self.up4 = Up(16, 8, bilinear)              # 上采样：连接上一层和 x1，通道数调整为 8
        self.outc = OutConv(8, n_classes)           # 输出层：将通道数从 8 转换为 n_classes

    def forward(self, x):
        """
        定义模型的前向传播过程。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, n_channels, height, width)

        返回:
            torch.Tensor: 输出张量，形状为 (batch_size, n_classes, height, width)
        """
        # 编码器路径：逐步下采样提取特征
        x1 = self.inc(x)      # 输入层处理，生成初始特征图
        x2 = self.down1(x1)   # 第一次下采样
        x3 = self.down2(x2)   # 第二次下采样
        x4 = self.down3(x3)   # 第三次下采样
        x5 = self.down4(x4)   # 第四次下采样，到达网络底部

        # 解码器路径：逐步上采样并融合跳跃连接的特征
        x = self.up1(x5, x4)  # 上采样并融合 x5 和 x4 的特征
        x = self.up2(x, x3)   # 上采样并融合上一层和 x3 的特征
        x = self.up3(x, x2)   # 上采样并融合上一层和 x2 的特征
        x = self.up4(x, x1)   # 上采样并融合上一层和 x1 的特征

        # 输出层处理
        x = self.outc(x)      # 生成最终分割图
        x = torch.sigmoid(x)  # 应用 sigmoid 激活，输出概率值（适用于二分类或多标签任务）
        return x
