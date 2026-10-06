
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np


# ==================================================
# 1. 卷积神经网络（二分类）
# 对应黑板：输入层（注意输入图像维度） + 决策层（简单二分类原理）
# ==================================================
class BinaryCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # ---------- 卷积特征提取层 ----------
        self.features = nn.Sequential(
            # 输入层：接收维度 [批次, 3通道, 高度, 宽度] = [1, 3, 224, 224]
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 尺寸减半：224 → 112

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 尺寸减半：112 → 56

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)   # 尺寸减半：56 → 28
        )

        # ---------- 二分类决策层 ----------
        self.classifier = nn.Sequential(
            nn.Flatten(),  # 将二维特征图展平为一维向量
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 1),  # 输出1个数值
            nn.Sigmoid()        # 映射到0~1区间，代表属于正类的概率，实现二分类
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ==================================================
# 2. 数据集增强技术
# 对应黑板：数据集增强技术
# ==================================================
data_transform = transforms.Compose([
    transforms.Resize((256, 256)),            # 统一图片尺寸
    transforms.RandomHorizontalFlip(p=0.5),   # 50%概率水平翻转
    transforms.RandomRotation(degrees=15),    # 随机旋转±15度
    transforms.RandomCrop((224, 224)),        # 随机裁剪到网络输入尺寸
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 随机调整亮度、对比度
    transforms.ToTensor(),                    # 转为PyTorch张量格式
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 数值归一化
])


# ==================================================
# 3. 经典深度学习算法
# 对应黑板：CNN、RNN、YOLO
# ==================================================
def test_classic_models():
    print("\n" + "=" * 50)
    print("3. 经典深度学习算法测试")
    print("=" * 50)

    # ① 经典CNN：ResNet18（业界标准卷积网络）
    print("\n① 经典CNN模型 ResNet18：")
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet.eval()
    test_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = resnet(test_input)
    print(f"   输入维度：{test_input.shape}")
    print(f"   输出分类维度：{output.shape}（1000类分类输出）")

    # ② 经典RNN变体：LSTM（序列任务常用模型）
    print("\n② 经典RNN变体 LSTM：")
    lstm = nn.LSTM(input_size=10, hidden_size=20, num_layers=2, batch_first=True)
    seq_input = torch.randn(8, 5, 10)  # [批次大小, 序列长度, 特征维度]
    output, (h_n, c_n) = lstm(seq_input)
    print(f"   输入维度：{seq_input.shape}")
    print(f"   输出维度：{output.shape}")

    # ③ YOLO目标检测（需安装ultralytics库）
    print("\n③ YOLO目标检测（代码调用示例）：")
    print("   from ultralytics import YOLO")
    print("   model = YOLO('yolov8n.pt')    # 加载模型")
    print("   result = model('test.jpg')     # 对图片做目标检测")

# 主程序：一键运行所有测试
if __name__ == "__main__":
    # 1. 测试自定义二分类CNN
    print("=" * 50)
    print("1. 自定义二分类CNN网络测试")
    print("=" * 50)
    # 模拟输入层：构造1张3通道、224×224的图片张量
    input_image = torch.randn(1, 3, 224, 224)
    model = BinaryCNN()
    output = model(input_image)
    print(f"输入图像维度：{input_image.shape}")
    print(f"二分类输出概率：{output.item():.4f}")

    # 2. 测试数据增强流水线
    print("\n" + "=" * 50)
    print("2. 数据集增强技术测试")
    print("=" * 50)
    # 生成一张模拟图片用于测试
    test_img_array = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    test_img = Image.fromarray(test_img_array)
    print(f"原始图片尺寸：{test_img.size}")
    augmented_img = data_transform(test_img)
    print(f"增强后张量维度：{augmented_img.shape}")

    # 3. 测试经典算法
    test_classic_models()
