# ===== 统一导入区 =====
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np


# ==================================================
# 1. 卷积神经网络（二分类）
# ==================================================
class BinaryCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 卷积特征提取层
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # 二分类决策层
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ==================================================
# 2. 数据集增强流水线
# ==================================================
data_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.RandomCrop((224, 224)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


# ==================================================
# 3. 经典深度学习算法测试
# ==================================================
def test_classic_models():
    print("\n" + "=" * 50)
    print("3. 经典深度学习算法测试")
    print("=" * 50)

    # 经典CNN：ResNet18
    print("\n① 经典CNN模型 ResNet18：")
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet.eval()
    test_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = resnet(test_input)
    print(f"   输入维度：{test_input.shape}")
    print(f"   输出分类维度：{output.shape}（1000类分类输出）")

    # 经典RNN变体：LSTM
    print("\n② 经典RNN变体 LSTM：")
    lstm = nn.LSTM(input_size=10, hidden_size=20, num_layers=2, batch_first=True)
    seq_input = torch.randn(8, 5, 10)
    output, (h_n, c_n) = lstm(seq_input)
    print(f"   输入维度：{seq_input.shape}")
    print(f"   输出维度：{output.shape}")

    # YOLO调用示例
    print("\n③ YOLO目标检测（代码调用示例）：")
    print("   from ultralytics import YOLO")
    print("   model = YOLO('yolov8n.pt')")
    print("   result = model('你的图片.jpg')")



# 主程序：使用你自己的图片运行

if __name__ == "__main__":
    img_path = "F:/seele/8D955976215C036F5273844FFB2C87AD.png"


    # 1. 自定义二分类CNN 真实图片测试
    print("=" * 50)
    print("1. 自定义二分类CNN网络测试")
    print("=" * 50)
    # 加载图片并转RGB三通道
    img = Image.open(img_path).convert("RGB")
    # 预处理成网络要求的格式
    input_tensor = data_transform(img).unsqueeze(0)

    model = BinaryCNN()
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)

    print(f"原始图片尺寸：{img.size}")
    print(f"网络输入维度：{input_tensor.shape}")
    print(f"二分类输出概率：{output.item():.4f}")

    # 2. 数据增强 真实图片测试
    print("\n" + "=" * 50)
    print("2. 数据集增强技术测试")
    print("=" * 50)
    print(f"原始图片尺寸：{img.size}")
    # 执行数据增强
    augmented_tensor = data_transform(img)
    print(f"增强后张量维度：{augmented_tensor.shape}")

    # 3. 经典算法测试
    test_classic_models()