import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# 1. 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, csv_file):
        """
        Args:
            csv_file (string): CSV文件的路径。
        """
        self.data_frame = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        features = self.data_frame.iloc[idx, 0:2].values.astype('float32')
        target = self.data_frame.iloc[idx, 2].astype('float32')
        sample = {'features': torch.from_numpy(features), 'target': torch.tensor(target)}
        return sample

# 2. 实例化数据集
# 确保 custom_data.csv 文件与此脚本在同一目录下，或者提供正确的文件路径
try:
    custom_dataset = CustomDataset(csv_file='custom_data.csv')
except FileNotFoundError:
    print("错误：custom_data.csv 未找到。请确保该文件与脚本位于同一目录，或提供正确的文件路径。")
    exit()


# 3. 创建DataLoader
# DataLoader 用于批量加载数据，并可选择性地进行打乱和并行加载
# batch_size 定义了每个批次中样本的数量
# shuffle=True 表示在每个epoch开始时打乱数据顺序
dataloader = DataLoader(custom_dataset, batch_size=4, shuffle=True)

# 4. 迭代数据加载器并打印一些样本 (可选的演示步骤)
print("通过DataLoader加载的批次数据示例：")
for i_batch, sample_batched in enumerate(dataloader):
    print(f"批次 {i_batch}:")
    print("特征:", sample_batched['features'])
    print("目标:", sample_batched['target'])
    if i_batch == 0: # 只打印第一个批次作为演示
        break

# 5. 定义一个简单的PyTorch模型 (可选的演示步骤)
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = torch.nn.Linear(2, 1) # 2个输入特征，1个输出特征

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

model = SimpleModel()
print("\n简单模型结构:")
print(model)

# 6. 训练循环的简单示例 (可选的演示步骤)
# 注意：这是一个非常基础的训练循环，仅用于演示目的。
# 实际应用中需要定义损失函数、优化器，并进行更完整的训练和评估。
print("\n开始一个简单的训练循环示例 (仅一个epoch，一个批次):")
criterion = torch.nn.BCELoss() # 二元交叉熵损失
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(1): # 训练1个epoch
    for i_batch, sample_batched in enumerate(dataloader):
        features = sample_batched['features']
        targets = sample_batched['target'].unsqueeze(1) # 调整目标形状以匹配模型输出

        # 前向传播
        outputs = model(features)
        loss = criterion(outputs, targets)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch+1}/1], Batch [{i_batch+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
        if i_batch == 0: # 只训练一个批次作为演示
            break 
    if epoch == 0: # 只训练一个epoch作为演示
        break

print("\n自定义数据集和DataLoader演示完成。")
print("您现在可以在 d:\\code\\flu\\test\\energy_test 目录下找到 custom_data.csv 和 custom_dataset_demo.py。")
print("要运行此演示，请确保已安装 pandas 和 torch。如果尚未安装 pandas，请运行：pip install pandas")
