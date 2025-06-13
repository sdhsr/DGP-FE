import argparse
import csv
import threading
import torch
import time

# import numpy as np
import matplotlib.pyplot as plt



def main():
    parser = argparse.ArgumentParser(description='Print command line arguments.')
    parser.add_argument('--machine', type=str, help='Machine name')
    parser.add_argument('--used_data', type=str, help='Used data')
    parser.add_argument('--prompt_content', type=str, help='Prompt content')

    args = parser.parse_args()

    print("Arguments passed to the script:")
    if args.machine:
        print(f"Machine: {args.machine}")
    if args.used_data:
        print(f"Used Data: {args.used_data}")
    if args.prompt_content:
        print(f"Prompt Content: {args.prompt_content}")


if __name__ == "__main__":
    main()





















# # 加载数据
# pred = np.load("D:\Python\FEDformer-master\\results\\taxibj_12_12_FEDformer_random_modes64_custom_ftM_sl12_ll12_pl12_dm512_nh8_el4_dl4_df2048_fc3_ebtimeF_dtTrue_'Exp'_2\pred.npy")
# true = np.load("D:\Python\FEDformer-master\\results\\taxibj_12_12_FEDformer_random_modes64_custom_ftM_sl12_ll12_pl12_dm512_nh8_el4_dl4_df2048_fc3_ebtimeF_dtTrue_'Exp'_2\\true.npy")
# # 如果需要，也可以加载 metrics，但这里主要关注 pred 和 true
# # metrics = np.load('metrics.npy')
#
# # 假设 pred 和 true 的形状是 (928, 12, 1024)
# # 选择第一个样本的第一个特征进行绘制
# sample_index = 333  # 样本索引
# feature_index = 3  # 特征索引（在最后一个维度上）
# time_step_index_start = 0  # 如果想绘制完整的时间步，可以从0开始
# # 由于第二个维度是12，假设我们绘制所有时间步
#
# # 提取第一个样本的第一个特征在所有时间步上的值
# pred_sample = pred[sample_index, :, feature_index]
# true_sample = true[sample_index, :, feature_index]
#
# # 创建时间序列（假设时间步是连续的整数）
# time_steps = np.arange(pred_sample.shape[0])
#
# # 绘制图形
# plt.figure(figsize=(10, 5))
# plt.plot(time_steps, true_sample, label='True Value', color='blue')
# plt.plot(time_steps, pred_sample, label='Predicted Value', color='red', linestyle='--')
# plt.xlabel('Time Step')
# plt.ylabel('Value')
# plt.title(f'True vs Predicted Values for Sample {sample_index}, Feature {feature_index}')
# plt.legend()
# plt.grid(True)
# plt.show()
#



















# 1. 加载原始数据
# data = np.load('/home/sdh/sdh/Python/UniFlow-main/dataset/train_data/UniFlow_dataset/TaxiBJ13_48.npy')  # 假设原始文件名为 input.npy

# 2. 重塑数据：将 (4848, 32, 32) 展平为 (4848, 1024)
# 使用 reshape(-1, 1024) 确保自动计算第一维大小（4848）
# reshaped_data = data.reshape(-1, 21099)

# print(data.shape)

# # # 3. 保存为 CSV 文件
# np.savetxt('D:\\TaxiBJ13_48_ts.csv', data, delimiter=',', fmt='%s')  # fmt 确保兼容所有数据类型
#

# import numpy as np
#
# # 假设你的 npz 文件名为 'data.npz'
# file_path = 'D:\\Python\\iTransformer-main\\dataset\\PEMS\\PEMS03.npz'
#
# # 使用 numpy 的 load 函数加载 npz 文件
# data = np.load(file_path)
#
# # 遍历文件中的每个数组，并打印其名称和形状
# for key in data.files:
#     array = data[key]
#     print(f"Array name: {key}, Shape: {array.shape}")




# # 假设你的 npz 文件名为 'data.npz'
# file_path = 'D:\\Python\\iTransformer-main\\dataset\\PEMS\\PEMS03.npz'
#
# # 加载 npz 文件
# data = np.load(file_path)
#
# # 假设 npz 文件中只有一个数组或者你知道数组的名称
# # 如果知道数组名称，比如 'arr_0'，可以直接使用 data['arr_0']
# # 否则，你可以列出所有键来选择正确的数组
# array_name = list(data.keys())[0]  # 这里假设只有一个数组或选择第一个
# array = data[array_name]
#
# # 检查数组形状是否为 (28224, 883, 1)
# if array.shape == (26208, 358, 1):
#     # 去掉最后一个维度，将其转换为二维数组 (28224, 883)
#     array_2d = array.reshape(26208, 358)
#
#     # 将二维数组转换为 Pandas DataFrame
#     df = pd.DataFrame(array_2d)
#
#     # 将 DataFrame 保存为 CSV 文件
#     csv_file_path = 'D:\\Pems03.csv'
#     df.to_csv(csv_file_path, index=False)
#
#     print(f"Data has been successfully saved to {csv_file_path}")
# else:
#     print("The array shape does not match (28224, 883, 1). Please check the data.")







# # 读取 CSV 文件
# file_path = 'D:\\Python\\FEDformer-master\\dataset\\traffic\\GraphSH.csv'  # 替换为你的 CSV 文件路径
# data = pd.read_csv(file_path)
#
# # 打印数据的形状
# print(f"CSV 文件的形状为: {data.shape}")





# # 读取原始CSV文件（假设没有标题行）
# df = pd.read_csv('D:\\GraphSH.csv', header=None)
#
# # 添加行号列作为第一列
# df.insert(0, 'new_col', range(len(df)))
#
# # 生成标题行（21100列）
# header = ['date'] + [str(i) for i in range(21098)] + ['OT']
#
# # 将标题行设置为DataFrame的列名
# df.columns = header
#
# # 保存为新的CSV文件（包含标题行）
# df.to_csv('GraphSH1.csv', index=False)


# # 假设你的 CSV 文件名为 'example.csv'
# file_name = 'D:\\GraphSH1.csv'
#
# # 打开 CSV 文件
# with open(file_name, mode='r', encoding='utf-8') as file:
#     csv_reader = csv.reader(file)
#
#     # 读取第一行
#     first_row = next(csv_reader)
#
#     # 打印第一行
#     print(first_row)





# # 模拟一个计算密集型任务，例如模型推理或数据预处理
# def compute_task(task_id):
#     print(f"Task {task_id} started")
#     # 模拟计算时间
#     time.sleep(2)  # 在实际应用中，这里可能是模型推理或数据预处理
#     # 创建一个随机张量作为计算结果（仅作为示例）
#     result = torch.randn(3, 3)
#     print(f"Task {task_id} completed with result: {result}")
#
#
# # 创建两个线程，每个线程执行一个计算任务
# thread1 = threading.Thread(target=compute_task, args=(1,))
# thread2 = threading.Thread(target=compute_task, args=(2,))
#
# # 启动线程
# thread1.start()
# thread2.start()
#
# # 等待所有线程完成
# thread1.join()
# thread2.join()
#
# print("All tasks completed")










# # 生成一些模拟数据
# np.random.seed(0)
# X = np.random.rand(2000, 2)  # 输入特征，2000个样本，每个样本2个特征
# y = (X[:, 0] + X[:, 1] > 1).astype(int)  # 目标变量，简单的线性决策边界
#
# # 将数据转换为PyTorch张量
# X_tensor = torch.tensor(X, dtype=torch.float32)
# y_tensor = torch.tensor(y, dtype=torch.long)
#
# # 创建数据集和数据加载器
# dataset = TensorDataset(X_tensor, y_tensor)
# train_size = int(0.8 * len(dataset))
# val_size = len(dataset) - train_size
# train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
#
# train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)
#
# # 定义第一个类：特征提取模块
# class FeatureExtractor(nn.Module):
#     def __init__(self):
#         super(FeatureExtractor, self).__init__()
#         self.linear = nn.Linear(2, 10)
#
#     def forward(self, x):
#         return torch.relu(self.linear(x))
#
# # 定义第二个类：分类器模块，引用特征提取模块
# class Classifier(nn.Module):
#     def __init__(self):
#         super(Classifier, self).__init__()
#         self.feature_extractor = FeatureExtractor()
#         self.classifier = nn.Linear(10, 2)
#
#     def forward(self, x):
#         features = self.feature_extractor(x)
#         return self.classifier(features)
#
# # 实例化模型
# model = Classifier()
#
# # 定义损失函数和优化器
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01)
#
# # 早停参数
# patience = 6  # 容忍多少个epoch验证损失没有改善
# interval=5
# best_val_loss = float('inf')  # 初始化为无穷大
# epochs_no_improve = 0
#
# # 训练模型
# num_epochs = 500
# for epoch in range(num_epochs):
#     model.train()
#
#     running_loss = 0.0
#     for inputs, labels in train_loader:
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#
#     epoch_loss = running_loss / len(train_loader)
#     print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')
#
#
#     if epoch%interval==0:
#         # 验证模型
#         model.eval()
#         val_loss = 0.0
#         with torch.no_grad():
#             for inputs, labels in val_loader:
#                 outputs = model(inputs)
#                 loss = criterion(outputs, labels)
#                 val_loss += loss.item()
#
#         val_loss /= len(val_loader)
#         print(f'Validation Loss: {val_loss:.4f}')
#
#         # 早停逻辑
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             epochs_no_improve = 0
#             print(f'Best validation loss updated: {best_val_loss:.4f}')
#         else:
#             epochs_no_improve += 1
#             print(f'No improvement in {epochs_no_improve} epochs')
#             if epochs_no_improve >= patience:
#                 print(f'Early stopping at epoch {epoch + 1}')
#                 break
#
#
#
#
# print(f'Best validation loss achieved: {best_val_loss:.4f}')