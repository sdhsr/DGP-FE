# import argparse
# import csv
# import threading
# import torch
# import time
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import json
# import os


import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d  # 需要用到高斯滤波来做平滑

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# 1. 准备数据 (根据你图片中的数据)
data = np.array([
    [16.593, 16.087, 16.467, 16.281, 16.182],
    [16.096, 16.225, 17.287, 16.770, 17.052],
    [16.525, 16.443, 16.349, 16.292, 17.183],
    [16.193, 16.192, 16.630, 16.289, 16.465],
    [16.991, 16.327, 16.582, 16.549, 16.569]
])

a_labels = [0.1, 0.2, 0.4, 0.6, 0.8]  # Y轴标签
b_labels = [0.1, 0.2, 0.4, 0.6, 0.8]  # X轴标签

df = pd.DataFrame(data, index=a_labels, columns=b_labels)

# ==========================================
# 方案一：热力图 (最推荐)
# ==========================================
plt.figure(figsize=(8, 6))
# cmap='viridis' (蓝绿黄) 或 'coolwarm' (蓝红) 或 'RdBu_r' (红蓝翻转)
# 如果数值越小越好，建议用 'viridis_r' 或自定义使得深色代表小数值
sns.heatmap(df, annot=True, fmt=".3f", cmap='viridis', cbar_kws={'label': 'Model Performance'})
plt.title('Performance Heatmap (a vs b)')
plt.ylabel('Variable a (cut)')
plt.xlabel('Variable b (balance)')
plt.show()

# ==========================================
# 方案二：3D 曲面图
# ==========================================
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

X, Y = np.meshgrid(range(len(b_labels)), range(len(a_labels)))
# 注意：Surface plot 需要数值矩阵
surf = ax.plot_surface(X, Y, data, cmap='viridis', edgecolor='none')

ax.set_xticks(range(len(b_labels)))
ax.set_xticklabels(b_labels)
ax.set_yticks(range(len(a_labels)))
ax.set_yticklabels(a_labels)

ax.set_xlabel('Variable b (balance)')
ax.set_ylabel('Variable a (cut)')
ax.set_zlabel('Performance')
ax.set_title('3D Performance Surface')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

# =
# import numpy as np
# import pandas as pd
#
# # 假设你有一个NumPy邻接矩阵
# # 这里创建一个示例矩阵（5x5的随机邻接矩阵）
# n = 5
# adjacency_matrix = np.load("D:\Python\数据集\Chengdu\chengdu_raw_adj.npy")
#
#
# # 方法2：使用pandas（更灵活，可以添加行列标签）
# df = pd.DataFrame(adjacency_matrix)
# # 可以添加行列索引标签（可选）
#
# df.to_csv('D:\\Python\\数据集\\Chengdu\\adjacency_matrix_pandas.csv', index=False, header=False)
#
# print("邻接矩阵已成功保存为CSV文件")



# import numpy as np
# import matplotlib.pyplot as plt
#
# # ==========================================
# # 1. 配置参数 (根据您提供的信息)
# # ==========================================
# MEAN = 230.2887
# STD = 145.570291
# RMSE_OURS = 29.6724  # 来源于您提供的表格
# MAE_OURS = 18.2722
#
# # 设置绘图风格
# plt.style.use('default')
# plt.rcParams['figure.figsize'] = (10, 5)
# plt.rcParams['font.size'] = 14
# plt.rcParams['font.family'] = 'sans-serif'
#
#
# # ==========================================
# # 2. 数据加载 / 模拟生成
# # ==========================================
#
# def load_or_simulate_data():
#     """
#     如果您有真实数据，请修改此函数加载您的 .npy 文件。
#     当前逻辑为模拟生成，仅供演示绘图代码。
#     """
#
#     # -----------------------------------------------------------
#     # [选项 A]：如果您有真实预测结果 (推荐)
#     # -----------------------------------------------------------
#     true_data = np.load('../dataset/train_data/UniFlow_dataset/GraphPems08_flow_288.npy') # 形状 (17856, 170, 1)
#     # pred_data = np.load('path_to_your_prediction.npy')   # 形状 (17856, 170, 1)
#     #
#     # # 选择一个节点 (例如第 0 个节点) 和一个时间段
#     node_idx = 100
#     y_true = true_data[:, node_idx, 0]
#     # y_pred = pred_data[:, node_idx, 0]
#     # -----------------------------------------------------------
#
#     # -----------------------------------------------------------
#     # [选项 B]：模拟数据 (根据您的统计数据生成)
#     # -----------------------------------------------------------
#     print("正在生成模拟数据以演示绘图...")
#     time_steps = 17856
#     x = np.linspace(0, 100 * np.pi, time_steps)
#
#     # # 模拟真实交通流：正弦波(周期性) + 均值 + 随机波动
#     # # 调整振幅以匹配您提供的 std (145.57)
#     # base_signal = np.sin(x) * (STD * 1.2) + MEAN
#     # # 确保没有负数 (交通流通常 >= 0)
#     # y_true = np.maximum(base_signal + np.random.normal(0, 20, time_steps), 0)
#
#     # 模拟预测值：真实值 + 误差 (基于您的 RMSE 29.67)
#     # 预测通常会稍微平滑一些，并带有滞后或噪声
#     noise = np.random.normal(0, RMSE_OURS, time_steps)
#     y_pred = y_true + noise
#
#     return y_true, y_pred
#
#
# # 加载数据
# y_true_all, y_pred_all = load_or_simulate_data()
#
# # ==========================================
# # 3. 绘图逻辑 (复刻目标图片风格)
# # ==========================================
#
# # 截取一个可视化窗口 (例如 250 个时间步，约 1 天的数据量，假设5分钟一个点)
# # 您可以修改 start_idx 来查看不同时间段
# start_idx = 1000
# window_size = 288  # 一天的切片 (12 * 24)
# end_idx = start_idx + window_size
#
# # 提取切片
# y_true_slice = y_true_all[start_idx:end_idx]
# y_pred_slice = y_pred_all[start_idx:end_idx]
# x_axis = np.arange(len(y_true_slice))
#
# # 开始绘图
# fig, ax = plt.subplots()
#
# # 画线
# # 蓝色线：Ground Truth (Actual)
# ax.plot(x_axis, y_true_slice, label='Actual', color='#1f77b4', linewidth=1.5)
# # 橙色线：Forecast (Prediction)
# ax.plot(x_axis, y_pred_slice, label='Forecast', color='#ff7f0e', linewidth=1.5)
#
# # 添加垂直分割线 (可选，模仿原图左侧的分割感)
# # ax.axvline(x=50, color='gray', linestyle='-', linewidth=1, alpha=0.7)
#
# # 设置标题和标签
# ax.set_title(f"Forecast vs. Actual (PEMS08)\nRMSE: {RMSE_OURS:.4f}, MAE: {MAE_OURS:.4f}", fontsize=16)
# ax.set_xlabel("Time Steps", fontsize=16)
# ax.set_ylabel("Traffic Flow", fontsize=16)
#
# # 设置图例 (去掉边框，放在合适位置)
# ax.legend(frameon=False, fontsize=14, loc='upper right')
#
# # 设置坐标轴刻度字体大小
# ax.tick_params(axis='both', which='major', labelsize=14)
#
# # 调整布局
# plt.tight_layout()
#
# # 保存或显示
# # plt.savefig('forecast_vs_actual.png', dpi=300)
# plt.show()







# ##D:\Python\UniFlow-main\dataset\train_data\UniFlow_dataset\TaxiBJ13_48.npy
# # 加载 .npy 文件   (2880, 10, 20)       (12672, 228, 1) (2688, 21099, 1)  (12671, 1026, 1)
# npy_array = np.load('../dataset/train_data/UniFlow_dataset/GraphPemsBay_288.npy')  # 替换为你的 .npy 文件路径
#
# print(npy_array.shape)GraphPemsBay_288.npy
#
#

#
# import numpy as np
# import os
#
#
# def add_node_noise_tnc(file_path, ratios=[0.03, 0.07]):
#     """
#     针对形状为 (T, N, C) 的数据进行节点级加噪
#     T: 时间戳数量 (12672)
#     N: 节点个数 (228)
#     C: 特征维度 (1)
#     """
#     # 1. 加载原始数据
#     data = np.load(file_path)
#     num_timesteps, num_nodes, num_features = data.shape
#     print(f"原始数据形状: {data.shape}")
#
#     # 计算全局标准差用于控制噪声强度
#     data_std = np.std(data)
#
#     # 创建保存目录
#     output_dir = "../dataset/train_data/UniFlow_dataset/Pems08_noise"
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#
#     results = {}
#
#     for r in ratios:
#         # 深度拷贝，确保各比例实验互不干扰
#         noisy_data = np.copy(data).astype(np.float32)
#
#         # 2. 确定要加噪的节点数量
#         num_noisy_nodes = int(np.ceil(num_nodes * r))
#
#         # 3. 随机选择节点索引 (从维度 1 中选择)
#         all_node_indices = np.arange(num_nodes)
#         selected_nodes = np.random.choice(all_node_indices, num_noisy_nodes, replace=False)
#
#         # 4. 生成高斯噪声
#         # 噪声形状匹配：(时间步, 选中的节点数, 特征维度)
#         noise_level = 0.8 * data_std
#         noise = np.random.normal(0, noise_level, size=(num_timesteps, num_noisy_nodes, num_features))
#
#         # 5. 注入噪声：作用于第 1 维 (Nodes)
#         noisy_data[:, selected_nodes, :] += noise
#
#         # 6. 保存结果
#         file_name = f"pems08_noise_{int(r * 100)}.npy"
#         save_path = os.path.join(output_dir, file_name)
#         np.save(save_path, noisy_data)
#
#         results[r] = {
#             "path": save_path,
#             "nodes_affected": num_noisy_nodes,
#             "indices": selected_nodes
#         }
#         print(f"完成 {int(r * 100)}% 节点加噪: 随机选中了 {num_noisy_nodes} 个节点，数据形状保持 {noisy_data.shape}")
#
#     return results
#
# # 执行加噪
# results = add_node_noise_tnc("../dataset/train_data/UniFlow_dataset/GraphPems08_flow_288.npy")
#
# # # 使用示例

# #1672
# "python main.py --machine machine  --patch_size=2 --dataset GraphPems08_flow_288_noise_1  --wide=170 --num_nodes=170 --task_id UniFlow --early_stop=10  --used_data 'GridGraphall' --model UniFlow"
# "python main.py --machine machine  --patch_size=1 --dataset GraphPems08_flow_288_noise_1  --wide=170 --num_nodes=170 --task_id UniFlow_A_D_M_F --early_stop=10  --used_data 'GridGraphall' --model UniFlow_A_D_M_F"
#
# "python main.py --machine machine  --patch_size=2 --dataset GraphPems08_flow_288_noise_5  --wide=170 --num_nodes=170 --task_id UniFlow --early_stop=10  --used_data 'GridGraphall' --model UniFlow"
# "python main.py --machine machine  --patch_size=1 --dataset GraphPems08_flow_288_noise_5  --wide=170 --num_nodes=170 --task_id UniFlow_A_D_M_F --early_stop=10  --used_data 'GridGraphall' --model UniFlow_A_D_M_F"
#
# "python main.py --machine machine  --patch_size=2 --dataset GraphPems08_flow_288_noise_10  --wide=170 --num_nodes=170 --task_id UniFlow --early_stop=10  --used_data 'GridGraphall' --model UniFlow"
# "python main.py --machine machine  --patch_size=1 --dataset GraphPems08_flow_288_noise_10  --wide=170 --num_nodes=170 --task_id UniFlow_A_D_M_F --early_stop=10  --used_data 'GridGraphall' --model UniFlow_A_D_M_F"
#
# "python main.py --machine machine  --patch_size=2 --dataset GraphPems08_flow_288_noise_20  --wide=170 --num_nodes=170 --task_id UniFlow --early_stop=10  --used_data 'GridGraphall' --model UniFlow"
# "python main.py --machine machine  --patch_size=1 --dataset GraphPems08_flow_288_noise_20  --wide=170 --num_nodes=170 --task_id UniFlow_A_D_M_F --early_stop=10  --used_data 'GridGraphall' --model UniFlow_A_D_M_F"
#
# "python main.py --machine machine  --patch_size=2 --dataset GraphPems08_flow_288_noise_30  --wide=170 --num_nodes=170 --task_id UniFlow --early_stop=10  --used_data 'GridGraphall' --model UniFlow"
# "python main.py --machine machine  --patch_size=1 --dataset GraphPems08_flow_288_noise_30  --wide=170 --num_nodes=170 --task_id UniFlow_A_D_M_F --early_stop=10  --used_data 'GridGraphall' --model UniFlow_A_D_M_F"

#"python main.py --machine machine --seq_len=128 --his_len=64 --pred_len=64  --patch_size=2 --dataset GraphPems08_speed_288 --wide=170 --num_nodes=170 --task_id UniFlow  --used_data 'GridGraphall' --model UniFlow"
#"python main.py --machine machine --seq_len=72 --his_len=36 --pred_len=36 --patch_size=1 --dataset GraphPems08_flow_288  --wide=170 --num_nodes=170 --task_id UniFlow_A_D_M_F  --used_data 'GridGraphall' --model UniFlow_A_D_M_F"
#"python main.py --machine machine --seq_len=128 --his_len=64 --pred_len=64  --patch_size=1 --dataset GraphPems08_speed_288 --wide=170 --num_nodes=170 --task_id UniFlow_A_D_M_F  --used_data 'GridGraphall' --model UniFlow_A_D_M_F"


# # # 加载.npy文件  原(12672, 6, 38)----》 (12672, 228, 1)                                                 (17856, 170, 1)
# data = np.load("../dataset/train_data/UniFlow_dataset/Pemsd7_288.npy")  # 形状为 PemsD7:(228, 228)  #(12672, 228, 1)
#
# # # # 2. 调整形状为 (17856, 170, 1)
# # data_reshaped = data.reshape(data.shape[0],325, 1)  # 或 data.reshape(17856, 170, 1)
# # #
# # # # 3. 保存为新的 .npy 文件
# # np.save("../dataset/train_data/UniFlow_dataset/GraphPemsBay_288.npy", data_reshaped)
#
# print(data.shape)#       ---->#(52105, 25, 13)



#                   # 生成timestamp
# import csv
#
# # 定义参数
# max_first_col = 6  # 第一列最大值
# max_second_col = 287  # 第二列最大值
# total_rows = 12672  # 总行数
#
# # 打开CSV文件准备写入
# with open('/media/sdh/E盘/download/PeMSD7_Full/timstemp2.csv', mode='w', newline='') as file:
#     writer = csv.writer(file)
#
#     for i in range(total_rows):
#         first_col = (i // (max_second_col + 1)) % (max_first_col + 1)
#         second_col = i % (max_second_col + 1)
#         writer.writerow([first_col, second_col])
#
# print("CSV 文件已生成，共12672行，文件名为 output.csv")




# 读取 CSV 文件（假设文件名为 input.csv，没有列名）  #(12672, 228)
# df = pd.read_csv('/home/sdh/sdh/Python/USTGCN-master/PeMSD7/PeMSD7_speed.csv', header=None)
#
# # 转换为 NumPy 数组
# array = df.to_numpy()
# print(array.shape)

#
# # 保存为 .npy 文件
# np.save('/media/sdh/E盘/download/PeMSD7_Full/timstemp.npy', array)
#
# print("转换完成，已保存为 output.npy")