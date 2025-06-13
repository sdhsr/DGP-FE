

# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, Dataset
# from sklearn.preprocessing import MinMaxScaler
#
# # ===== 1. 加载数据 =====
# data = np.load('dataset/train_data/UniFlow_dataset/TaxiNYCIn_48.npy')  # shape: (2880, 10, 20)
# data = data.astype(np.float32)
#
# # 归一化（每个网格点独立归一化）
# scaler = MinMaxScaler()
# data_reshaped = data.reshape(-1, 1)
# data_scaled = scaler.fit_transform(data_reshaped).reshape(data.shape)
#
# # ===== 2. 构造数据集 =====
# class TrafficDataset(Dataset):
#     def __init__(self, data, input_len=12, pred_len=12):
#         self.X = []
#         self.Y = []
#         for i in range(len(data) - input_len - pred_len + 1):
#             x = data[i:i+input_len]
#             y = data[i+input_len:i+input_len+pred_len]
#             self.X.append(x)
#             self.Y.append(y)
#         self.X = np.stack(self.X)  # (N, 12, 10, 20)
#         self.Y = np.stack(self.Y)  # (N, 12, 10, 20)
#
#     def __len__(self):
#         return len(self.X)
#
#     def __getitem__(self, idx):
#         return torch.tensor(self.X[idx]), torch.tensor(self.Y[idx])
#
# dataset = TrafficDataset(data)
# train_size = int(0.8 * len(dataset))
# train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset)-train_size])
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=32)
#
# # ===== 3. 模型定义（ConvLSTM） =====
# class ConvLSTMBlock(nn.Module):
#     def __init__(self, input_dim, hidden_dim, kernel_size):
#         super().__init__()
#         self.convlstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
#
#     def forward(self, x):
#         # x: (B, T, H, W) -> reshape为 (B, T, H*W)
#         B, T, H, W = x.shape
#         x = x.view(B, T, H * W)
#         out, _ = self.convlstm(x)
#         out = out.view(B, T, H, W)
#         return out
#
# class TrafficPredictor(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.encoder = ConvLSTMBlock(10*20, 200, 3)
#         self.decoder = nn.Sequential(
#             nn.Conv2d(12, 64, kernel_size=3, padding=1),  # (B, 12, 10, 20)
#             nn.ReLU(),
#             nn.Conv2d(64, 12, kernel_size=3, padding=1)   # 12个时间戳的输出
#         )
#
#     def forward(self, x):
#         # x: (B, 12, 10, 20)
#         out = self.encoder(x)
#         out = self.decoder(out)
#         return out  # (B, 12, 10, 20)
#
# model = TrafficPredictor().to('cuda' if torch.cuda.is_available() else 'cpu')
#
# # ===== 4. 训练 =====
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# device = next(model.parameters()).device
#
# for epoch in range(10):
#     model.train()
#     total_loss = 0
#     for X, Y in train_loader:
#         X, Y = X.to(device), Y.to(device)
#         optimizer.zero_grad()
#         pred = model(X)
#         loss = criterion(pred, Y)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
#
# # ===== 5. 预测示例 =====
# model.eval()
# with torch.no_grad():
#     for X, Y in test_loader:
#         X = X.to(device)
#         pred = model(X)  # shape: (B, 12, 10, 20)
#         break
#
# # 反归一化结果
# pred_numpy = pred.cpu().numpy().reshape(-1, 1)
# pred_original = scaler.inverse_transform(pred_numpy).reshape(pred.shape)
#
# print("预测结果形状：", pred_original.shape)  # (batch, 12, 10, 20)