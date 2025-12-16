import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as geo_nn
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from tqdm import tqdm
from models.informer import Informer
from torch_geometric.nn import GCNConv
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from utils.metrics import metric

# 对数据进行归一化处理
scaler = StandardScaler()

# 定义自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.X = x
        self.Y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]
        X_t = self.create_time(x)
        Y_t = self.create_time(y)

        value = x[:, 1:]
        label = y[:, 1:]

        # 强制转换为 float32
        value = np.float32(value)  # 原代码已有
        label = np.float32(label)  # 原代码已有

        # 将时间编码 X_t, Y_t 转换为 float32 张量
        X_t = torch.tensor(X_t, dtype=torch.float32)
        Y_t = torch.tensor(Y_t, dtype=torch.float32)

        return value, label, X_t, Y_t

    def create_time(self, data):
        time = data[:, 0]
        time = pd.to_datetime(time)

        # 获取总秒数（从数据起始时间开始计算）
        start_time = pd.to_datetime(data[0, 0])  # 假设第一行是起始时间
        total_seconds = (time - start_time).total_seconds().values.astype(np.int32)
        max_seconds = (pd.to_datetime(data[-1, 0]) - start_time).total_seconds()
        absolute_time = (total_seconds / max_seconds).astype(np.float32)[:, None]
        second = np.int32(time.second)[:, None]

        # 局部周期编码（周期设为60秒，适应20秒间隔）
        local_seconds = total_seconds % 60  # 每60秒一个周期
        local_sec_sin = np.sin(2 * np.pi * local_seconds / 60).astype(np.float32)[:, None]
        local_sec_cos = np.cos(2 * np.pi * local_seconds / 60).astype(np.float32)[:, None]

        # 局部周期编码（可调整周期，适应不同步长）
        local_seconds = total_seconds % 100  # 每60秒一个周期
        local_sec_sin_120 = np.sin(2 * np.pi * local_seconds / 100).astype(np.float32)[:, None]
        local_sec_cos_120 = np.cos(2 * np.pi * local_seconds / 100).astype(np.float32)[:, None]


        # 合并特征 [全局周期, 局部周期]
        return np.concatenate([absolute_time, local_sec_sin, local_sec_cos,local_sec_sin_120,local_sec_cos_120], axis=-1)

# 定义GCN-Informer模型
class GCNInformer(nn.Module):
    def __init__(self, gcn_in_channels, gcn_hidden_channels, gcn_out_channels, num_gcn_layers,
                 informer_enc_in, informer_dec_in, informer_c_out, informer_out_len):
        super(GCNInformer, self).__init__()

        # GCN部分
        self.gcn = GCN(in_channels=gcn_in_channels, hidden_channels=gcn_hidden_channels, out_channels=gcn_out_channels, num_layers=num_gcn_layers)

        # 全连接层
        self.fc = nn.Linear(3, informer_enc_in)

        # Informer部分
        self.informer = Informer(enc_in=informer_enc_in, dec_in=informer_dec_in, c_out=informer_c_out, out_len=informer_out_len)

    def forward(self, data, xt, dec_y, yt):
        # 确保 xt, yt 为 float32
        xt = xt.to(torch.float32)
        yt = yt.to(torch.float32)

        # GCN 前向传播
        gcn_output = self.gcn(data)
        gcn_output = gcn_output.permute(0, 2, 1)
        gcn_output = self.fc(gcn_output)

        # Informer 前向传播
        informer_output = self.informer(gcn_output, xt, dec_y, yt)
        return informer_output

# 定义GCN模型
class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.convs.append(GCNConv(in_channels, hidden_channels))
            elif i < num_layers - 1:
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
            else:
                self.convs.append(GCNConv(hidden_channels, out_channels))

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_weight=edge_weight)
            x = F.relu(x)
            if i != self.num_layers - 1:
                x = F.dropout(x, p=0.5, training=self.training)
        return x
def load_data(file_path):
    """从CSV文件加载数据"""
    datas = pd.read_csv(file_path)
    # scaler.fit(datas.values[:,1:])
    return datas.values

# 准备用于训练和测试的数据
def prepare_data(data, seq_length, label_length,output_steps, train_size=0.7, val_size=0.15):
    # 计算分割点
    split_point_train = int(len(data) * train_size)
    split_point_val = int(len(data) * (train_size + val_size))

    # 分割数据
    train_data = data[:split_point_train]
    test_data = data[split_point_train-seq_length:split_point_val]
    val_data = data[split_point_val-seq_length:]
    # val_data = data[split_point_train:split_point_val]
    # test_data = data[split_point_val:]

    # 分离时间列
    train_data_time = train_data[:, 0].reshape(-1, 1)  # 时间列
    val_data_time = val_data[:, 0].reshape(-1, 1)  # 时间列
    test_data_time = test_data[:, 0].reshape(-1, 1)  # 时间列

    # 特征列
    train_features = train_data[:, 1:]
    val_features = val_data[:, 1:]
    test_features = test_data[:, 1:]

    # 标准化特征数据
    train_features_scaled = scaler.fit_transform(train_features)
    val_features_scaled = scaler.transform(val_features)
    test_features_scaled = scaler.transform(test_features)

    # 将时间列和标准化后的特征数据拼接
    train_data = np.hstack((train_data_time, train_features_scaled))
    val_data = np.hstack((val_data_time, val_features_scaled))
    test_data = np.hstack((test_data_time, test_features_scaled))

    x_train, y_train = create_data(train_data, seq_length, label_length,output_steps, output_steps)
    x_val, y_val = create_data(val_data, seq_length,label_length, output_steps, output_steps)
    x_test, y_test = create_data(test_data, seq_length,label_length, output_steps, output_steps)

    return x_train, x_val, x_test, y_train, y_val, y_test

def create_data(data,seq_length,label_length,output_steps,step):
    X, y = [], []
    for i in range(0,len(data) - seq_length - output_steps + 1,step):
        seq_data = data[i:i + seq_length]
        X.append(seq_data)
        target_data = data[i + seq_length - label_length:i + seq_length + output_steps]
        y.append(target_data)
    X = np.array(X)
    y = np.array(y)
    return X,y

def plot_results(actual, predicted):
    print(actual.shape)
    actual = actual.T
    predicted = predicted.T
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体，以便支持中文标签
    labels = ['CPU', 'Memory', 'Response Time']  # 每个指标对应的标签

    # 创建和绘制每个指标的图形
    for i in range(len(labels)):
        plt.figure(figsize=(10, 3))  # 为每个指标创建一个新的图形
        plt.plot(actual[i], label='Actual', color='red')  # 绘制实际值
        plt.plot(predicted[i], label='Prediction', color='green')  # 绘制预测值
        plt.xlabel('Time Index')
        plt.ylabel('Value')
        plt.title(labels[i])
        plt.legend()
        plt.show()

# def plot_results(actual, predicted):
#     print(actual.shape)
#     actual = actual.T
#     predicted = predicted.T
#     plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体，以便支持中文标签
#     labels = ['CPU', 'Memory', 'Response Time']  # 每个指标对应的标签
#
#     # 创建和绘制每个指标的图形
#     for i in range(len(labels)):
#         plt.figure(figsize=(10, 3))  # 为每个指标创建一个新的图形
#         plt.plot(actual[i], label='Actual', color='red')  # 绘制实际值
#         plt.plot(predicted[i], label='Prediction', color='green')  # 绘制预测值
#
#         # 设置纵坐标范围
#         if i == 0:  # CPU指标
#             plt.ylim(70, 90)
#         elif i == 2:  # 响应时间指标
#             plt.ylim(1, 1.5)
#
#         plt.xlabel('时间索引')
#         plt.ylabel('数值')
#         plt.title(labels[i])
#         plt.legend()
#         plt.show()

def create_edge_index_and_weights(device):
    # 定义边的连接
    edge_index = torch.tensor([
        [0, 2],
        [2, 0],
        [0, 1],
        [1, 0],
        [1,2],
        [2,1]
    ], dtype=torch.long).t().contiguous().to(device)

    # 为每条边分配权重
    edge_weight = torch.tensor([
        0.56,
        0.56,
        0.66,
        0.66,
        0.52,
        0.52
    ], dtype=torch.float32).to(device)

    return edge_index, edge_weight

import os

def save_predictions(trues, preds, output_steps):
    """
    改进版预测结果保存函数
    :param trues: 真实值数组 (N, 3) [cpu, mem, response_time]
    :param preds: 预测值数组 (N, 3) [cpu, mem, response_time]
    :param output_steps: 预测步长，用于列名后缀
    """
    trues = trues[:500]
    preds = preds[:500]

    # 指标配置文件（新增name字段用于生成列名）
    metrics_config = [
        {'file': 'predict_cpu.csv', 'name': 'cpu', 'true_col': 0, 'pred_col': 0},
        {'file': 'predict_memory.csv', 'name': 'mem', 'true_col': 1, 'pred_col': 1},
        {'file': 'predict_response_time.csv', 'name': 'response', 'true_col': 2, 'pred_col': 2}
    ]

    for config in metrics_config:
        file_name = config['file']
        true_idx = config['true_col']
        pred_idx = config['pred_col']
        metric_name = config['name']

        # 生成带步长的列名
        true_col = f'true_{output_steps}'
        pred_col = f'pre_{metric_name}_{output_steps}'

        # 提取对应列数据
        true_values = trues[:, true_idx]
        pred_values = preds[:, pred_idx]

        # 创建/更新数据框
        if os.path.exists(file_name):
            df = pd.read_csv(file_name)
            # 更新或添加列
            df[true_col] = true_values
            df[pred_col] = pred_values
        else:
            # 新建文件包含两个新列
            df = pd.DataFrame({
                true_col: true_values,
                pred_col: pred_values
            })

        # 统一保存格式
        df.to_csv(file_name, index=False)

def main():
    file_path = 'myData/2024-05-5_train_constant.csv'  # 数据文件路径
    seq_length = 10  # 输入序列长度
    label_length = 10  # 输出步长
    output_steps = 5  # 输出步长
    num_epochs = 100  # 训练轮数
    batch_size = 32  # 批量大小
    # learning_rate = 0.0001  # 学习率
    learning_rate = 0.0001  # 学习率
    train_size = 0.8
    val_size = 0.1

    # 检查GPU是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # 加载并归一化数据
    dataWithTime = load_data(file_path)  # 加载数据
    data = dataWithTime

    # 准备数据
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(data, seq_length,label_length,output_steps, train_size, val_size)

    # 边索引创建权重
    edge_index, edge_weight = create_edge_index_and_weights(device)

    # 创建数据加载器
    train_dataset = CustomDataset(X_train, y_train)
    val_dataset = CustomDataset(X_val, y_val)
    test_dataset = CustomDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 对不同指标使用不同权重
    def custom_loss(output, target):
        weights = torch.tensor([1.0, 1, 1.2]).to(device)  # 给不同特征不同权重
        return (weights * (output - target) ** 2).mean()

    # 构建GCN-Informer模型,in_channels一般指的是节点的特征数，这里即为步长
    model = GCNInformer(gcn_in_channels=seq_length, gcn_hidden_channels=3, gcn_out_channels=seq_length, num_gcn_layers=3, informer_enc_in=3, informer_dec_in=3, informer_c_out=3,
                        informer_out_len=output_steps).to(device)

    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 初始化早停参数
    patience = 10  # 允许验证损失不改善的最大周期数
    min_delta = 0.0001 # 认为是有意义的改进的最小阈值

    best_val_loss = float('inf')  # 用于记录最佳验证损失
    patience_counter = 0  # 记录验证损失未改善的周期数

    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()
        for inputs, labels, xt, yt in tqdm(train_loader, desc=f"Epoch {epoch + 1}", leave=False):
            optimizer.zero_grad()

            inputs = inputs.permute(0, 2, 1).to(device)
            labels = labels.to(device)
            xt = xt.to(device)
            yt = yt.to(device)

            # 创建一个Data对象，包含节点特征x、边索引edge_index和边权重edge_weight
            data = Data(x=inputs, edge_index=edge_index, edge_weight=edge_weight).to(device)

            mask = torch.zeros_like(labels)[:, label_length:].to(device)
            dec_y = torch.cat([labels[:, :label_length], mask], dim=1).to(device)

            # Forward pass
            informer_output = model(data, xt, dec_y, yt)
            loss = custom_loss(informer_output, labels[:, label_length:])

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")

        # 验证模型
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels, xt, yt in val_loader:
                inputs = inputs.permute(0, 2, 1).to(device)
                labels = labels.to(device)
                xt = xt.to(device)
                yt = yt.to(device)

                # 创建一个Data对象，包含节点特征x、边索引edge_index和边权重edge_weight
                data = Data(x=inputs, edge_index=edge_index, edge_weight=edge_weight).to(device)

                mask = torch.zeros_like(labels)[:, label_length:].to(device)
                dec_y = torch.cat([labels[:, :label_length], mask], dim=1)

                # Forward pass
                informer_output = model(data, xt, dec_y, yt)

                # 计算损失
                loss = custom_loss(informer_output, labels[:, label_length:])
                val_loss += loss.item() * inputs.size(0)

        val_loss = val_loss / len(val_loader.dataset)
        print(f"Validation Loss: {val_loss:.4f}")

        # 早停检查
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            print("文件更新-----------------")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss
            }, "D:\\checkPoint\\GcnInformer_master_bak.pth")
            print(f"Saved new best model with validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered. Validation loss has not improved for {patience} epochs.")
            break

    # 测试模型
    # checkpoint = torch.load("D:\checkPoint\GcnInformer_master_bak.pth")
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # print("Loaded best model parameters.")

    with torch.no_grad():
        model.eval()
        preds = []
        trues = []

        for inputs, labels, xt, yt in test_loader:
            inputs = inputs.permute(0, 2, 1).to(device)
            labels = labels.to(device)
            xt = xt.to(device)
            yt = yt.to(device)

            # 创建一个Data对象，包含节点特征x、边索引edge_index和边权重edge_weight
            data = Data(x=inputs, edge_index=edge_index, edge_weight=edge_weight).to(device)

            # 训练Informer模型
            mask = torch.zeros_like(labels)[:, label_length:].to(device)
            dec_y = torch.cat([labels[:, :label_length], mask], dim=1)

            informer_output = model(data, xt, dec_y, yt)

            predicted = np.squeeze(informer_output.cpu().detach().numpy())
            actual = np.squeeze(labels[:, label_length:].cpu().detach().numpy())

            preds.append(predicted)
            trues.append(actual)

        preds = np.array(preds)
        trues = np.array(trues)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print("data形状", preds.shape, trues.shape)
        print(f'MSE: {mse:.4f}')
        print(f'MAE: {mae:.4f}')

        preds = preds.reshape(-1, preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-1])
        preds = scaler.inverse_transform(preds)
        trues = scaler.inverse_transform(trues)

        # 保存预测结果到CSV文件
        save_predictions(trues, preds, output_steps)

        plot_results(trues[:], preds[:])


if __name__ == '__main__':
    main()
