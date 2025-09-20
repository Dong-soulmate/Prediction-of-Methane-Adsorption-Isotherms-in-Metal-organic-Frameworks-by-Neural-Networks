import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape
import csv

import logging
import datetime

n_parameter = 4
mask_probability = 1.00
num_epochs = 15000
n_fea_cols = 206
batch_size = 8192

now_time = datetime.datetime.now()
timestamp = now_time.strftime("%Y-%m-%d_%H_%M-%S")
logfile = f'nn_train_masked_{num_epochs}_{n_parameter}p_{mask_probability}_{timestamp}_.log'
logging.basicConfig(filename=logfile, level=logging.INFO, filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')

# 参数配置
current_dir = os.path.dirname(os.path.abspath(__file__))

pressures = np.array(
    [
        0.001,
        5.000949999999783,
        10.000899999999682,
        15.000849999999435,
        20.00080000000054,
        25.000750000000036,
        30.000699999998385,
        35.000650000000526,
        40.000600000001235,
        45.00054999999244,
        50.00050000000269,
        55.00045000000194,
        60.00040000000764,
        65.00034999999808,
        70.0003000000032,
        75.00025000000272,
        80.00019999999458,
        85.00014999999743,
        90.00009999998561,
        95.00005000000186,
        99.99999999999812,
    ])

n_pressures = len(pressures)

file_path = os.path.join(current_dir, f'mofdatabase_features_clean_data_{n_parameter}P.csv')
n_output = n_pressures + n_parameter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 读取数据
def read_data(file_path):
    try:
        data = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading the file: {e}")
        logging.info(f"Error reading the file: {e}")
        raise
    return data


data = read_data(file_path)

T = data.loc[:, 'TK'].values

# 特征与目标变量处理
feature_columns = data.columns[1:n_fea_cols]
target_columns = data.columns[n_fea_cols:]

# 处理非数值列
non_numeric_columns = ['functional_groups', 'topology', 'metal type']
# 使用OneHotEncoder处理类别变量
one_hot_encoder = OneHotEncoder(sparse_output=False)
categorical_features = data[non_numeric_columns].values
categorical_encoded = one_hot_encoder.fit_transform(categorical_features)

# 处理缺失值
data_without_zero_nan = data[feature_columns].drop(data.columns[data.isna().all() | (data == 0).all()], axis=1)
data_without_zero_nan = data_without_zero_nan.drop(non_numeric_columns, axis=1)
X = np.concatenate((categorical_encoded, data_without_zero_nan.values), axis=1)
y = data[target_columns].values
# non_numerical_rows = data[data['functional_groups'].apply(lambda x: isinstance(x,str))]

min_target = y.min(axis=0)
max_target = y.max(axis=0)
y_normalized = (y - min_target) / (max_target - min_target)

# 数据集分割
X_train, X_test, y_train, y_test = train_test_split(X, y_normalized, test_size=0.2, random_state=0)

# 标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# 定义自定义Dataset类
class MOFDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32)
        )

        if self.transform:
            sample = self.transform(sample)

        return sample


# 创建DataLoader实例
train_dataset = MOFDataset(X_train, y_train)
test_dataset = MOFDataset(X_test, y_test)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 模型定义
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.tanh1 = nn.Tanh()
        self.dropout1 = nn.Dropout(p=0.0)

        self.layer2 = nn.Linear(64, 128)
        self.tanh2 = nn.Tanh()
        self.dropout2 = nn.Dropout(p=0.0)

        self.layer3 = nn.Linear(128, 256)
        self.tanh3 = nn.Tanh()
        self.dropout3 = nn.Dropout(p=0.0)

        self.layer4 = nn.Linear(256, 256)
        self.tanh4 = nn.Tanh()
        self.dropout4 = nn.Dropout(p=0.0)

        self.layer5 = nn.Linear(256, 128)
        self.tanh5 = nn.Tanh()
        self.dropout5 = nn.Dropout(p=0.0)

        self.layer6 = nn.Linear(128, 64)
        self.tanh6 = nn.Tanh()
        self.dropout6 = nn.Dropout(p=0.0)

        self.layer7 = nn.Linear(64, n_output)

    def forward(self, x):
        x = self.layer1(x)
        x = self.tanh1(x)
        x = self.dropout1(x)

        x = self.layer2(x)
        x = self.tanh2(x)
        x = self.dropout2(x)

        x = self.layer3(x)
        x = self.tanh3(x)
        x = self.dropout3(x)

        x = self.layer4(x)
        x = self.tanh4(x)
        x = self.dropout4(x)

        x = self.layer5(x)
        x = self.tanh5(x)
        x = self.dropout5(x)

        x = self.layer6(x)
        x = self.tanh6(x)
        x = self.dropout6(x)

        x = self.layer7(x)
        return x


# 初始化模型
input_dim = X_train.shape[1]
model = NeuralNetwork(input_dim=input_dim).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 修改训练函数以处理批次
def train_model(model, criterion, optimizer, train_dataloader, test_dataloader, mask_probability, num_epochs):
    train_loss_history = []
    val_loss_history = []
    best_val_loss = float('inf')
    best_model_state_dict = None

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0

        for inputs, targets in train_dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            """
            #定义遮罩
            if n_parameter == 3:
                mask = torch.tensor([[0, 0, 0]+[1]*n_pressures for _ in range(len(outputs))],dtype=torch.float).to(device)
            elif n_parameter == 4:
                mask = torch.tensor([[0, 0, 0, 0]+[1]*n_pressures for _ in range(len(outputs))],dtype=torch.float).to(device)
            else:
                print("error!, the number of parameters must be 3 or 4")
                logging.info("error!, the number of parameters must be 3 or 4")
            """
            mask = torch.ones((len(outputs), n_output)).to(device).bernoulli(mask_probability)

            # 应用掩码遮罩预测值与目标值
            masked_outputs = outputs * mask
            masked_targets = targets * mask
            loss = nn.MSELoss(reduction='sum')(masked_outputs, masked_targets) / mask.sum()
            # loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * inputs.size(0)

        epoch_train_loss = running_train_loss / len(train_dataset)

        model.eval()
        with torch.no_grad():
            running_val_loss = 0.0

            for inputs, targets in test_dataloader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                val_outputs = model(inputs)
                val_loss = criterion(val_outputs, targets)

                running_val_loss += val_loss.item() * inputs.size(0)

            epoch_val_loss = running_val_loss / len(test_dataset)

        train_loss_history.append(epoch_train_loss)
        val_loss_history.append(epoch_val_loss)

        # Print the losses for this epoch
        if (epoch + 1) % 100 == 0:
            print(f"Epoch: {epoch + 1}, Train Loss: {epoch_train_loss:.4e}, Val Loss: {epoch_val_loss:.4e}")
            logging.info(f"Epoch: {epoch + 1}, Train Loss: {epoch_train_loss:.4e}, Val Loss: {epoch_val_loss:.4e}")
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_state_dict = model.state_dict()
            torch.save(best_model_state_dict,
                       f'best_mask_weights_{num_epochs}_{n_parameter}p_{mask_probability}_[000].pth')

    return train_loss_history, val_loss_history, best_model_state_dict


'''
# 调用训练函数
start_time = time.time()
train_loss_history, val_loss_history, best_model_state_dict = train_model(model, criterion, optimizer, train_dataloader, 
        test_dataloader, mask_probability=mask_probability,num_epochs=num_epochs)
end_time = time.time()
running_time = end_time - start_time
print(f"程序运行时间: {running_time} 秒")
logging.info(f"程序运行时间: {running_time} 秒")
'''

# 绘图
model.load_state_dict(torch.load('best_mask_weights_15000_4p_1.0.pth'))
model.eval()

min_target_tensor = torch.FloatTensor(min_target).to(device)
max_target_tensor = torch.FloatTensor(max_target).to(device)

# 转换为张量
X_train_tensor = torch.FloatTensor(X_train).to(device)
y_train_tensor = torch.FloatTensor(y_train).to(device)
X_test_tensor = torch.FloatTensor(X_test).to(device)
y_test_tensor = torch.FloatTensor(y_test).to(device)

y_traindata_predict = model(X_train_tensor)
y_testdata_predict = model(X_test_tensor)
y_traindata_predict_unscaled = (max_target_tensor - min_target_tensor) * y_traindata_predict + min_target_tensor
y_testdata_predict_unscaled = (max_target_tensor - min_target_tensor) * y_testdata_predict + min_target_tensor

y_train_tensor_unscaled = (max_target_tensor - min_target_tensor) * y_train_tensor + min_target_tensor
y_test_tensor_unscaled = (max_target_tensor - min_target_tensor) * y_test_tensor + min_target_tensor

X_train = X_train_tensor.cpu().detach().numpy()
X_test = X_test_tensor.cpu().detach().numpy()

X_train_unscaled = scaler.inverse_transform(X_train)
X_test_unscaled = scaler.inverse_transform(X_test)
Temp_train = X_train_unscaled[:, -1]
Temp_test = X_test_unscaled[:, -1]

y_true_testdata = y_test_tensor_unscaled.cpu().detach().numpy()
y_true_traindata = y_train_tensor_unscaled.cpu().detach().numpy()

y_predict_testdata = y_testdata_predict_unscaled.cpu().detach().numpy()
y_predict_traindata = y_traindata_predict_unscaled.cpu().detach().numpy()


def langmuir_isotherms(pressures, Temp, parameters):
    assert len(parameters) == n_parameter, "The number of parameters do not match"
    if len(parameters) == 3:
        b0 = parameters[0]
        qmax = parameters[1]
        delta_H = parameters[2]
        n_ads = qmax * (b0 * pressures * np.exp(delta_H / Temp) /
                        (1 + b0 * pressures * np.exp(delta_H / Temp)))
    elif len(parameters) == 4:
        qmax_a = parameters[0]
        qmax_b = parameters[1]
        b0 = parameters[2]
        delta_H = parameters[3]
        n_ads = (qmax_a + Temp * qmax_b) * (b0 * pressures * np.exp(delta_H / Temp) /
                                            (1 + b0 * pressures * np.exp(delta_H / Temp)))
    else:
        print("error!, the number of parameters must be 3 or 4")
        logging.info("error!, the number of parameters must be 3 or 4")
    return n_ads


def plot_isotherms(y_true, y_predict, pressures, Temp, title=None, is_show=True):
    n_ads_true = y_true[n_parameter:]
    n_ads_hat = y_predict[n_parameter:]
    n_lang_true = langmuir_isotherms(pressures, Temp, y_true[0:n_parameter])
    n_lang_hat = langmuir_isotherms(pressures, Temp, y_predict[0:n_parameter])

    error_normal = [mse(n_ads_true, n_ads_hat), r2(n_ads_true, n_ads_hat), mae(n_ads_true, n_ads_hat),
                    mape(n_ads_true, n_ads_hat)]
    error_lang = [mse(n_lang_true, n_lang_hat), r2(n_lang_true, n_lang_hat), mae(n_lang_true, n_lang_hat),
                  mape(n_lang_true, n_lang_hat)]
    return error_normal, error_lang


X_scaled = scaler.transform(X)
X_tensor = torch.FloatTensor(X_scaled).to(device)
y_predict = model(X_tensor)
y_predict_unscaled = (max_target_tensor - min_target_tensor) * y_predict + min_target_tensor

y_hat = y_predict_unscaled.cpu().detach().numpy()
y_true = y

# Prepare the output header
if n_parameter == 3:
    output_train_text = 'name,b0,qmax,delta_H,p_0,p_1,p_2,p_3,p_4,p_5,p_6,p_7,p_8,p_9,p_10,p_11,p_12, \
    p_13,p_14,p_15,p_16,p_17,p_18,p_19,p_20,b0_hat,qmax_hat,delta_H_hat,p_0_hat,p_1_hat,p_2_hat, \
    p_3_hat,p_4_hat,p_5_hat,p_6_hat,p_7_hat,p_8_hat,p_9_hat,p_10_hat,p_11_hat,p_12_hat,p_13_hat,p_14_hat,p_15_hat, \
    p_16_hat,p_17_hat,p_18_hat,p_19_hat,p_20_hat,mse_n,r2_n,mae_n,mape_n,mse_lang,r2_lang,mae_lang,mape_lang\n'
    output_test_text = output_train_text  # Same header for test file
elif n_parameter == 4:
    output_train_text = 'name,qmax_a,qmax_b,b0,delta_H,p_0,p_1,p_2,p_3,p_4,p_5,p_6,p_7,p_8,p_9,p_10,p_11,p_12, \
    p_13,p_14,p_15,p_16,p_17,p_18,p_19,p_20,qmax_a_hat,qmax_b_hat,b0_hat,delta_H_hat,p_0_hat,p_1_hat,p_2_hat, \
    p_3_hat,p_4_hat,p_5_hat,p_6_hat,p_7_hat,p_8_hat,p_9_hat,p_10_hat,p_11_hat,p_12_hat,p_13_hat,p_14_hat,p_15_hat, \
    p_16_hat,p_17_hat,p_18_hat,p_19_hat,p_20_hat,mse_n,r2_n,mae_n,mape_n,mse_lang,r2_lang,mae_lang,mape_lang\n'
    output_test_text = output_train_text  # Same header for test file
else:
    print("error!, the number of parameters must be 3 or 4")
    logging.info("error!, the number of parameters must be 3 or 4")

# Split the results into train and test
for i in range(len(y_hat)):
    name = data['name'][i]
    Temp = data['TK'][i]
    error_normal, error_lang = plot_isotherms(y_true[i, :], y_hat[i, :], pressures=pressures, Temp=Temp,
                                              title=f'{name}@{Temp}K', is_show=False)
    arr_str1 = ','.join(map(str, y_true[i, :]))
    arr_str2 = ','.join(map(str, y_hat[i, :]))
    error_nstr = ','.join(map(str, error_normal))
    error_langstr = ','.join(map(str, error_lang))

    output_line = f'{name},{arr_str1},{arr_str2},{error_nstr}, {error_langstr}\n'
    print(name, *y_true[i, :], *y_hat[i, :], *error_normal, *error_lang, sep=',')
    logging.info(f'{name},{arr_str1},{arr_str2},{error_nstr}, {error_langstr}\n')

    # Determine if this entry is in the train or test set based on the index
    if i < len(X_train):  # If it's in the training set
        output_train_text += output_line
    else:  # If it's in the testing set
        output_test_text += output_line

# Write results to two separate files
train_file_name = f'results_train_{num_epochs}_{n_parameter}P_{mask_probability}.csv'
test_file_name = f'results_test_{num_epochs}_{n_parameter}P_{mask_probability}.csv'

with open(train_file_name, 'w') as f:
    f.write(output_train_text)

with open(test_file_name, 'w') as f:
    f.write(output_test_text)

logging.info(f"Training data results saved to {train_file_name}")
logging.info(f"Testing data results saved to {test_file_name}")



