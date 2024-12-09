import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import csv
import pandas as pd
import os
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# def load_data(train_file, test_file):
#     train_data = []
#     test_data = []
#     #train_data
#     with open(train_file, "r") as f:
#         reader = csv.reader(f, delimiter=",")
#         next(reader)
#         for row in reader:
#             train_data.append([ float(row[2]), float(row[3]), float(row[4]),float(row[5])])  #  S1, S2,S3, T1
#     #test_data
#     with open(test_file, "r") as f:
#         reader = csv.reader(f, delimiter=",")
#         next(reader)
#         for row in reader:
#             test_data.append([ float(row[2]), float(row[3]), float(row[4]),float(row[5])])  #  S1, S2,S3, T1
#     # 进行数据划分，可以根据你的需求来调整划分比例
#     Val_ratio = 0.5  # 训练集占总数据集的比例
#     Val_size = int(Val_ratio * len(test_data))
#     val_data, test_data = test_data[:Val_size], test_data[Val_size:]

#     return train_data, val_data, test_data

def load_trial_data(data_folder, trial_name):
    data = []
    file_path = os.path.join(data_folder, f"filter_merged_data_{trial_name}.csv")
    with open(file_path, "r") as f:
        reader = csv.reader(f, delimiter=",")
        next(reader)
        for row in reader:
            data.append([float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7])])
    return data

def load_data(data_folder):
    train_data = []
    val_data = []
    test_data = []
    test_trial_sizes=[]
    val_trial_sizes=[]
    train_trial_sizes=[]
    Val_ratio = 0.5
    # 读取测试集数据
    for A in range(0, 360, 45):
        for B in range(0, 11):
            trial_name = f"{A}d-{B}x-10y-1cms-10"
            test_trial_data = load_trial_data(data_folder, trial_name)
            Val_size = int(Val_ratio * len(test_trial_data))
            val_trial_data, test_trial_data = test_trial_data[:Val_size], test_trial_data[Val_size:]
            val_data.extend(val_trial_data)
            test_data.extend(test_trial_data)
            val_trial_sizes.append(len(val_trial_data))
            test_trial_sizes.append(len(test_trial_data))

    # 读取训练集数据
    for A in range(0, 360, 45):
        for B in range(0, 11):
            trial_name = f"{A}d-{B}x-10y-1cms-20"
            train_trial_data = load_trial_data(data_folder, trial_name)
            train_data.extend(train_trial_data)
            train_trial_sizes.append(len(train_trial_data))


    return train_data, val_data, test_data, train_trial_sizes,val_trial_sizes,test_trial_sizes

# class MyDataset(Dataset):
#     def __init__(self, data, targets, sequence_length):
#         self.data = data
#         self.targets = targets
#         self.sequence_length = sequence_length

#     def __len__(self):
#         return len(self.data) - self.sequence_length + 1

#     def __getitem__(self, idx):
#         # 取连续的 sequence_length 个数据组成滑动窗口
#         input_data = self.data[idx:idx + self.sequence_length]

#         target_data = self.targets[idx + self.sequence_length - 1]  # 对应的目标数据在滑动窗口的最后一个位置
        
#         input_data = torch.tensor(input_data, dtype=torch.float32)
        
#         target_data = torch.tensor(target_data, dtype=torch.float32)

#         target_data = target_data.unsqueeze(0)

#         return input_data, target_data

# class MyDataset(Dataset):
#     def __init__(self, data, targets, sequence_length):
#         self.data = data
#         self.targets = targets
#         self.sequence_length = sequence_length

#     def __len__(self):
#         return len(self.data) - self.sequence_length + 1

#     def __getitem__(self, idx):
#         # 取连续的 sequence_length 个数据组成滑动窗口
#         input_data = self.data[idx:idx + self.sequence_length]

#         target_data = self.targets[idx + self.sequence_length - 1]  # 对应的目标数据在滑动窗口的最后一个位置
        
#         input_data = torch.tensor(input_data, dtype=torch.float32)

        
#         target_data = torch.tensor(target_data, dtype=torch.float32)

#         return input_data, target_data
    
# class VariabletrialDataset(Dataset):
#     def __init__(self, data, targets, sequence_length, trial_sizes):
#         self.data = data
#         self.targets = targets
#         self.sequence_length = sequence_length
#         self.trial_sizes = trial_sizes

#     def __len__(self):
#         return sum(self.trial_sizes) - self.sequence_length * len(self.trial_sizes) + len(self.trial_sizes)

#     def __getitem__(self, index):
#         trial_idx = 0
#         while index >= sum(self.trial_sizes[:trial_idx]):
#             trial_idx += 1

#         trial_size = self.trial_sizes[trial_idx - 1]
#         start_idx = sum(self.trial_sizes[:trial_idx - 1])
#         end_idx = start_idx + trial_size - self.sequence_length + 1

#         input_data_batch = []
#         target_data_batch = []

#         for i in range(start_idx, end_idx, 1):
#             input_data = self.data[i:i + self.sequence_length]
#             target_data = self.targets[i + self.sequence_length - 1]
#             input_data = torch.tensor(input_data, dtype=torch.float32)
#             target_data = torch.tensor(target_data, dtype=torch.float32)
#             input_data_batch.append(input_data)
#             target_data_batch.append(target_data)

#         input_data_batch = torch.stack(input_data_batch, dim=0)
#         target_data_batch = torch.stack(target_data_batch, dim=0)

#         return input_data_batch, target_data_batch
        
def dataset( data, targets, sequence_length, trial_sizes):
        data = data
        targets = targets
        sequence_length = sequence_length
        trial_sizes = trial_sizes
        output_pair=[]
        for i in range (len(trial_sizes)):
            trial_size = trial_sizes[i]
            start_idx = sum(trial_sizes[:i])
            end_idx = start_idx + trial_size
            for j in range(start_idx, end_idx, 1):
                if j -start_idx -sequence_length+1 < 0:
                    num_zeros = start_idx + sequence_length - j - 1
                    input_data = np.pad(data[start_idx:j+1], ((num_zeros, 0), (0, 0)), 'constant')
                else:
                    input_data = data[j - sequence_length + 1 : j + 1]
                target_data = targets[j]
                input_data = torch.tensor(input_data, dtype=torch.float32)
                target_data = torch.tensor(target_data, dtype=torch.float32)
                output_pair.append((input_data,target_data))
        return output_pair   


def save_output_image(model, data_loader, laser, output_file):
    model.eval()
    all_outputs = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)

            outputs= model(inputs)

            # 将输出和真实标签保存到列表中
            all_outputs.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # 将列表转换为 numpy 数组
    all_outputs = np.concatenate(all_outputs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # 在一个图像中展示所有验证集样本的输出
    plt.figure(figsize=(25, 6))
    plt.plot(laser, label='Laser')
    plt.plot(all_outputs, label='Whisker')
    plt.plot(all_labels, label='Ground Truth')
    plt.legend()
    plt.xlabel('Sample Index')
    plt.ylabel('Output')
    plt.savefig(f"{output_file}")
    plt.close()

def save_deviation_image(model, data_loader, laser, output_file):
    model.eval()
    all_outputs = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)

            outputs= model(inputs)

            # 将输出和真实标签保存到列表中
            all_outputs.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # 将列表转换为 numpy 数组
    all_outputs = np.concatenate(all_outputs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # output the deviation
    plt.figure(figsize=(10, 6))
    plt.scatter(laser , all_labels,s=3,label="laser")
    plt.scatter(all_outputs , all_labels,s=3,label='whisker')
    plt.scatter(all_labels , all_labels ,s=3,label='GT')
    plt.title(f"MLP_deviation")
    plt.legend()
    plt.xlabel('Distance (mm)')
    plt.ylabel('Output')
    plt.savefig(f"{output_file}")
    plt.close()

def save_reconstruction_surface_1(model, data_loader, orientation, position, laser, output_file):
    model.eval()
    all_outputs = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)

            outputs= model(inputs)

            # 将输出和真实标签保存到列表中
            all_outputs.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # 将列表转换为 numpy 数组
    all_outputs = np.concatenate(all_outputs, axis=0).flatten()
    all_labels = np.concatenate(all_labels, axis=0).flatten()

    position_X = position[:,1] * 1000
    position_Y = position[:,0] * 1000

    GT_x = -(218 - all_labels + 16) * np.sin(np.radians(orientation)) + position_X
    GT_y = (218 - all_labels + 16) * np.cos(np.radians(orientation)) + position_Y

    Laser_x = -(218 - laser + 16) * np.sin(np.radians(orientation)) + position_X
    Laser_y = (218 - laser + 16) * np.cos(np.radians(orientation)) + position_Y
    
    Whisker_x = -(218 - all_outputs +16) * np.sin(np.radians(orientation)) + position_X
    Whisker_y = (218 - all_outputs +16) * np.cos(np.radians(orientation)) + position_Y
    # output the deviation
    plt.figure(figsize=(10, 6))
    plt.scatter(GT_x[0:151], GT_y[0:151], s=1, c='black', label='GT')
    plt.scatter(position_X[0:151] , position_Y[0:151] ,s=1, label='position')
    plt.scatter(Laser_x[0:151], Laser_y[0:151],s=1, label='laser')
    plt.scatter(Whisker_x[0:151], Whisker_y[0:151],s=1, label='whisker')
    plt.title(f"Surface Reconstruction_Whisker_trial1")
    plt.legend()
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.savefig(f"{output_file[:-4]}_trial1{output_file[-4:]}")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.scatter(GT_x[151:305], GT_y[151:305], s=1, c='black', label='GT')
    plt.scatter(position_X[151:305] , position_Y[151:305] ,s=1, label='position')
    plt.scatter(Laser_x[151:305], Laser_y[151:305],s=1, label='laser')
    plt.scatter(Whisker_x[151:305], Whisker_y[151:305],s=1, label='whisker')
    plt.title(f"Surface Reconstruction_Whisker_trial2")
    plt.legend()
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.savefig(f"{output_file[:-4]}_trial2{output_file[-4:]}")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.scatter(GT_x[305:445], GT_y[305:445], s=1, c='black', label='GT')
    plt.scatter(position_X[305:445] , position_Y[305:445] ,s=1, label='position')
    plt.scatter(Laser_x[305:445], Laser_y[305:445],s=1, label='laser')
    plt.scatter(Whisker_x[305:445], Whisker_y[305:445],s=1, label='whisker')
    plt.title(f"Surface Reconstruction_Whisker_trial3")
    plt.legend()
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.savefig(f"{output_file[:-4]}_trial3{output_file[-4:]}")
    plt.close()

def save_reconstruction_surface_2(model, data_loader, orientation, position, laser, output_file):
    model.eval()
    all_outputs = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)

            outputs= model(inputs)

            # 将输出和真实标签保存到列表中
            all_outputs.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # 将列表转换为 numpy 数组
    all_outputs = np.concatenate(all_outputs, axis=0).flatten()
    all_labels = np.concatenate(all_labels, axis=0).flatten()

    position_X = position[:,1] * 1000
    position_Y = position[:,0] * 1000

    GT_x = -(218 - all_labels + 16) * np.sin(np.radians(orientation)) + position_X
    GT_y = (218 - all_labels + 16) * np.cos(np.radians(orientation)) + position_Y

    Laser_x = -(218 - laser + 16) * np.sin(np.radians(orientation)) + position_X
    Laser_y = (218 - laser + 16) * np.cos(np.radians(orientation)) + position_Y
    
    Whisker_x = -(218 - all_outputs +16) * np.sin(np.radians(orientation)) + position_X
    Whisker_y = (218 - all_outputs +16) * np.cos(np.radians(orientation)) + position_Y
    # output the deviation
    plt.figure(figsize=(10, 6))
    plt.scatter(GT_x[0:151], GT_y[0:151], s=1, c='black', label='GT')
    plt.scatter(position_X[0:151] , position_Y[0:151] ,s=1, label='position')
    plt.scatter(Laser_x[0:151], Laser_y[0:151],s=1, label='laser')
    plt.scatter(Whisker_x[0:151], Whisker_y[0:151],s=1, label='whisker')
    plt.title(f"Surface Reconstruction_Whisker_trial1")
    plt.legend()
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.savefig(f"{output_file[:-4]}_trial1{output_file[-4:]}")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.scatter(GT_x[151:314], GT_y[151:314], s=1, c='black', label='GT')
    plt.scatter(position_X[151:314] , position_Y[151:314] ,s=1, label='position')
    plt.scatter(Laser_x[151:314], Laser_y[151:314],s=1, label='laser')
    plt.scatter(Whisker_x[151:314], Whisker_y[151:314],s=1, label='whisker')
    plt.title(f"Surface Reconstruction_Whisker_trial2")
    plt.legend()
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.savefig(f"{output_file[:-4]}_trial2{output_file[-4:]}")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.scatter(GT_x[314:457], GT_y[314:457], s=1, c='black', label='GT')
    plt.scatter(position_X[314:457] , position_Y[314:457] ,s=1, label='position')
    plt.scatter(Laser_x[314:457], Laser_y[314:457],s=1, label='laser')
    plt.scatter(Whisker_x[314:457], Whisker_y[314:457],s=1, label='whisker')
    plt.title(f"Surface Reconstruction_Whisker_trial3")
    plt.legend()
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.savefig(f"{output_file[:-4]}_trial3{output_file[-4:]}")
    plt.close()

    
def save_loss_image(train_losses, val_losses, output_file):
    epochs = len(train_losses)
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
    # plt.plot(range(1, epochs + 1), train_each_losses_1, label='Train each Loss_X')
    # plt.plot(range(1, epochs + 1), val_each_losses_1, label='Validation each Loss_X')
    # plt.plot(range(1, epochs + 1), train_each_losses_2, label='Train each Loss_Y')
    # plt.plot(range(1, epochs + 1), val_each_losses_2, label='Validation each Loss_Y')
    # plt.plot(range(1, epochs + 1), train_each_losses_3, label='Train each Loss_O')
    # plt.plot(range(1, epochs + 1), val_each_losses_3, label='Validation each Loss_O')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(output_file)
    plt.close()

def min_max_normalization(data):
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data

def evaluate(net, test_dataset, output_file):
    """
    Evaluate the model on a test dataset.
    
    Args:
        net: The trained model.
        test_dataset: The test dataset.
        output_file: File path to save the prediction plot.
        
    Returns:
        MSE: Mean Squared Error on the test set.
        MAE: Mean Absolute Error on the test set.
        all_outputs: All predicted outputs.
        all_labels: All ground truth labels.
    """
    net.eval()
    test_MSE = 0
    test_MAE = 0
    all_outputs = None
    all_labels = None
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            
            if all_outputs is None:
                all_outputs = outputs
                all_labels = targets
            else:
                all_outputs = torch.cat((all_outputs, outputs), dim=0)
                all_labels = torch.cat((all_labels, targets), dim=0)
            
            # Compute loss
            MSE_loss = nn.MSELoss()(outputs, targets.unsqueeze(1))
            test_MSE += MSE_loss.item()
            MAE_loss = nn.L1Loss()(outputs, targets.unsqueeze(1))
            test_MAE += MAE_loss.item()
    
    # Plot predictions vs ground truth
    plt.figure(figsize=(25, 5))  # Adjust the width of the canvas
    plt.plot(all_outputs.cpu().numpy(), label='Predictions', color='blue')
    plt.plot(all_labels.cpu().numpy(), label='Ground Truth', color='green')
    plt.legend()
    plt.xlabel('Sample Index')
    plt.ylabel('Output')
    plt.title('Model Predictions vs Ground Truth')
    plt.savefig(output_file)
    
    # Compute average metrics
    MSE = test_MSE / len(test_loader)
    MAE = test_MAE / len(test_loader)
    
    return MSE, MAE, all_outputs.cpu().numpy().flatten()
