import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_absolute_error
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    # task_losses = [0.0] * 3  # Initialize a list to store losses for each task

    weight1 = 1  # Set the weight for task 1
    # weight2 = 1  # Set the weight for task 2
    # weight3 = 1  # Set the weight for task 3

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        # output1, output2, output3 = model(inputs)
        output= model(inputs)
        # Calculate task-specific losses
        loss = criterion(output.squeeze(1), targets)
        # Backpropagate and optimize
        loss_sum = loss
        loss_sum.backward()
        optimizer.step()
        
        total_loss += loss_sum.item()

    avg_total_loss = total_loss / len(train_loader)
    # avg_task_losses = [task_loss / len(train_loader) for task_loss in task_losses]

    # return avg_total_loss, avg_task_losses[0], avg_task_losses[1], avg_task_losses[2]
    return avg_total_loss

def test_model(model, test_loader, criterion):
    model.eval()
    total_loss = 0.0
    # task_losses = [0.0] * 3
    total_mse = 0.0
    avg_rmse = 0.0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            # output1, output2, output3 = model(inputs)
            output= model(inputs)
            # Calculate task-specific losses
            loss = criterion(output.squeeze(1), targets)
            mse = nn.MSELoss()(output.squeeze(1).cpu(), targets.cpu())
            loss_sum = loss  
            # Accumulate losses for each task
            # task_losses[0] += loss1.item()
            # task_losses[1] += loss2.item()
            # task_losses[2] += loss3.item()
            total_loss += loss_sum.item()
            total_mse += mse.item()

        avg_mse = total_mse / len(test_loader)
        avg_loss = total_loss / len(test_loader)
        # avg_task_losses = [task_loss / len(test_loader) for task_loss in task_losses]
        avg_rmse = np.sqrt(avg_mse)
        # avg_task_rmses = [rmse / len(test_loader) for rmse in avg_task_losses]

    # return total_mse, avg_task_losses[0], avg_task_losses[1], avg_task_losses[2], total_rmse, avg_task_rmses[0], avg_task_rmses[1], avg_task_rmses[2]
    return avg_loss, avg_mse, avg_rmse