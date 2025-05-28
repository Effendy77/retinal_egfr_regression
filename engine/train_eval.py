
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    y_true, y_pred = [], []

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)
        y_true.extend(targets.cpu().numpy())
        y_pred.extend(outputs.cpu().numpy())

    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)

    return {'mae': mae, 'rmse': rmse, 'r2': r2}
