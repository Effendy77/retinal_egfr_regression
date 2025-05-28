import os
import torch
from engine.train_eval import train_one_epoch, evaluate
from models.retfound_regression import build_model
from datasets.retinal_dataset import RetinalDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd

def main():
    # Paths
    image_dir = "data/images"
    train_csv = "data/fold1_train.csv"
    val_csv = "data/fold1_val.csv"
    output_path = "checkpoints/retfound_egfr_fold1.pth"

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Dataset and DataLoader
    train_dataset = RetinalDataset(train_csv, image_dir, transform=transform, target_col='egfr')
    val_dataset = RetinalDataset(val_csv, image_dir, transform=transform, target_col='egfr')

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model = build_model()
    model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # Loss
    criterion = torch.nn.MSELoss()

    # Training
    epochs = 50
    for epoch in range(epochs):
        train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        mae, rmse, r2 = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1} - Val MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.3f}")

    # Save model
    torch.save(model.state_dict(), output_path)
    print(f"Model saved to {output_path}")

if __name__ == "__main__":
    main()
