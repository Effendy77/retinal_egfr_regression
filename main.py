
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from models.retfound_regression import RETFoundRegression
from datasets.retinal_dataset import RetinalPWVDataset
from engine.train_eval import train_one_epoch, evaluate
import timm
import os

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Configs
    image_dir = 'data/images'
    train_csv = 'data/fold1_train.csv'
    val_csv = 'data/fold1_val.csv'
    batch_size = 32
    num_epochs = 10
    lr = 1e-4

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Datasets and loaders
    train_dataset = RetinalPWVDataset(train_csv, image_dir, transform)
    val_dataset = RetinalPWVDataset(val_csv, image_dir, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Load base RETFound model
    backbone = timm.create_model('vit_base_patch16_224', pretrained=True)
    model = RETFoundRegression(backbone).to(device)

    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    # Training loop
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val MAE: {val_metrics['mae']:.4f} | RMSE: {val_metrics['rmse']:.4f} | R2: {val_metrics['r2']:.4f}")

    # Save model
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/retfound_regression_fold1.pth')

if __name__ == '__main__':
    main()
