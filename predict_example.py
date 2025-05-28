import torch
from torchvision import transforms
from PIL import Image
import argparse
from models.retfound_regression import build_model

def predict(image_path, checkpoint):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = build_model()
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device)
    model.eval()

    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(image_tensor).item()

    print(f"Predicted eGFR: {prediction:.2f}")
    return prediction

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run eGFR prediction on a new image.")
    parser.add_argument("--image", required=True, help="Path to image file (PNG or JPG)")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pth)")
    args = parser.parse_args()

    predict(args.image, args.checkpoint)
