import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
from models.retfound_regression import build_model
import argparse

def get_last_conv_layer(model):
    for name, module in reversed(model.named_modules()):
        if isinstance(module, torch.nn.Conv2d):
            return name
    return None

def apply_gradcam(model, image_tensor, target_layer):
    gradients = []
    activations = []

    def save_gradient(grad):
        gradients.append(grad)

    def forward_hook(module, input, output):
        activations.append(output)
        output.register_hook(save_gradient)

    handle = dict(model.named_modules())[target_layer].register_forward_hook(forward_hook)
    model.eval()
    output = model(image_tensor.unsqueeze(0))
    handle.remove()

    grads = gradients[0]
    acts = activations[0]

    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * acts).sum(dim=1).squeeze()

    cam = torch.relu(cam)
    cam -= cam.min()
    cam /= cam.max()
    cam = cam.cpu().detach().numpy()

    return cam

def show_cam_on_image(img, mask, output_path):
    heatmap = plt.get_cmap("jet")(mask)[..., :3]
    heatmap = (heatmap * 255).astype(np.uint8)
    overlay = 0.5 * np.array(img) + 0.5 * heatmap
    overlay = np.uint8(overlay)
    Image.fromarray(overlay).save(output_path)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model()
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    image = Image.open(args.image).convert("RGB")
    image_tensor = transform(image).to(device)

    target_layer = get_last_conv_layer(model)
    mask = apply_gradcam(model, image_tensor, target_layer)
    mask = np.uint8(255 * mask)
    mask = Image.fromarray(mask).resize(image.size, Image.BILINEAR)
    mask = np.array(mask) / 255

    output_path = args.output or "gradcam_output.png"
    show_cam_on_image(image, mask, output_path)
    print(f"Grad-CAM saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--output", help="Path to save Grad-CAM image")
    args = parser.parse_args()
    main(args)
