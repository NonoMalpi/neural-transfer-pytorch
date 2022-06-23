import torch
import torchvision.transforms as transforms

from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def image_loader(image_path: str, height: int, width: int) -> torch.Tensor:

    loader = transforms.Compose([
        transforms.Resize((height, width)),  # scale imported image
        transforms.ToTensor()])  # transform it into a torch tensor

    image = Image.open(image_path)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)
