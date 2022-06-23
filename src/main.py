import argparse

import torch
import torchvision.models as models

from aux import image_loader
from model import run_style_transfer


parser = argparse.ArgumentParser('Neural transfer')
parser.add_argument('--style', type=str)
parser.add_argument('--content', type=str)
parser.add_argument('--height', type=int)
parser.add_argument('--width', type=int)
parser.add_argument('--num_steps', type=int, default=500)
parser.add_argument('--style_weight', type=int, default=100000)
parser.add_argument('--content_weight', type=float, default=1)
args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cnn = models.vgg19(pretrained=True).features.to(device).eval()

# VGG networks are trained on images with each channel normalized
# by mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)


style_img = image_loader(f"images/styles/{args.style}.jpeg", height=args.height, width=args.width)
content_img = image_loader(f"images/beaches/{args.content}.jpeg", height=args.height, width=args.width)

assert style_img.size() == content_img.size(), \
    "we need to import style and content images of the same size"

input_img = content_img.clone()

output_name = f"images/output/{args.content}_{args.style}_{args.num_steps}_{args.style_weight}_{args.content_weight}_iteration_"
output = run_style_transfer(cnn=cnn,
                            normalization_mean=cnn_normalization_mean, normalization_std=cnn_normalization_std,
                            content_img=content_img, style_img=style_img, input_img=input_img,
                            num_steps=args.num_steps, style_weight=args.style_weight, content_weight=args.content_weight,
                            output_name=output_name)



