import torch
import torch.nn as nn

from torchvision.utils import save_image

from model import get_style_model_and_losses, get_input_optimizer


def run_style_transfer(cnn: nn.modules.container.Sequential,
                       normalization_mean: torch.Tensor, normalization_std: torch.Tensor,
                       content_img: torch.Tensor, style_img: torch.Tensor, input_img: torch.Tensor,
                       num_steps: int = 500, style_weight: int = 1000000, content_weight: int = 1,
                       output_name: str = "output") -> torch.Tensor:

    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(
        cnn=cnn, normalization_mean=normalization_mean, normalization_std=normalization_std,
        style_img=style_img, content_img=content_img)

    # We want to optimize the input and not the model parameters, so we
    # update all the requires_grad fields accordingly
    input_img.requires_grad_(True)
    model.requires_grad_(False)

    optimizer = get_input_optimizer(input_img=input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()
                save_image(input_img, output_name + f"{run[0]}.jpeg")

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img
