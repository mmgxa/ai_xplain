from pathlib import Path
import os
import matplotlib
import timm
from mdutils.mdutils import MdUtils


# import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T

# from torchvision.models import resnet34
from captum.attr import (
    GradientShap,
    IntegratedGradients,
    NoiseTunnel,
    Occlusion,
    Saliency,
)
from captum.attr import visualization as viz
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

src_path = Path("./src/")
out_path = Path("./out/")


def get_integrated_gradients(
    file,
    model,
    image_tensor: torch.Tensor,
    default_cmap: LinearSegmentedColormap,
    transformed_img: torch.Tensor,
    pred_label_idx: torch.Tensor,
) -> None:
    """To explain the model using IntegratedGradients."""

    integrated_gradients = IntegratedGradients(model)
    attributions_ig = integrated_gradients.attribute(
        image_tensor, target=pred_label_idx, n_steps=10
    )

    a = viz.visualize_image_attr(
        np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        method="heat_map",
        cmap=default_cmap,
        show_colorbar=True,
        sign="positive",
        outlier_perc=1,
        use_pyplot=False,
        title="IntegratedGradients",
    )
    a[0].savefig(out_path / f"{file}_integ_grad.png")


def get_noise_tunnel(
    file,
    model,
    image_tensor: torch.Tensor,
    default_cmap: LinearSegmentedColormap,
    transformed_img: torch.Tensor,
    pred_label_idx: torch.Tensor,
) -> None:
    """To explain the model using Noise tunnel with IntegratedGradients."""

    integrated_gradients = IntegratedGradients(model)
    noise_tunnel = NoiseTunnel(integrated_gradients)

    attributions_ig_nt = noise_tunnel.attribute(
        image_tensor, nt_samples=1, nt_type="smoothgrad", target=pred_label_idx
    )

    b = viz.visualize_image_attr_multiple(
        np.transpose(attributions_ig_nt.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        # ["original_image", "heat_map"],
        # ["all", "positive"],
        ["heat_map"],
        ["positive"],
        cmap=default_cmap,
        show_colorbar=True,
        use_pyplot=False,
    )
    b[0].savefig(out_path / f"{file}_integ_grad_noise.png")


def get_shap(
    file,
    model,
    image_tensor: torch.Tensor,
    default_cmap: LinearSegmentedColormap,
    transformed_img: torch.Tensor,
    pred_label_idx: torch.Tensor,
) -> None:
    """To explain the model using SHAP."""

    gradient_shap = GradientShap(model)

    rand_img_dist = torch.cat([image_tensor * 0, image_tensor * 1])

    attributions_gs = gradient_shap.attribute(
        image_tensor,
        n_samples=50,
        stdevs=0.0001,
        baselines=rand_img_dist,
        target=pred_label_idx,
    )

    c = viz.visualize_image_attr_multiple(
        np.transpose(attributions_gs.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        # ["original_image", "heat_map"],
        # ["all", "absolute_value"],
        ["heat_map"],
        ["absolute_value"],
        cmap=default_cmap,
        show_colorbar=True,
        use_pyplot=False,
    )
    c[0].savefig(out_path / f"{file}_grad_shap.png")


def get_occlusion(
    file,
    model,
    image_tensor: torch.Tensor,
    transformed_img: torch.Tensor,
    pred_label_idx: torch.Tensor,
) -> None:
    """To explain the model using Occlusion."""

    occlusion = Occlusion(model)

    attributions_occ = occlusion.attribute(
        image_tensor,
        strides=(3, 8, 8),
        target=pred_label_idx,
        sliding_window_shapes=(3, 15, 15),
        baselines=0,
    )

    d = viz.visualize_image_attr_multiple(
        np.transpose(attributions_occ.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        # ["original_image", "heat_map"],
        # ["all", "positive"],
        ["heat_map"],
        ["positive"],
        show_colorbar=True,
        outlier_perc=2,
        use_pyplot=False,
    )
    d[0].savefig(out_path / f"{file}_occlusion.png")


def get_saliency(
    file, model, image_tensor: torch.Tensor, pred_label_idx: torch.Tensor
) -> None:
    """To explain the model using Saliency."""

    saliency = Saliency(model)
    grads = saliency.attribute(image_tensor, target=pred_label_idx)
    grads = np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0))

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    inv_transform = T.Compose(
        [
            T.Normalize(
                mean=(-1 * np.array(mean) / np.array(std)).tolist(),
                std=(1 / np.array(std)).tolist(),
            )
        ]
    )

    original_image = np.transpose(
        inv_transform(image_tensor).squeeze(0).cpu().detach().numpy(), (1, 2, 0)
    )

    _ = viz.visualize_image_attr(
        None, original_image, method="original_image", title="Original Image"
    )
    e = viz.visualize_image_attr(
        grads,
        original_image,
        method="blended_heat_map",
        sign="absolute_value",
        show_colorbar=True,
        title="Overlaid Gradient Magnitudes",
        use_pyplot=False,
    )
    e[0].savefig(out_path / f"{file}_saliency.png")


def get_gradcam(
    file, model, image_tensor: torch.Tensor, pred_label_idx: torch.Tensor
) -> None:
    """To explain the model using GradCAM."""

    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)  # ,use_cuda=True)
    targets = [ClassifierOutputTarget(pred_label_idx)]

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=image_tensor, targets=targets)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    inv_transform = T.Compose(
        [
            T.Normalize(
                mean=(-1 * np.array(mean) / np.array(std)).tolist(),
                std=(1 / np.array(std)).tolist(),
            )
        ]
    )

    grayscale_cam = grayscale_cam[0, :]
    rgb_img = (
        inv_transform(image_tensor).cpu().squeeze().permute(1, 2, 0).detach().numpy()
    )
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    matplotlib.image.imsave(out_path / f"{file}_gradcam.png", visualization)


def get_gradcamplusplus(
    file, model, image_tensor: torch.Tensor, pred_label_idx: torch.Tensor
) -> None:
    """To explain the model using GradCAM++"""

    target_layers = [model.layer4[-1]]
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers)  # , use_cuda=True)
    targets = [ClassifierOutputTarget(pred_label_idx)]

    grayscale_cam = cam(input_tensor=image_tensor, targets=targets)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    inv_transform = T.Compose(
        [
            T.Normalize(
                mean=(-1 * np.array(mean) / np.array(std)).tolist(),
                std=(1 / np.array(std)).tolist(),
            )
        ]
    )

    grayscale_cam = grayscale_cam[0, :]
    rgb_img = (
        inv_transform(image_tensor).cpu().squeeze().permute(1, 2, 0).detach().numpy()
    )
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    matplotlib.image.imsave(out_path / f"{file}_gradcampp.png", visualization)


def explain_model() -> None:
    model = timm.create_model("resnet18", pretrained=True)
    # model.fc = torch.nn.Linear(512, 6)
    # model.load_state_dict(torch.load("model.bin"))
    device = torch.device("cuda:0")
    model = model.to(device)

    # categories = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transforms = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
        ]
    )
    transform_normalize = T.Normalize(mean=mean, std=std)

    file_path = Path("./explanation.md")

    with open(file_path, "w") as f:
        f.write("# Model Explanation  \n")
        f.write(
            "| Original Image| Integrated Gradients \
                | Noise Tunnel | Saliency | Occlusion | SHAP | GradCAM | GradCAM++ | \n"
        )
        f.write(
            "| -------- | -------- \
                | -------- | -------- | -------- | -------- | -------- | -------- | \n"
        )

        images = src_path.glob("*.JPEG")
        for file in images:
            print(file)
            filename = os.path.splitext(os.path.basename(file))[0]
            print(filename)
            image = Image.open(file)
            transformed_img = transforms(image)
            image_tensor = transform_normalize(transformed_img)
            image_tensor = image_tensor.unsqueeze(0).to(device)

            output = model(image_tensor)
            output = F.softmax(output, dim=1)
            _, pred_label_idx = torch.topk(output, 1)

            pred_label_idx.squeeze_()
            # predicted_label = categories[pred_label_idx.item()]

            default_cmap = LinearSegmentedColormap.from_list(
                "custom blue",
                [(0, "#ffffff"), (0.25, "#000000"), (1, "#000000")],
                N=256,
            )

            get_integrated_gradients(
                filename,
                model,
                image_tensor,
                default_cmap,
                transformed_img,
                pred_label_idx,
            )

            get_noise_tunnel(
                filename,
                model,
                image_tensor,
                default_cmap,
                transformed_img,
                pred_label_idx,
            )

            get_shap(
                filename,
                model,
                image_tensor,
                default_cmap,
                transformed_img,
                pred_label_idx,
            )

            get_occlusion(
                filename, model, image_tensor, transformed_img, pred_label_idx
            )

            image_tensor_grad = image_tensor
            image_tensor_grad.requires_grad = True

            get_saliency(filename, model, image_tensor_grad, pred_label_idx)
            get_gradcam(filename, model, image_tensor_grad, pred_label_idx)
            get_gradcamplusplus(filename, model, image_tensor_grad, pred_label_idx)

            f.write(
                f'| <p align="center" style="padding: 10px"><img src="{file}" width=250><br></p>\
                 | <p align="center" style="padding: 10px"><img src="./out/{filename}_integ_grad.png" width=250><br></p>\
                 | <p align="center" style="padding: 10px"><img src="./out/{filename}_integ_grad_noise.png" width=250><br></p>\
                 | <p align="center" style="padding: 10px"><img src="./out/{filename}_saliency.png" width=250><br></p>\
                 | <p align="center" style="padding: 10px"><img src="./out/{filename}_occlusion.png" width=250><br></p>\
                 | <p align="center" style="padding: 10px"><img src="./out/{filename}_grad_shap.png" width=250><br></p>\
                 | <p align="center" style="padding: 10px"><img src="./out/{filename}_gradcam.png" width=250><br></p>\
                 | <p align="center" style="padding: 10px"><img src="./out/{filename}_gradcampp.png" width=250><br></p> |  \n'
            )


def main() -> None:
    explain_model()


if __name__ == "__main__":
    main()
