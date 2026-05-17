import torch 
import os 
import matplotlib.pyplot as plt 
from typing import List


def visualizeMap(image, saliency_map, path_model_folder, i=0):
    
    fig, (ax1, ax2) = plt.subplots(1,2)
    fig.suptitle("Image and its Saliency Map next to each other")
    ax1.imshow(image)
    ax1.axis('off')
    ax2.imshow(saliency_map, cmap='viridis')
    ax2.axis('off')
    
    plt.savefig(os.path.join(path_model_folder, f"saliency_map_img_{i}"))

    return None 


def saliency(model, image):
    """
    forward pass image through model --> without softmax! because we need score in the end 
    get index corresponding to the maximum score and the maximum score itself 
    backward function on max_score performs backward pass in the computation graph and 
    calculates the gradient of max_score with respect to nodes in the computation graph 
    Salience is the gradient with respect to the input image now 
    to solve for multiple channels need to take maximum magnitude across all color channels 
    TODO: Make it work for more than one image 
    """
    image.requires_grad_()
    logits = model(image)
    max_score_index = logits.argmax(dim=1)
    max_score = logits[0,max_score_index]
    
    max_score.backward()
    
    saliency, _ = torch.max(image.grad.data.abs(), dim=1)
    return saliency 


def main_saliency(model, dataloader, device, path_model_folder, num_images=1):
    model.eval()

    # Get one image from dataloader
    data_iter = iter(dataloader)
    images, labels, _ = next(data_iter)
    image = images[0:num_images].to(device)
    
    ### normalize tensor images using the mean and standart deviation of the dataset 
    #TODO why normalizing here? main_cka and main_saliency should not normalize just once at the beginning - dataset side
    saliency_val = saliency(model, image)    
    
    # Move image and saliency map to CPU and convert to numpy
    image_np = image.cpu().squeeze().permute(1, 2, 0).detach().numpy()
    saliency_np = saliency_val.cpu().squeeze().numpy()
    
    visualizeMap(image_np, saliency_np, path_model_folder)   



def compute_saliency(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    """
    Compute saliency maps for a batch of images.

    Args:
        model: trained model (CNN or ViT)
        images: tensor of shape (B, C, H, W)

    Returns:
        saliency: tensor of shape (B, H, W)
    """
    model.eval()
    images = images.clone().detach().requires_grad_(True)

    logits = model(images)                      # (B, num_classes)
    preds = logits.argmax(dim=1)                # (B,)
    scores = logits[torch.arange(len(preds)), preds].sum()

    model.zero_grad()
    scores.backward()

    # max over channels
    saliency, _ = images.grad.abs().max(dim=1)
    return saliency


def run_saliency(
    model: torch.nn.Module,
    images: torch.Tensor,
    device: torch.device,
) -> List[plt.Figure]:
    """
    Compute and visualize saliency maps.

    Args:
        model: trained model
        images: tensor (B, C, H, W)
        device: torch.device

    Returns:
        figs: list of matplotlib Figure objects
    """
    images = images.to(device)
    saliency_maps = compute_saliency(model, images)

    figs = []
    for i in range(images.shape[0]):
        img = images[i].detach().cpu()
        sal = saliency_maps[i].detach().cpu()

        # de-normalize for visualization (assumes [-1, 1] or [0,1] already handled upstream)
        img_np = img.permute(1, 2, 0).numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))
        ax1.imshow(img_np)
        ax1.set_title("Image")
        ax1.axis("off")

        ax2.imshow(sal.numpy(), cmap="viridis")
        ax2.set_title("Saliency")
        ax2.axis("off")

        plt.tight_layout()
        figs.append(fig)

    return figs
