import torch 
import numpy as np
import os 
import matplotlib.pyplot as plt 
from architectures.vit import Transformer
from architectures.cnn import CNNModel
import yaml 
from pathlib import Path 
from utilities.load_data import load_petface_data
import torch as t 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 

def cka_viz(cka_results: pd.DataFrame, path, vit_layers, cnn_layers): 
    
    plt.clf() 
 
    heatmap_df = cka_results.pivot(
    index='cnn_id',
    columns='vit_id',
    values='cka'
    ).sort_index().sort_index(axis=1)

    sns.heatmap(
    heatmap_df,
    cmap='viridis'
    )
    
    plt.xlabel("ViT layers")
    plt.ylabel("CNN layers")
    plt.title("CKA: CNN and ViT model")
    plt.savefig(path)

def create_hook(name, activations): 
    def hook(model, input, output): 
        activations[name].append(output.detach().cpu())
    return hook 

def all_hooks(layer_dict): 
    activations = { name: [] for name in layer_dict}
    handles = []
    
    for name in layer_dict: 
        layer = layer_dict[name]
        handle = layer.register_forward_hook(create_hook(name, activations))
        handles.append(handle)
        
    return activations, handles

def cka(layer1, layer2): 
    n = layer1.shape[0]
    center = torch.eye(n=n) - (1/n) * torch.ones([n,n])
    
    
    #layer1 = np.array(layer1)
    #layer2 = np.array(layer2)
    layer1 = t.flatten(layer1, start_dim=1)
    layer2 = t.flatten(layer2, start_dim=1)
    v = layer1 @ layer1.T
    w = layer2 @ layer2.T
    cka_u = t.trace(v @ center @ w @ center)
    
    hv = center @ v @ center
    hw = center @ w @ center
    cka_d_l = torch.norm(hv,p='fro')
    cka_d_r = torch.norm(hw,p='fro')
    cka_d = cka_d_l * cka_d_r 
    
    cka = cka_u / cka_d
    return cka 
    
def main_cka(vit: torch.nn.Module, vit_layers: dict, 
             cnn: torch.nn.Module, cnn_layers: dict,
             dataloader, device, num_batches:int): 
    vit.eval()
    cnn.eval()
    
    ### create hooks 
    vit_activations, vit_handles = all_hooks(vit_layers)
    cnn_activations, cnn_handles = all_hooks(cnn_layers)
    
    ### run through model 
    with torch.no_grad(): 
        i = 0 
        for images, _ , _ in dataloader: 
            images.to(device)
            _ = cnn(images)
            _ = vit(images)
            i += 1 
            if i == num_batches: 
                break 
            

    ### concat activations for all batches 
    for name in vit_activations: 
        vit_activations[name] = torch.cat(vit_activations[name], dim=0)
    
       
    for name in cnn_activations: 
        cnn_activations[name] = torch.cat(cnn_activations[name], dim=0)
       
    ### remove hooks for memory handling 
    for handle in vit_handles + cnn_handles: 
        handle.remove()

    ### calculate the central kernel alignment (CKA)
    cka_results = {}
    
    for vit_name, vit_act in vit_activations.items(): 
        # TO DO: transform activations into shape so fits to cka 
        cka_results[vit_name] = {}
        
        for cnn_name, cnn_acts in cnn_activations.items(): 
            cka_score = cka(vit_act, cnn_acts)
            #print(f"ViT layer: {vit_name} and CNN layer: {cnn_name} have score = {cka_score}")
            cka_results[vit_name][cnn_name] = cka_score
              
    return cka_results

def cka_df_create(cka_results: dict, vit_layers:list, cnn_layers:list): 
    cka_df = pd.DataFrame(cka_results)
    
    cnn_layer_id = {l: i for i, l in enumerate(cnn_layers)}
    vit_layer_id = {l: i for i, l in enumerate(vit_layers)}
    
    cka_df = cka_df.reset_index().melt(id_vars='index', 
                                       var_name='vit_layer', 
                                       value_name='cka').rename(columns={'index':'cnn_layer'})
    
    cka_df['cnn_id'] = cka_df['cnn_layer'].map(cnn_layer_id)
    cka_df['vit_id'] = cka_df['vit_layer'].map(vit_layer_id)   
    cka_df['cka'] = cka_df['cka'].apply(lambda x: x.item() if t.is_tensor(x) else x)
    return cka_df 
     
    
def load_cka(vit_path: os.path, cnn_path: os.path, batch_size: int, device):
    project_root = Path(__file__).resolve().parent 
    
    ## Load data 
    train_ldr, val_ldr, test_ldr, n_breeds  = load_petface_data(batch_size=batch_size,
                                                            label_path=os.path.join(project_root, "data","cat","cat.csv",),
                                                            strip_percent=100,
                                                            visualize_breed_distribution=False) 
    
    ### Load both models 
    tog =  str(vit_path) + "_" + str(cnn_path)
    ## load ViT 
    vit_path = project_root / "data" / "vit" /vit_path 
    vit_cfg_path =  vit_path / "config.yaml"
    vit_model_path = vit_path / "model.pt"
    
    with open(vit_cfg_path) as f: 
        vit_cfg = yaml.safe_load(f)
    
    vit = Transformer(images_size=vit_cfg["data"]["images_size"], 
                        batch_size=vit_cfg["training"]["batch_size"], 
                        patches_size=vit_cfg["model"]["patches_size"], 
                        embedding_dim=vit_cfg["model"]["embedding_dim"],
                        num_encoder_blocks=vit_cfg["model"]["num_encoder_blocks"],
                        num_attention_heads=vit_cfg["model"]["num_attention_heads"],
                        factor_hidden_size_encoder=vit_cfg["model"]["factor_hidden_size_encoder"],
                        dropout_en=vit_cfg["model"]["dropout_en"],
                        num_classes=n_breeds)

    vit.load_state_dict(torch.load(vit_model_path))

    ## load CNN
    cnn_path = project_root / "data" / "cnn"/ cnn_path 
    cnn_cfg_path = cnn_path / "config.yaml"
    cnn_model_path = cnn_path / "model.pt"
    with open(cnn_cfg_path) as f: 
        cfg = yaml.safe_load(f)
    
    cnn = CNNModel(input_channels=cfg["data"]["input_channels"], num_classes=24)
    cnn.load_state_dict(torch.load(cnn_model_path))
    
    ### Define layers for comparison 
    vit_layers = {"lin_emb": vit.linear_proj, 
                 "block_0" : vit.encoder_blocks[0], 
                 "block_1" : vit.encoder_blocks[1],
                 "block_2" : vit.encoder_blocks[1]
                 }
    
    cnn_layers = {"pool_1": cnn.pool1, 
                 "pool_2": cnn.pool2,
                 "pool_3": cnn.pool3,
                 "pool_4": cnn.pool4,
                 "global_avg_pool" : cnn.global_avg_pool, 
                 "fc_1": cnn.fc1, 
                 "fc_2": cnn.fc2}
    
    print("Loaded all models")
    
    
    cka_results = main_cka(vit=vit, cnn=cnn, vit_layers=vit_layers, 
                           cnn_layers=cnn_layers, dataloader=test_ldr,
                           device=device, num_batches= 20)
    
    cka_path = project_root / "data" / "cka_results" / tog
    cka_df_path = cka_path / "cka_results.csv"
    cka_viz_path = cka_path / "visualization.png"
    
    os.makedirs(cka_path, exist_ok=True)
    ### Save cka results as dataframe 
    vit_layers = list(vit_layers.keys())
    cnn_layers = list(cnn_layers.keys())
    cka_df = cka_df_create(cka_results, vit_layers, cnn_layers)
    cka_df.to_csv(cka_df_path, index=False)
    
    ### Visualize CKA  
    cka_viz(cka_df, cka_viz_path, vit_layers, cnn_layers)


### forward pass image through model --> without softmax! because we need score in the end 
### get index corresponding to the maximum score and the maximum score itself 
### backward function on max_score performs backward pass in the computation graph and 
### calculates the gradient of max_score with respect to nodes in the computation graph 
### Salience is the gradient with respect to the input image now 
### to solve for multiple channels need to take maximum magnitude across all color channels 
## TO DO: Make it work for more than one image 
def saliency(model, image): 
    image.requires_grad_()
    logits = model(image)
    max_score_index = logits.argmax(dim=1)
    max_score = logits[0,max_score_index]
    
    max_score.backward()
    
    saliency, _ = torch.max(image.grad.data.abs(), dim=1)
    return saliency 

def visualizeMap(image, saliency_map, path_model_folder, i=0):
    
    fig, (ax1, ax2) = plt.subplots(1,2)
    fig.suptitle("Image and its Saliency Map next to each other")
    ax1.imshow(image)
    ax1.axis('off')
    ax2.imshow(saliency_map, cmap='viridis')
    ax2.axis('off')
    
    plt.savefig(os.path.join(path_model_folder, f"saliency_map_img_{i}"))

    return None 

def main_saliency(model, dataloader, device, path_model_folder, num_images=1):
    model.eval()

    # Get one image from dataloader
    data_iter = iter(dataloader)
    images, labels, _ = next(data_iter)
    image = images[0:num_images].to(device)
    
    ### normalize tensor images using the mean and standart deviation of the dataset 
    
    saliency_val = saliency(model, image)    
    
    # Move image and saliency map to CPU and convert to numpy
    image_np = image.cpu().squeeze().permute(1, 2, 0).detach().numpy()
    saliency_np = saliency_val.cpu().squeeze().numpy()
    
    visualizeMap(image_np, saliency_np, path_model_folder)   
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_cka(vit_path= "20260127_114014", cnn_path= "20260124_154632", batch_size =64, device=device)
