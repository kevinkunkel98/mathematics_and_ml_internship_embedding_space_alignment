from architectures.cnn import CNNModel, create_resnet
from architectures.vit import Transformer, create_imageNet_preTrained_vit

def build_model(config, num_classes):
    model_architecture = config["model"]["architecture"]

    if model_architecture == "custom_cnn":
        return CNNModel(
            input_channels=config["data"]["input_channels"],
            num_classes=num_classes
        )

    elif model_architecture == "resnet18":
        return create_resnet(num_classes=num_classes)

    elif model_architecture == "custom_vit":
        return Transformer(
            images_size=config["data"]["images_size"], 
            batch_size=config["training"]["batch_size"], 
            patches_size=config["model"]["patches_size"], 
            embedding_dim=config["model"]["embedding_dim"],
            num_encoder_blocks=config["model"]["num_encoder_blocks"],
            num_attention_heads=config["model"]["num_attention_heads"],
            factor_hidden_size_encoder=config["model"]["factor_hidden_size_encoder"],
            dropout_en=config["model"]["dropout_en"],
            num_classes=num_classes,
        )
    elif model_architecture == "imagenet_pretrained_vit":
        return create_imageNet_preTrained_vit(num_classes=num_classes)

    else:
        raise ValueError(f"Unknown model name: {model_architecture}")
