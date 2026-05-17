import torch
import torchvision.models as models 

def create_imageNet_preTrained_vit(num_classes: int):
    """
    Create pretrained ViT to run petfaces dataset. 
    """ 
    weights = models.ViT_B_16_Weights.IMAGENET1K_V1
    model = models.vit_b_16(weights=weights)
    model.heads = torch.nn.Linear(model.heads.head.in_features, num_classes)
    
    for params in model.parameters(): 
        params.requires_grad = False
        
    for params in model.heads.parameters(): 
        params.requires_grad = True 
    return model
    
class TransfomerEncoder(torch.nn.Module): 
    
    def __init__(self, num_heads, token_dim, embedding_dim, hidden_fact = 4, dropout = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.hidden_fact = hidden_fact
        self.dropout = dropout
        
        assert self.embedding_dim % self.num_heads == 0, f"Embedding dimension {self.embedding_dim} can't be devided through number of heads {self.num_heads}."
        self.head_dim = token_dim // num_heads
        
        self.layer_norm_1 = torch.nn.LayerNorm(self.embedding_dim)
        self.layer_norm_2 = torch.nn.LayerNorm(self.embedding_dim)
        self.attention = torch.nn.MultiheadAttention(self.embedding_dim, self.num_heads, self.dropout, batch_first=True)
        self.mlp_block = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_dim, self.embedding_dim * self.hidden_fact), 
            torch.nn.GELU(),
            torch.nn.Linear(self.embedding_dim * self.hidden_fact, self.embedding_dim), 
            torch.nn.Dropout(self.dropout)
        )
         
        
    def forward(self, tokens): 
        normed_tokens = self.layer_norm_1(tokens)
        attention_tokens, _ = self.attention(normed_tokens, normed_tokens, normed_tokens)
        tokens = tokens + attention_tokens
        normed_tokens = self.layer_norm_2(tokens)
        mlp_block_tokens = self.mlp_block(normed_tokens)
        tokens = tokens + mlp_block_tokens
        return tokens 
    
    
class Transformer(torch.nn.Module): 
    
    def __init__(self, 
                 images_size,
                 batch_size, 
                 patches_size,
                 embedding_dim, 
                 num_encoder_blocks, 
                 num_attention_heads, 
                 factor_hidden_size_encoder,
                 dropout_en: 0.0, 
                 num_classes):
        super().__init__()
        print(type(images_size))
        self.images_size = images_size
        self.patches_size = patches_size
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.num_encoder_blocks = num_encoder_blocks
        self.num_attention_heads = num_attention_heads
        self.factor_hidden_size_encoder = factor_hidden_size_encoder
        self.dropout_en = dropout_en
        self.num_classes = num_classes
        
        self.n_patches = images_size[0] * images_size[1] // (self.patches_size)**2 
        self.token_dim = images_size[2]  * self.patches_size * self.patches_size
        
        self.class_token = torch.nn.Parameter(torch.randn(1, 1, self.embedding_dim))
        self.positional_embedding = torch.nn.Parameter(torch.zeros(1,self.n_patches +1, self.embedding_dim))
        
        self.linear_proj = torch.nn.Linear(self.token_dim, self.embedding_dim)
        
        self.encoder_blocks = torch.nn.Sequential(
            *(TransfomerEncoder(num_heads=self.num_attention_heads,
                                               token_dim=self.token_dim, 
                                               embedding_dim=self.embedding_dim,
                                               hidden_fact=self.factor_hidden_size_encoder,
                                               dropout=self.dropout_en) for _ in range(self.num_encoder_blocks)))
        ### this is very variable --> adding a layer norm before linear, leaving the relu? 
        self.mlp_head = torch.nn.Sequential(torch.nn.LayerNorm(self.embedding_dim),
            torch.nn.Linear(self.embedding_dim, self.num_classes)
        )
        
        
        
        
    def patched(self, images):
        ### inputs have size (BS, C, H, W) with (H,W) being the resolution and C the number of channels 
        b, c, h, w = images.shape
        p = self.patches_size
        
        assert h % p == 0, "Input hight does not fit to chosen patch size."
        assert w % p == 0, "Input width does not fit to chosen patch size."
        
        n_patches = self.n_patches
        
        ### want to transform it into N * (P,P,C) with (P,P) being the resolution of each patch and C the number of channels 
        patches = images.unfold(2, p,p)
        patches = patches.unfold(3,p,p)
        ## transform (B, C, p_h, p_w, p, p) in shape (B, N, C, P, P,) 
        patches = patches.permute(0,2,3,1,4,5)
        ## flatten to (N, P*P*C)
        patches = patches.reshape(b,n_patches,c*p*p)
        
        return patches 

    
    def forward(self, images): 
        patches = self.patched(images)
        tokens = self.linear_proj(patches)
        
        ### add class tokens and position embeddings
        b = tokens.shape[0]
        class_token =  self.class_token.expand(b,-1,-1)
        tokens = torch.cat([class_token, tokens], dim=1)
        tokens = tokens + self.positional_embedding
        
        ### feed tokens into Encoder blocks 
        tokens = self.encoder_blocks(tokens)
        
        ### feed class token to mlp block for classification 
        out = self.mlp_head(tokens[:,0,:])
        return out 
        


