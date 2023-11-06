import torch
from ViT import *

v = ViT(
    image_size = 64,
    patch_size = 16,
    num_classes = 200,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

img = torch.randn(10, 3, 64, 64)

preds = v(img) # (1, 1000)

print(preds.shape)