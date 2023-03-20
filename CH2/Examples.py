#%%
#2.1.1
import torch

from torchvision import models
# %%
#2.1.2 AlexNet
alexnet = models.AlexNet()

#%%
#2.1.3

resnet = models.resnet101()

resnet

from torchvision import transforms
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485,0.456,0.406],
    std=[0.229,0.224,0.225]
    )
])


# %%
from PIL import Image
# %%
img = Image.open(".../data/p1ch2/bobby.jpg")
# %%
