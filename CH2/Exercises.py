#%%
import torch
import torch.nn  as nn
import torch.nn.functional  as F

from torchvision import models
from torchvision import transforms

from PIL import Image

#Exercise 1
#load bobby.jpg
img = Image.open("../data/p1ch2/bobby.jpg")

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.6848, 0.5311, 0.3384],
    std=[0.2027, 0.1943, 0.1984]
    )]
)


img_t = preprocess(img)
batch_t = torch.unsqueeze(img_t,0)


#load classes
with open("../data/p1ch2/imagenet_classes.txt") as f:
    labels = [line.strip() for line in f.readlines()]
# %%
from Examples import ResNetGenerator

netG = ResNetGenerator()

batch_out = netG(batch_t)

out_t = (batch_out.data.squeeze()+1.0)/2.0
out_img = transforms.ToPILImage()(out_t)
out_img

# %%
#Exercise 2
