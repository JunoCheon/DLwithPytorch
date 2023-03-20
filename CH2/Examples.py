#%%
#2.1.1
import torch
import torch.nn  as nn
import torch.nn.functional  as F

from torchvision import models

# %%
#2.1.2 AlexNet
alexnet = models.AlexNet()

#%%
#2.1.3
resnet = models.resnet101(pretrained = True)

#%%
#2.1.4

from torchvision import transforms
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.6848, 0.5311, 0.3384],
    std=[0.2027, 0.1943, 0.1984]
    )])

# %%
from PIL import Image

# %%
img = Image.open("./data/p1ch2/bobby.jpg")

# %%
img
# img.show()

#%%
img_t = preprocess(img)

# %%
batch_t = torch.unsqueeze(img_t,0)

# %%
#2.1.5
resnet.eval()

# %%
out = resnet(batch_t)
out

# %%
with open("./data/p1ch2/imagenet_classes.txt") as f:
    labels = [line.strip() for line in f.readlines()]

# %%
_, index = torch.max(out,1)

# %%
precentage = F.softmax(out,dim = 1)[0]*100
labels[index[0]], precentage[index[0]].item()

#%%
_,indices = torch.sort(out, descending= True)
[(labels[idx],precentage[idx].item()) for idx in indices[0][:5]]

#%%
#2.2
class ResNetBlock(nn.Module): # <1>

    def __init__(self, dim):
        super(ResNetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim)

    def build_conv_block(self, dim):
        conv_block = []

        conv_block += [nn.ReflectionPad2d(1)]

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
                       nn.InstanceNorm2d(dim),
                       nn.ReLU(True)]

        conv_block += [nn.ReflectionPad2d(1)]

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
                       nn.InstanceNorm2d(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x) # <2>
        return out
    

class ResNetGenerator(nn.Module):

    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=9): # <3> 

        assert(n_blocks >= 0)
        super(ResNetGenerator, self).__init__()

        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=True),
                 nn.InstanceNorm2d(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=True),
                      nn.InstanceNorm2d(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResNetBlock(ngf * mult)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=True),
                      nn.InstanceNorm2d(int(ngf * mult / 2)),
                      nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input): # <3>
        return self.model(input)
    
#%%
netG = ResNetGenerator()

# %%
model_path = './data/p1ch2/horse2zebra_0.4.0.pth'
model_data = torch.load(model_path)
netG.load_state_dict(model_data)

# %%
netG.eval()

# %%
preprocess = transforms.Compose(
    [transforms.Resize(256),
     transforms.ToTensor()]
)

# %%
img = Image.open("./data/p1ch2/horse.jpg")
img

# %%
img_t = preprocess(img)
batch_t = torch.unsqueeze(img_t,0)

# %%
batch_out = netG(batch_t)

# %%
out_t = (batch_out.data.squeeze()+1.0)/2.0
out_img = transforms.ToPILImage()(out_t)
out_img

# %%
#2.3-4
resnet18_model = torch.hub.load('pytorch/vision',
                                'resnet18',
                                pretrained = True)
# %%
resnet18_model()