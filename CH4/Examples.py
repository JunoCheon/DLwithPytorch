#%%
import imageio  
import torch
#%%
############################################
##                Exercise.1              ##
############################################
# 1. Load the image from the file and convert it to a tensor.
# 2. Extract the red, green, and blue mean values for the image.
# 3. Can you distinguish? What are the mean values for the red, green, and blue channels?


img  = imageio.imread("..\data\p1ch4\image-cats\cat3.png")
img = torch.from_numpy(img).float()
#shape : 256, 256, 3

# HWC -> CHW , shape : 3, 256, 256
img = img.permute(2,0,1)
# %%
torch.mean(img, dim=(1,2)) #R,G,B means
# %%
############################################
##                Exercise.2              ##
############################################
