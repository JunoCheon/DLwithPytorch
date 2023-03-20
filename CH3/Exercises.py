#%%
import torch

#1
a = torch.tensor(list(range(9)))
# b = torch.tensor(a).reshape(3,3)
b = a.view(3,3)
b
#%%
b.stride()
# %%
b.storage()
# %%
b.storage_offset()


# %%
c = b[1:,1:]
c
#%%
c.stride()
#%%
c.storage()
#%%
c.storage_offset()
# %%


#2
#Cosine
torch.cos(a)
#Sqrt
torch.sqrt(a)
