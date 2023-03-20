#!/usr/bin/env python
# coding: utf-8

# In[1]:


a = [1.0, 2.0, 1.0]


# In[2]:


a[0]


# In[3]:


a[2] = 3.0
a


# In[4]:


import torch # <1>
a = torch.ones(3) # <2>
a


# In[5]:


a[1]


# In[6]:


float(a[1])


# In[7]:


a[2] = 2.0
a


# In[8]:


points = torch.zeros(6) # <1>
points[0] = 4.0 # <2>
points[1] = 1.0
points[2] = 5.0
points[3] = 3.0
points[4] = 2.0
points[5] = 1.0


# In[9]:


points = torch.tensor([4.0, 1.0, 5.0, 3.0, 2.0, 1.0])
points


# In[10]:


float(points[0]), float(points[1])


# In[11]:


points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points


# In[12]:

points.shape


# In[13]:


points = torch.zeros(3, 2)
points


# In[14]:


points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points


# In[15]:


points[0, 1]


# In[16]:


points[0]


# In[17]:


points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points.storage()


# In[18]:


points_storage = points.storage()
points_storage[0]


# In[19]:


points.storage()[1]


# In[20]:


points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points_storage = points.storage()
points_storage[0] = 2.0
points


# In[21]:


points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
second_point = points[1]
second_point.storage_offset()


# In[22]:


second_point.size()


# In[23]:


second_point.shape


# In[24]:


points.stride()


# In[25]:


second_point = points[1]
second_point.size()


# In[26]:


second_point.storage_offset()


# In[27]:


second_point.stride()


# In[28]:


points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
second_point = points[1]
second_point[0] = 10.0
points


# In[29]:


points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
second_point = points[1].clone()
second_point[0] = 10.0
points


# In[30]:


points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points


# In[31]:


points_t = points.t()
points_t


# In[32]:

#only in data_ptr
id(points.storage().data_ptr()) == id(points_t.storage().data_ptr())


# In[33]:


points.stride()


# In[34]:


points_t.stride()


# In[35]:


some_t = torch.ones(3, 4, 5)
transpose_t = some_t.transpose(0, 2)
some_t.shape


# In[36]:


transpose_t.shape


# In[37]:


some_t.stride()


# In[38]:


transpose_t.stride()


# In[39]:


points.is_contiguous()


# In[40]:


points_t.is_contiguous()


# In[41]:


points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points_t = points.t()
points_t


# In[42]:


points_t.storage()


# In[43]:


points_t.stride()


# In[44]:


points_t_cont = points_t.contiguous()
points_t_cont


# In[45]:


points_t_cont.stride()


# In[46]:


points_t_cont.storage()


# In[47]:


double_points = torch.ones(10, 2, dtype=torch.double)
short_points = torch.tensor([[1, 2], [3, 4]], dtype=torch.short)


# In[48]:


short_points.dtype


# In[49]:


double_points = torch.zeros(10, 2).double()
short_points = torch.ones(10, 2).short()


# In[50]:


double_points = torch.zeros(10, 2).to(torch.double)
short_points = torch.ones(10, 2).to(dtype=torch.short)


# In[51]:


points_64 = torch.rand(5, dtype=torch.double)  # <1>
points_short = points_64.to(torch.short)
points_64 * points_short  # works from PyTorch 1.3 onwards


# In[52]:


# reset points back to original value
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])


# In[53]:


some_list = list(range(6))
some_list[:]     # <1>
some_list[1:4]   # <2>
some_list[1:]    # <3>
some_list[:4]    # <4>
some_list[:-1]   # <5>
some_list[1:4:2] # <6>


# In[54]:


points[1:]       # <1>
points[1:, :]    # <2>
points[1:, 0]    # <3>
points[None]     # <4>


# In[55]:


points = torch.ones(3, 4)
points_np = points.numpy()
points_np


# In[56]:


points = torch.from_numpy(points_np)


# In[57]:


torch.save(points, '../data/p1ch3/ourpoints.t')


# In[58]:


with open('../data/p1ch3/ourpoints.t','wb') as f:
   torch.save(points, f)


# In[59]:


points = torch.load('../data/p1ch3/ourpoints.t')


# In[60]:


with open('../data/p1ch3/ourpoints.t','rb') as f:
   points = torch.load(f)


# In[61]:


import h5py

f = h5py.File('../data/p1ch3/ourpoints.hdf5', 'w')
dset = f.create_dataset('coords', data=points.numpy())
f.close()


# In[62]:


f = h5py.File('../data/p1ch3/ourpoints.hdf5', 'r')
dset = f['coords']
last_points = dset[-2:]


# In[63]:


last_points = torch.from_numpy(dset[-2:])
f.close()


# In[64]:


points_gpu = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]], device='cuda')


# In[65]:


points_gpu = points.to(device='cuda')


# In[66]:


points_gpu = points.to(device='cuda:0')


# In[67]:


points = 2 * points  # <1>
points_gpu = 2 * points.to(device='cuda')  # <2>


# In[68]:


points_gpu = points_gpu + 4


# In[69]:


points_cpu = points_gpu.to(device='cpu')


# In[70]:


points_gpu = points.cuda()  # <1>
points_gpu = points.cuda(0)
points_cpu = points_gpu.cpu()


# In[71]:


a = torch.ones(3, 2)
a_t = torch.transpose(a, 0, 1)

a.shape, a_t.shape


# In[72]:


a = torch.ones(3, 2)
a_t = a.transpose(0, 1)

a.shape, a_t.shape


# In[73]:


a = torch.ones(3, 2)


# In[74]:


a.zero_()
a

