# %%
import imageio
import torch

# %%
img_arr = imageio.imread('../data/p1ch4/image-dog/bobby.jpg')
img_arr.shape

# %%
img = torch.from_numpy(img_arr)
out = img.permute(2,0,1)
out.shape

# %%
batch_size = 3
batch = torch.zeros(batch_size,3,256,256,dtype=torch.uint8)

# %%
import os

data_dir = '../data/p1ch4/image-cats/'
fielnames = [name for name in os.listdir(data_dir) 
             if os.path.splitext(name)[-1] == '.png']
for i, filename in enumerate(fielnames):
    img_arr = imageio.imread(data_dir+filename)
    img_t = torch.from_numpy(img_arr)
    img_t = img_t.permute(2,0,1)
    img_t = img_t[:3]
    batch[i] = img_t
# %%
batch = batch.float()
batch /= 255.0
#%%
n_channels = batch.shape[1]
for c in range(n_channels):
    mean = torch.mean(batch[:,c])
    std = torch.std(batch[:,c])
    batch[:,c] = (batch[:,c] - mean) / std
# %%
batch
# %%
#4.2-4.3

dir_path = "../data/p1ch4/volumetric-dicom/2-LUNG 3.0  B70f-04083/"
vol_arr = imageio.volread(dir_path, 'DICOM')
vol_arr.shape
# %%
vol = torch.from_numpy(vol_arr).float()
# create 1 dimension in 0 index
vol = torch.unsqueeze(vol,0)
vol.shape
# %%
import csv
import numpy as np
wine_path = "../data/p1ch4/tabular-wine/winequality-white.csv"
wineq_numpy = np.loadtxt(wine_path, dtype=np.float32, delimiter=";", skiprows=1)

wineq_numpy
# %%
col_list = next(csv.reader(open(wine_path), delimiter=";"))
# %%
wineq = torch.from_numpy(wineq_numpy)

wineq.shape, wineq.dtype
# %%
data = wineq[:,:-1]
target = wineq[:,-1]
data,data.shape
#%%
target,target.shape
# %%
target = target.long()
# %%
target_onehot = torch.zeros(target.shape[0],10)
# underbar(_)는 inplace operation을 의미한다.
# scatter_ dim: 0 or 1, index: 4899*1 tensor, 넣을값
# https://hongl.tistory.com/201

target_onehot.scatter_(1,target.unsqueeze(1),1.0)
# %%
data_mean = torch.mean(data,dim=0)
data_var = torch.var(data,dim=0)
data_normalized = (data - data_mean) / torch.sqrt(data_var)
data_normalized

# %%
bad_indices = target <=3
bad_indices,bad_indices.dtype,bad_indices.sum()

# %%
bad_data = data[bad_indices]
bad_data.shape

# %%
bad_data = data[target <= 3]
mid_data = data[(target > 3) & (target < 7)]
good_data = data[target >= 7]

bad_mean = torch.mean(bad_data,dim=0)
mid_mean = torch.mean(mid_data,dim=0)
good_mean = torch.mean(good_data,dim=0)

print("{:2} {:20} {:>8} {:>8} {:>8}".format("", "feature", "bad_mean", "mid_mean", "good_mean"))
for i ,arg in enumerate(zip(col_list, bad_mean, mid_mean, good_mean)):
    print("{:2d} {:20} {:>8.2f} {:>8.2f} {:>8.2f}".format(i, *arg))

# %%
total_sulfur_threshold = 141.83
total_sulfur_data = data[:,6]
predicted_indexes = torch.less(total_sulfur_data, total_sulfur_threshold)

#torch.less = torch.lt
# %%
predicted_indexes.shape, predicted_indexes.dtype, predicted_indexes.sum()
# %%
actual_indices = target > 5

actual_indices.shape, actual_indices.dtype, actual_indices.sum()
# %%
n_matches = torch.sum(actual_indices & predicted_indexes).item()
n_predicted = torch.sum(predicted_indexes).item()
n_actual = torch.sum(actual_indices).item()

print(n_matches,n_matches / n_predicted,n_matches/ n_actual)
# %%
############################################
####                                    ####
####          4.4 time series           ####
####                                    ####
############################################

bikes_np = np.loadtxt("../data/p1ch4/bike-sharing-dataset/hour-fixed.csv",
                        dtype=np.float32,
                        delimiter=",",
                        skiprows=1,
                        converters={1: lambda x: float(x[8:10])})
bikes = torch.from_numpy(bikes_np)
bikes.shape
# %%
bikes.stride()
# %%
daily_bikes = bikes.view(-1,24,bikes.shape[1])
daily_bikes.shape, daily_bikes.stride()
# (365*2) N * 24 L? * 17 Channel
# %%
daily_bikes = daily_bikes.permute(0,2,1)
daily_bikes.shape, daily_bikes.stride()
#permute(0,2,1) = transpose(1,2)
daily_bikes.shape, daily_bikes.stride()
# %%
first_day = bikes[:24].long()
weather_onehot = torch.zeros(first_day.shape[0],4)

weather_onehot.scatter_(1,first_day[:,9].unsqueeze(1).long()-1,1.0)
# %%
torch.cat((bikes[:24],weather_onehot),dim=1)[:1]
# %%
daily_weather_onehot = torch.zeros(daily_bikes.shape[0],4,daily_bikes.shape[2])
daily_weather_onehot.scatter_(1,daily_bikes[:,9,:].unsqueeze(2).long()-1,1.0)

daily_bikes = torch.cat((daily_bikes,daily_weather_onehot),dim = 1)
# %%
# min-max normalization
temp = daily_bikes[:,10,:]
temp_,min = torch.min(temp,dim=0)
temp_,max = torch.max(temp,dim=0)
daily_bikes[:,10,:] = (temp - temp_.min()) / (temp_.max() - temp_.min())
# %%
# z-score normalization
temp = daily_bikes[:,10,:]
temp_mean = torch.mean(temp,dim=0)
temp_std = torch.std(temp,dim=0)
daily_bikes[:,10,:] = (temp - temp_mean) / temp_std

# %%
############################################
####                                    ####
####          4.5-4.6 text              ####
####                                    ####
############################################

with open("../data/p1ch4/jane-austen/1342-0.txt","r",encoding="utf8") as f:
    text = f.read()
# %%
lines = text.split("\n")
line = lines[200]
line
# %%
letter_t = torch.zeros(len(line),128)
letter_t.shape
# %%
for i,letter in enumerate(line.lower().strip()):
    letter_index = ord(letter) if ord(letter) < 128 else 0
    letter_t[i][letter_index] = 1

# letter_t[i][ord(letter)] = 1
#strip : 문자열의 양쪽 공백을 제거
#ord : 문자의 아스키 코드값을 돌려주는 함수
# %%
import string
def clean_words(input_str):
    # punctuation = string.punctuation +"“" + "”"
    punctuation = '.,;:“”!?"_-'
    wordlist = input_str.lower().replace('\n',' ').split()
    wordlist = [word.strip(punctuation) for word in wordlist]
    return wordlist

# split : 문자열을 공백을 기준으로 분리하여 리스트로 반환
# string.punctuation : !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
# strip(punctuation) : 문자열의 양쪽에서 punctuation에 포함된 문자를 제거
# %%
words_in_line = clean_words(line)
line, words_in_line
# %%
wordlist = sorted(set(clean_words(text)))
word2index_dict = {word: i for (i,word) in enumerate(wordlist)}
len(word2index_dict), word2index_dict['impossible']
# %%

word_t = torch.zeros(len(words_in_line),len(word2index_dict))
for i,word in enumerate(words_in_line):
    word_index = word2index_dict[word]
    word_t[i][word_index] = 1
    print('{:2} {:4} {}'.format(i,word_index,word))

