# %% [markdown]
# ## Dataloader

# %%
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
import re
import os
from pathlib import Path
import numpy as np
import random
# class裡面的括號表示繼承torch dataset的東西
class MusicDataset(Dataset):
    def __init__(self, audio_files, labels, sample_rate=22050, n_fft=2048, hop_length=512):
        self.audio_files = audio_files
        self.labels = labels
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        length = 130
        audio_file = self.audio_files[idx]
        label = self.labels[idx]
        audio, _ = librosa.load(audio_file, sr=self.sample_rate)
        
        # 新加入
        return_data = []
        n_segments = 10
        n_mfcc = 40
        samples_per_segment = int (self.sample_rate*30/n_segments)
        
        for n in range (n_segments):
            mfcc = librosa.feature.mfcc(y=audio[samples_per_segment*n:samples_per_segment*(n+1)], sr=self.sample_rate, n_mfcc=n_mfcc, n_fft=self.n_fft, hop_length=self.hop_length)
            # mfcc = librosa.feature.mfcc(audio[samples_per_segment*n:samples_per_segment*(n+1)],sr=self.sample_rate, n_mfcc=n_mfcc, n_fft=self.n_fft,hop_length=self.hop_length)
            mfcc = mfcc.T
            mfcc = torch.FloatTensor(mfcc)
            # print(mfcc.shape)
            if mfcc.shape[0] < length:
                pad_width = length - mfcc.shape[0]
                mfcc =  torch.from_numpy(np.pad(mfcc, ((0, pad_width), (0, 0)), 'constant'))
            elif mfcc.shape[0] > length:
                mfcc = mfcc[:, :length]
            return_data.append(mfcc)
        # return_data = torch.cuda.FloatTensor(return_data)
        label = torch.LongTensor([label])
        return return_data, label
    
        # 原本的
        # stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        # magnitude, phase = librosa.magphase(stft)
        # # print(stft.shape)
        # magnitude = torch.FloatTensor(magnitude)
        # phase = torch.FloatTensor(phase)
        # if magnitude.shape[1] < length:
        #     pad_width = length - magnitude.shape[1]
        #     magnitude =  torch.from_numpy(np.pad(magnitude, ((0, 0), (0, pad_width)), 'constant'))
        #     phase = torch.from_numpy(np.pad(phase, ((0, 0), (0, pad_width)), 'constant'))
        # elif magnitude.shape[1] > length:
        #     magnitude = magnitude[:, :length]
        #     phase = phase[:, :length]
        # label = torch.LongTensor([label])
        # # print("call get item",audio_file)
        # # print(magnitude.shape)
        # # print(phase.shape)
        # return magnitude, phase, label
        
        

# %% [markdown]
# ### Test load data

# %%
# data_dir = Path('/home/fanal/disk2/luo/genre_classification/genre34/country')
# file_list = os.listdir(data_dir)

# index = 3
# file_path = data_dir / file_list[index]
# print(file_path)
# print(file_list[index])
# if (file_list[index]=='.DS_Store'):
#     print("error")
# else:
#     label = re.findall('^[a-z]+', file_path.name)[0]  # 提取標籤
#     number = re.findall('\d+', file_path.name)[0]  # 提取編號
#     print(label)
#     print(number)

# %% [markdown]
# ## CNN model

# %%
import torch.nn as nn
import torch.optim as optim

class MusicCNN(nn.Module):
    def __init__(self, n_classes):
        super(MusicCNN, self).__init__()
        self.conv1 = nn.Conv2d(10, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.fc1 = nn.Linear(64 * 32 * 10, 128)
        # self.fc1 = nn.Linear(64 * 256 * 323, 128)

        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1, x.size(2), x.size(3)) # [batch_size, 10, height, width] -> [batch_size, 10*1, height, width]
        # x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        # print("before view",x.shape)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x


# %% [markdown]
# ### load data

# %%
audio_files = []
labels = []
postfix = ['12','34','56','78','910']
for i in range(5):
    music_folder = '/home/fanal/disk2/luo/genre_classification/genre'
    music_folder += postfix[i]
    genres = os.listdir(music_folder)        
    for genre in genres:
        if(genre !='.DS_Store'):
            genre_folder = os.path.join(music_folder, genre)
            genre_files = os.listdir(genre_folder)
            for file in genre_files:
                if file.endswith('.wav'):
                    file_path = os.path.join(genre_folder, file)
                    label = re.findall('^[a-z]+', file)[0]
                    if (label=="blues"):
                        label = 0
                    elif (label=="classical"):
                        label = 1
                    elif (label=="country"):
                        label = 2    
                    elif (label=="disco"):
                        label = 3
                    elif (label=="hiphop"):
                        label = 4 
                    elif (label=="jazz"):
                        label = 5 
                    elif (label=="metal"):
                        label = 6 
                    elif (label=="pop"):
                        label = 7
                    elif (label=="reggae"):
                        label = 8 
                    elif (label=="rock"):
                        label = 9  
                    audio_files.append(file_path)
                    labels.append(label)




dataset = MusicDataset(audio_files, labels)
train_size = int(len(dataset) * 0.8)
train_set, test_set = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])

dataloader = DataLoader(train_set, batch_size=32, shuffle=True)




# %% [markdown]
# ### torch summary
# 

# %%
from torchsummary import summary
n_classes = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device",device)
model = MusicCNN(n_classes)
model.to(device)
optimizer = optim.Adam(model.parameters())
loss_function = nn.CrossEntropyLoss()

# summary(model, input_size=(1025,1293), batch_size=32)
summary(model, input_size=(10,130,40), batch_size=32)



# %% [markdown]
# ## Train

# %%
# def batch_change(data,label):
#     print(label)
#     label = label.repeat(1, 10)
#     label = label.view(1, -1)
#     return label

n_epochs = 150
for epoch in range(n_epochs):
    # for i, (magnitude, phase, label) in enumerate(dataloader):
    for i, (magnitude, label) in enumerate(dataloader):
        # print("label",label)
        # print(i)
        optimizer.zero_grad()
        magnitude = torch.stack(magnitude)
        magnitude = magnitude.permute(1, 0, 2, 3)
        # size = np.array(magnitude[0]).shape
        # print("magshape",magnitude.shape)
        # new_label = batch_change(magnitude,label)
        # print(device)
        magnitude = magnitude.type(torch.cuda.FloatTensor)
        magnitude.to(device)
        label = label.type(torch.cuda.LongTensor)
        label.to(device)
        # print("mag",magnitude.device)
        output = model(magnitude)
        # print('modelover')
        loss = loss_function(output, label.squeeze())
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print("Epoch {} Batch {}: Loss = {}".format(epoch, i, loss.item()))

# %% [markdown]
# ## save model

# %%
torch.save(model.state_dict(), 'model_100ep.pt')

# %%

def evaluate(model, dataloader):
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for data, labels in dataloader:
            data= torch.stack(data)
            data = data.permute(1, 0, 2, 3)
            data = data.type(torch.cuda.FloatTensor)
            data.to(device)
            labels = labels.type(torch.cuda.LongTensor)
            labels.to(device)

            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            labels = labels.squeeze()
            # print('pred',predicted.shape)
            # print('label',labels.shape)
            # print((predicted == labels).sum())
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()
    # print(total_samples)
    accuracy = 100.0 * total_correct / total_samples
    return accuracy
test_loader = DataLoader(test_set, batch_size=32, shuffle=True)

accuracy = evaluate(model, test_loader)
print(f"Test accuracy: {accuracy:.2f}%")



