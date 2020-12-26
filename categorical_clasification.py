#1.Module

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pathlib
import seaborn as sns
sns.set(style="white")

#pytorch
import torch
from torch import nn
from torch import functional as F
from torch import optim
from torch.utils.data import Dataset,DataLoader,random_split
from torchvision import transforms
from torch.nn import Module
from torchvision import models
from PIL import Image

#pytorch lightning
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy


#kagleでofflineでmodelを使用する場合はpathで追加する必要あり！！！！

#sys
import sys
# efficientnet with internet_off
sys.path.append("../input/efficientnet-pytorch/EfficientNet-PyTorch/EfficientNet-PyTorch-master")

#2.Path

#欠損値
print(df.isnull().sum())
#ラベルデータの確認
print(df.shape)
#ラベル数の分布
print(df["label"].value_counts())

#plot
df["label"].value_counts().plot(kind="bar")
plt.show()


#load sample_image
def load_image(img_id):
    #data_path
    img_path =pathlib.Path("../input/cassava-leaf-disease-classification/train_images")
    #open cv2
    img_bgr=cv2.imread(str(img_path/img_id))  #BGR
    img_rgb=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb

#EDA

#Recognition Color
SAMPLE_LEN=100
train_images=[]
for i in range(len(df.iloc[:SAMPLE_LEN])):
    loaded=load_image(df["image_id"][i])
    train_images.append(loaded)
else:
    train_images=np.array(train_images)

red_values = [np.mean(train_images[idx][:, :, 0]) for idx in range(len(train_images))]
green_values = [np.mean(train_images[idx][:, :, 1]) for idx in range(len(train_images))]
blue_values = [np.mean(train_images[idx][:, :, 2]) for idx in range(len(train_images))]
values = [np.mean(train_images[idx]) for idx in range(len(train_images))]

import seaborn as sns
sns.distplot(values,color="grey")
#上記で出力は可能
plt.show()

#red channels values
sns.distplot(red_values,color="red")
#上記で出力は可能
plt.show()

#Green channels values
sns.distplot(green_values,color="green")
plt.show()

#Blue Channels values
sns.distplot(blue_values,color="blue")
plt.show()

#box plot
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.boxplot([red_values,green_values,blue_values],labels=["R","G","B"])
plt.grid("true")
plt.show()

#---------------------------

#Split Data
from sklearn.model_selection import train_test_split
test_size=0.2
df_train,df_val=train_test_split(df,test_size=test_size,
                                 stratify=df["label"],
                                random_state=0)

print("DataFrame Shape")
print("train:",df_train.shape)
print("test:",df_val.shape)
print("---------------")
print(df_train.label.value_counts())
print(df_val.label.value_counts())

#DataSet and DataLoader

#pytorch
import torch
from torch import nn
from torch import functional as F
from torch import optim
from torch.utils.data import Dataset,DataLoader,random_split
from torchvision import transforms
from torch.nn import Module
from torchvision import models
from PIL import Image

#pytorch lightning
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy

#EfficientNet
#import efficientnet_pytorch

# Transforms with Albumentation¶

import albumentations as A

#Albumentations基本的な使い方

#BGR読み込み
image_bgr = cv2.imread\
            ('../input/cassava-leaf-disease-classification/train_images/1000723321.jpg', 1) 
img_rgb=cv2.cvtColor(image_bgr,cv2.COLOR_BGR2RGB)



#albumentationsで出力を調整する場合。
# Declare an augmentation pipeline
transform = A.Compose([
    A.RandomCrop(width=256, height=256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
])

#出力確認
val = transform(image=img_rgb)
#定義したtransformだけではなく、["image"]を適用することが必要
img1=val["image"]
print(img1.shape)
plt.imshow(img1)
plt.show()



import albumentations

out_size=224

# augmentations taken from: https://www.kaggle.com/khyeh0719/pytorch-efficientnet-baseline-train-amp-aug
train_aug = albumentations.Compose([
            albumentations.RandomResizedCrop(out_size,out_size),
            albumentations.Transpose(p=0.5),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.ShiftScaleRotate(p=0.5),
            albumentations.HueSaturationValue(
                hue_shift_limit=0.2, 
                sat_shift_limit=0.2, 
                val_shift_limit=0.2, 
                p=0.5
            ),
            albumentations.RandomBrightnessContrast(
                brightness_limit=(-0.1,0.1), 
                contrast_limit=(-0.1, 0.1), 
                p=0.5
            ),
            albumentations.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225], 
                max_pixel_value=255.0, 
                p=1.0
            ),
            albumentations.CoarseDropout(p=0.5),
            albumentations.Cutout(p=0.5)], p=1.)
  
        
valid_aug = albumentations.Compose([
            albumentations.CenterCrop(256, 256, p=1.),
            albumentations.Resize(out_size,out_size),
            albumentations.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225], 
                max_pixel_value=255.0, 
                p=1.0
            )], p=1.)
            
class My_Dataset(Dataset):
    def __init__(self,df,transform):
        #dataframeを格納
        self.df = df
        self.img_path =pathlib.Path("../input/cassava-leaf-disease-classification/train_images")
        
        #transform
        self.transform=transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,index):
        #image_id
        img_id=self.df.iloc[index,0]
        
        #open cv2
        img_bgr=cv2.imread(str(self.img_path/img_id))  #BGR
        img_rgb=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        #transform with albumentations
        if self.transform is not None:
            img_transformed=self.transform(image=img_rgb)["image"]
        
        #change channels first
        img=np.einsum('ijk->kij', img_transformed)
        
        #label
        target=self.df.iloc[index,1]
        #target
        
        return img,target
        
#DataSet
train_dataset=My_Dataset(df_train,transform=train_aug)
val_dataset=My_Dataset(df_val,transform=valid_aug)

#Dataloader
batch_size=10
train_dataloader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
val_dataloader=DataLoader(val_dataset,batch_size=batch_size,shuffle=False)

#dict
dataloaders_dict={"train":train_dataloader,"val":val_dataloader}

#test sample
batch_iterator=iter(dataloaders_dict["train"])
inputs,labels=next(batch_iterator)
print(inputs.shape) 
print(labels.shape)
print(labels)

#torch.Size([10, 3, 224, 224])
#torch.Size([10])
#tensor([2, 3, 3, 3, 3, 3, 0, 4, 3, 3])

from efficientnet_pytorch import EfficientNet
net = EfficientNet.from_name('efficientnet-b5')

#set pretrained_weight
weight_path=pathlib.Path("../input/efficientnet-pytorch/efficientnet-b5-586e6cc6.pth")
os.path.exists(weight_path)

#Load Weight Parameter


# 学習済みパラメータをロード
net_weights = torch.load(
    weight_path,
    map_location=torch.device('cpu') )
keys = list(net_weights.keys())

weights_load = {}

for i in range(len(keys)):
    weights_load[list(net.state_dict().keys())[i]
                 ] = net_weights[list(keys)[i]]

# コピーした内容をモデルに与える設定
state =net.state_dict()
state.update(weights_load)
net.load_state_dict(state)

#Fine Tuning with EfficientNet


#最後layerの出力入れ替える場合は
num_features=net._fc.in_features
print(num_features)
#FC層を入れ替える
net._fc=nn.Linear(num_features,5)

#model構造を確認
net


#GPU

import torch
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("使用device",device)
#networkがある程度固定であれば高速化するプログラムを設定
torch.backends.cudnn.benchmark=True


#-----train------

#Early Stopping

class EarlyStopping:
    def __init__(self,patience=0,verbose=0):
        self._step=0
        self._loss=float("inf")
        self.patience=patience
        self.verbose=verbose
    
    def __call__(self,loss):
        if self._loss < loss:
            self._step += 1
            if self._step > self.patience:
                if self.verbose:
                    print("early stopping")
                return True
            
        else:
            self._step=0
            self._loss=loss
        
        return False


def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):

    # 初期設定
    # GPUが使えるかを確認
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用デバイス：", device)

    # ネットワークをGPUへ
    net.to(device)

    # ネットワークがある程度固定であれば、高速化させる
    torch.backends.cudnn.benchmark = True
    
    #early_stopping:
    es=EarlyStopping(patience=5,verbose=1)  #インスタンスを作成
    
    #dict形式で出力値をストック
    hist={"loss":[],"acc":[],"val_loss":[],"val_acc":[]}
    
    # epochのループ
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-------------')

        # epochごとの訓練と検証のループ
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()  # モデルを訓練モードに
            else:
                net.eval()   # モデルを検証モードに

            epoch_loss = 0.0  # epochの損失和
            epoch_corrects = 0  # epochの正解数
            epoch_f1 = 0.
            epoch_recall=0.
            epoch_precision=0.

            # 未学習時の検証性能を確かめるため、epoch=0の訓練は省略
            if (epoch == 0) and (phase == 'train'):
                continue

            # データローダーからミニバッチを取り出すループ
            for inputs, labels in tqdm(dataloaders_dict[phase]):

                # GPUが使えるならGPUにデータを送る
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # optimizerを初期化
                optimizer.zero_grad()

                # 順伝搬（forward）計算
                with torch.set_grad_enabled(phase == 'train'):

                    outputs= net(inputs)
                    loss = criterion(outputs, labels)  # 損失を計算
                    
                    # 訓練時はバックプロパゲーション
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # 結果の計算
                    epoch_loss += loss.item() * inputs.size(0)  # lossの合計を更新
                    
                    # 正解数の合計を更新
                    _,pred = torch.max(outputs.data, 1)
                    epoch_corrects += torch.sum(pred == labels.data)
                     
            # epochごとのlossと正解率を表示
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            
            epoch_corrects=epoch_corrects.float()
            epoch_acc = epoch_corrects/ len(dataloaders_dict[phase].dataset)
            
            if phase=="train":
                hist["loss"].append(epoch_loss)
                hist["acc"].append(epoch_acc)
            else:
                hist["val_loss"].append(epoch_loss)
                hist["val_acc"].append(epoch_acc)
                
            print("{} Loss:{:.4f},Accuracy:{:.4f},".format(phase, epoch_loss, epoch_acc))
        
        #-----early_stopping:-----
        if phase=="val":
            if es(epoch_loss):
                print("early stopping")
                #グラフ表示の設定:
                fig,(axL,axR)=plt.subplots(ncols=2,figsize=(20,5))
                # plot learning curve
                plt.figure()
                #1回目の計算を取得しないようにしているため,-1を行う必要あり。
                axL.plot(hist["loss"],color='skyblue', label='loss')
                axL.plot(hist["val_loss"], color='orange', label='val_loss')
                axL.legend()
                axL.set_xlabel('epochs')
                axL.set_ylabel('loss')
                axL.grid(True)
                plt.figure()
                axR.plot(hist["acc"], color='skyblue', label='acc')
                axR.plot(hist["val_acc"], color='orange', label='val_acc')
                axR.legend()
                axR.set_xlabel('epochs')
                axR.set_ylabel('accuracy')
                axR.grid(True)
                #early Stoppingでtrainを終了する
                break

        
        

#-----epochs-----
num_epochs=30
#-----models-----
net=net
#-----criterion-----
criterion=nn.CrossEntropyLoss()

#-----learning rate-----
learning_rate=0.0001

#-----optimizer-----
optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate)

from tqdm import tqdm
#正答率、損失関すのグラフまで出力
train_model(net=net,
            dataloaders_dict=dataloaders_dict,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=100)

#Predict
test_path=pathlib.Path("../input/cassava-leaf-disease-classification/test_images")
test_data_dir=[x for x in test_path.iterdir() if x.is_file()]
test_img=Image.open(test_data_dir[0])
plt.imshow(test_img)
plt.show()

valid_aug = albumentations.Compose([
            albumentations.CenterCrop(256, 256, p=1.),
            albumentations.Resize(out_size,out_size),
            albumentations.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225], 
                max_pixel_value=255.0, 
                p=1.0
            )], p=1.)

#BGR読み込み
test_path=pathlib.Path("../input/cassava-leaf-disease-classification/test_images")
test_data_dir=[x for x in test_path.iterdir() if x.is_file()]

test_image_bgr = cv2.imread(str(test_data_dir[0]))
#bgr->rgb
test_img_rgb=cv2.cvtColor(test_image_bgr,cv2.COLOR_BGR2RGB)

# Augment an image
img_transformed = valid_aug(image=test_img_rgb)["image"]
#channel first
#change channels first
img_transformed=np.einsum('ijk->kij', img_transformed)
img_transformed = torch.tensor(img_transformed)
print(img_transformed.shape)

#predict 
net.eval()

#to deviceに調整
output=net(img_transformed.unsqueeze(0).to(device))

#to deviceを忘れずに
with torch.no_grad():
    output=net(img_transformed.unsqueeze(0).to(device))
    
_,pred=torch.max(output,1)
print(pred)

pred=pred.cpu().numpy()
print(pred)


#submit
df_sub.label=pred
df_sub.to_csv("submission.csv",index=False)

