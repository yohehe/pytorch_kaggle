#efficient Netを使用する場合
#inputに+add DataでEfficientNetをダウンロードする

#sys
import sys
# efficientnet with internet_off
sys.path.append("../input/efficientnet-pytorch/EfficientNet-PyTorch/EfficientNet-PyTorch-master")

from efficientnet_pytorch import EfficientNet
net = EfficientNet.from_name('efficientnet-b5')

#set pretrained_weight
weight_path=pathlib.Path("../input/efficientnet-pytorch/efficientnet-b5-586e6cc6.pth")
os.path.exists(weight_path)


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
#<All keys matched successfully>


#最後前結合層のlayerの出力入れ替える場合は
num_features=net._fc.in_features
print(num_features)
#上記の方法でアクセス可能。
#FC層を入れ替える
net._fc=nn.Linear(num_features,5)

#あとはいつも通りでOK
import torch
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("使用device",device)
#networkがある程度固定であれば高速化するプログラムを設定
torch.backends.cudnn.benchmark=True
