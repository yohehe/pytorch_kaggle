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
        
#train

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
