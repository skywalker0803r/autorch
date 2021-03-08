import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset,DataLoader
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import warnings
warnings.simplefilter('ignore')

class WADDA(nn.Module):
  def __init__(self,src_x,src_y,tgt_x,tgt_y,
    normalize_idx_list = None,
    limit_y_range = False,
    train_stage_1_epochs = 3000,
    train_stage_2_epochs = 10000,
    ):
    super().__init__()
    '''
    src_x : 模擬數據的x type:pd.DataFrame()
    src_y : 模擬數據的y type:pd.DataFrame()
    tgt_x : 真實數據的x type:pd.DataFrame()
    tgt_y : 真實數據的y type:pd.DataFrame()
    normalize_idx_list : 告訴模型那些輸出欄位加總必須等於1
    limit_y_range : 限制輸出範圍
    '''
    
    # config
    self.device = 'cpu'
    self.x_col = src_x.columns.tolist()
    self.y_col = src_y.columns.tolist()
    self.α = 0.00005
    self.c = 0.01
    self.m = 64
    self.ncritic = 5
    self.input_dim = src_x.shape[1]
    self.output_dim = src_y.shape[1]
    self.normalize_idx_list = normalize_idx_list
    self.limit_y_range = limit_y_range
    self.train_stage_1_epochs = train_stage_1_epochs
    self.train_stage_2_epochs = train_stage_2_epochs
    
    # scaled feature
    self.scaler_x =  StandardScaler().fit(src_x.loc[:])
    src_x.loc[:] = self.scaler_x.transform(src_x.loc[:])
    tgt_x.loc[:] = self.scaler_x.transform(tgt_x.loc[:])
    
    # scaled output if you need
    if self.limit_y_range == True:
      self.scaler_y = MinMaxScaler().fit(src_y.loc[:])
      src_y.loc[:] = self.scaler_y.transform(src_y.loc[:])
      src_y.loc[:] = self.scaler_y.transform(src_y.loc[:])
    
    # pd.DataFrame -> torch.FloatTensor
    src_x,src_y = torch.FloatTensor(src_x.values),torch.FloatTensor(src_y.values)
    tgt_x,tgt_y = torch.FloatTensor(tgt_x.values),torch.FloatTensor(tgt_y.values)
    
    # make two dataset
    self.src_dataset = TensorDataset(src_x,src_y)
    self.tgt_dataset = TensorDataset(tgt_x,tgt_y)
    
    # src data encoder
    self.SRC_F = nn.Sequential(
        nn.Linear(self.input_dim,128),
        nn.ReLU(),
        nn.Linear(128,128),
        nn.ReLU(),
        )
    
    # target data encoder
    self.TGT_F = nn.Sequential(
        nn.Linear(self.input_dim,128),
        nn.ReLU(),
        nn.Linear(128,128),
        nn.ReLU(),
        )
    
    # regression head
    self.regression = nn.Sequential(
        nn.Linear(128,128),
        nn.Dropout(0.25),# more robust
        nn.ReLU(),
        nn.Linear(128,self.output_dim),
        )
    if limit_y_range == True:
      self.regression = nn.Sequential(self.regression,nn.Sigmoid())
    
    # regression loss function
    self.reg_loss = nn.SmoothL1Loss()
    
    
    # 判别器最后一层去掉sigmoid reference Wasserstein GAN(reference Wasserstein GAN)
    self.discriminator = nn.Sequential(nn.Linear(128,1)) # 線性分類器

    # optimizer train_stage_1(回歸訓練) 
    self.S_optimizer = optim.Adam(self.SRC_F.parameters(),lr=1e-4)
    self.R_optimizer = optim.Adam(self.regression.parameters(),lr=1e-4)

    # optimizer train_stage_2(GAN訓練)不要用基于动量的优化算法（包括momentum和Adam），推荐RMSProp，SGD也行(reference Wasserstein GAN)
    self.T_optimizer = optim.RMSprop(self.TGT_F.parameters(),lr=self.α)
    self.D_optimizer = optim.RMSprop(self.discriminator.parameters(),lr=self.α)
  
  def forward(self,src_x,tgt_x):
    src_feat,tgt_feat = self.SRC_F(src_x),self.TGT_F(tgt_x) #特徵抽取
    src_reg,tgt_reg = self.regression(src_feat),self.regression(tgt_feat) #regression預測
    src_domain,tgt_domain = self.discriminator(src_feat),self.discriminator(tgt_feat) #discriminator區分domain
    return src_reg,src_domain,tgt_reg,tgt_domain
  
  def train_S_R(self,src_x,src_y):
    '''
    input : src_x(FloatTensor),src_y(FloatTensor)
    output : loss(Scalar)
    update_method : 一般監督學習
    '''
    self.SRC_F.train()
    self.regression.train()
    
    # forward
    src_feat = self.SRC_F(src_x)
    y_hat = self.regression(src_feat)
    
    # compute loss
    loss = self.reg_loss(y_hat,src_y).mean()
    loss.backward()
    
    # update weight
    self.S_optimizer.step()
    self.R_optimizer.step()
    self.S_optimizer.zero_grad()
    self.R_optimizer.zero_grad()
    
    return loss.item()
  
  def train_T_D(self,src_x,tgt_x,tgt_y):
    '''
    input: src_x(FloatTensor),tgt_x(FloatTensor),tgt_y(FloatTensor)
    return :d_loss(Scalar),t_loss(Scalar),r_loss(Scalar)
    '''
    # 生成器和判别器的loss不取log(reference Wasserstein GAN)
    # train discriminator ncritic times
    for i in range(self.ncritic):
      src_feat = self.SRC_F(src_x).detach()
      tgt_feat = self.TGT_F(tgt_x).detach()
      d_loss = -torch.mean(self.discriminator(src_feat)) + torch.mean(self.discriminator(tgt_feat))
      d_loss.backward()
      self.D_optimizer.step()
      self.D_optimizer.zero_grad()
      # 每次更新判别器的参数之后把它们的绝对值截断到不超过一个固定常数c
      for p in self.discriminator.parameters():
        p.data.clamp_(-self.c,self.c)
    
    # train TGT_F
    tgt_feat = self.TGT_F(tgt_x)
    t_loss = -torch.mean(self.discriminator(tgt_feat))
    t_loss.backward()
    self.T_optimizer.step()
    self.T_optimizer.zero_grad()

    # train regression
    tgt_reg = self.regression(tgt_feat.detach())
    r_loss = self.reg_loss(tgt_reg,tgt_y).mean()
    r_loss.backward()
    self.R_optimizer.step()
    self.R_optimizer.zero_grad()
    
    return d_loss.item(),t_loss.item(),r_loss.item()

  def train_stage_1(self,num_epoch=3000,log_interval=100):
    history = []
    for ep in tqdm(range(num_epoch)):
      idx = random.sample([*range(len(self.src_dataset))],self.m)
      src_x,src_y = self.src_dataset[idx]
      loss = self.train_S_R(src_x,src_y)
      history.append(loss)
      if ep % log_interval == 0:
        print("ep:{} loss:{}".format(ep,loss))
    plt.plot(history,label='train_loss')
    plt.legend()
    plt.show()
  
  def train_stage_2(self,num_epoch=10000,log_interval=100):
    d_history = []
    t_history = []
    r_history = []
    for ep in tqdm(range(num_epoch)):
      tgt_idx = random.sample([*range(len(self.tgt_dataset))],self.m)
      src_idx = random.sample([*range(len(self.src_dataset))],self.m)
      tgt_x,tgt_y = self.tgt_dataset[tgt_idx]
      src_x,src_y = self.src_dataset[src_idx]
      d_loss,t_loss,r_loss = self.train_T_D(src_x,tgt_x,tgt_y)
      d_history.append(d_loss)
      t_history.append(t_loss)
      r_history.append(r_loss)
      if ep % log_interval == 0:
        print("ep:{} d_loss:{} t_loss:{} r_loss:{}".format(ep,d_loss,t_loss,r_loss))
    plt.plot(d_history,label='d_loss')
    plt.plot(t_history,label='t_loss')
    plt.plot(r_history,label='r_loss')
    plt.legend()
    plt.show()
  
  def train(self,log_interval=100):
    print('start train')
    self.train_stage_1(self.train_stage_1_epochs,log_interval)
    self.train_stage_2(self.train_stage_2_epochs,log_interval)
    print('end train')
  
  @staticmethod
  def normalize(x):
    '''
    x : pandas.DataFrame()
    return : normalize x
    '''
    x_idx,x_col = x.index,x.columns
    x = x.values
    x = x / x.sum(axis=1).reshape(-1,1)
    return pd.DataFrame(x,index=x_idx,columns=x_col)

  def predict(self,tgt_x):
    '''
    input: pd.DataFrame()
    output: pd.DataFrame()
    '''
    self.TGT_F.eval()
    self.regression.eval()
    tgt_x = self.scaler_x.transform(tgt_x)
    tgt_x = torch.FloatTensor(tgt_x)
    tgt_feat = self.TGT_F(tgt_x)
    tgt_reg = self.regression(tgt_feat).detach().cpu().numpy()
    tgt_reg = pd.DataFrame(tgt_reg,columns=self.y_col)
    
    if self.limit_y_range == True:
      tgt_reg[:] = self.scaler_y.inverse_transform(tgt_reg[:])
    
    if self.normalize_idx_list != None:
      for idx in self.normalize_idx_list:
        tgt_reg.iloc[:,idx] = self.normalize(tgt_reg.iloc[:,idx])
    
    return tgt_reg
  
  def shrink(self):
    '''
    drop all data ready to save as pkl file
    '''
    self.src_dataset = None
    self.tgt_dataset = None
    self.S_optimizer = None
    self.R_optimizer = None
    self.T_optimizer = None
    self.D_optimizer = None
    
  def to(self,device):
    self.device = device
    self.TGT_F.to(self.device)
    self.SRC_F.to(self.device)
    self.regression.to(self.device)
    self.discriminator.to(self.device)
  
if __name__ == '__main__':
    
    src_x = pd.DataFrame(np.random.uniform(size=(1000,100)))*100
    src_y = pd.DataFrame(np.random.uniform(size=(1000,41*4)))*100
    tgt_x = pd.DataFrame(np.random.uniform(size=(1000,100)))*100
    tgt_y = pd.DataFrame(np.random.uniform(size=(1000,41*4)))*100
    
    normalize_idx_list = []
    for i in range(41):
      normalize_idx_list.append([i,i+41,i+41*2,i+41*3])
    
    model = WADDA(src_x, src_y, tgt_x, tgt_y,
                  normalize_idx_list = normalize_idx_list,
                  limit_y_range = True,
                  train_stage_1_epochs=100,
                  train_stage_2_epochs=100,
                  )
    model.train()
    res = model.predict(tgt_x)
    print(res.describe())
    for idx in normalize_idx_list:
      print(res.loc[:,idx].sum(axis=1))
    