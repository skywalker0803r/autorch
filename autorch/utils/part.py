from torch.utils.data import TensorDataset,DataLoader
from copy import deepcopy
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch import tensor
from torch.nn import Linear,ReLU,Sigmoid,Tanh
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score,mean_squared_error
from math import sqrt
import joblib
import warnings;warnings.simplefilter('ignore')
from tqdm import tqdm
import os
from sklearn.utils import shuffle
import random
from robust_loss_pytorch import AdaptiveLossFunction

class PartBulider(object):
    '''
    # df : pandas.DataFrame
    # x_col : list of columns name
    # y_col : list of columns name
    '''
    def __init__(
        self,
        df,
        x_col,
        y_col,
        hidden_size = 256,
        batch_size = 64,
        lr = 1e-3,
        max_epochs = 300,
        log_interval = 50,
        n_round = 32,# The number after the floating point number
        cut_way = [0.8,0.9],
        normalize_idx_list = None,
        device = "cpu",
        use_robust_Loss = False,
        limit_y_range = False
        ):
        
        # config
        self.use_robust_Loss = use_robust_Loss
        self.n_round = n_round
        self.cut_way = cut_way
        self.normalize_idx_list = normalize_idx_list
        self.log_interval = log_interval
        self.device = device
        self.x_col = x_col
        self.y_col = y_col
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.lr = lr
        self.max_epochs = max_epochs
        self.ss_x = MinMaxScaler().fit(df[x_col])
        self.limit_y_range = limit_y_range
        
        if self.limit_y_range == True:
            self.ss_y = MinMaxScaler().fit(df[y_col])
        if self.limit_y_range == False:
            self.ss_y = None
        
        # net
        if self.limit_y_range == True:
            self.net = nn.Sequential(
                nn.Linear(len(self.x_col),self.hidden_size),nn.ReLU(),
                nn.Linear(self.hidden_size,self.hidden_size),nn.ReLU(),
                nn.Linear(self.hidden_size,len(self.y_col)),nn.Sigmoid(),
                ).apply(self.init_weights).to(self.device)
        
        if self.limit_y_range == False:
            self.net = nn.Sequential(
                nn.Linear(len(self.x_col),self.hidden_size),nn.ReLU(),
                nn.Linear(self.hidden_size,self.hidden_size),nn.ReLU(),
                nn.Linear(self.hidden_size,len(self.y_col)),
                ).apply(self.init_weights).to(self.device)

        
        # loss function
        if self.use_robust_Loss == True:
            print('use_robust_Loss')
            adaptive = AdaptiveLossFunction(
                num_dims = len(self.y_col), 
                float_dtype = np.float32, 
                device = self.device)
            def loss_fn(y_i,y):
                return torch.mean(adaptive.lossfun((y_i - y)))
            self.loss_fn = loss_fn
            params = list(self.net.parameters()) + list(adaptive.parameters())
        else:
            self.loss_fn = nn.SmoothL1Loss()
            params = list(self.net.parameters())
        
        # optimizer
        self.optimizer = torch.optim.Adam(params,lr=self.lr)
        
        # dataset
        self.data = self.split_data(df,self.x_col,self.y_col)
        
        # train_data_iter
        if self.limit_y_range == True:
            self.train_data = TensorDataset(
                torch.FloatTensor(self.ss_x.transform(self.data['X_train'])).to(self.device),
                torch.FloatTensor(self.ss_y.transform(self.data['Y_train'])).to(self.device),
                )
            self.train_iter = DataLoader(self.train_data,batch_size=self.batch_size)

            # vaild_data_iter
            self.vaild_data = TensorDataset(
                torch.FloatTensor(self.ss_x.transform(self.data['X_vaild'])).to(self.device),
                torch.FloatTensor(self.ss_y.transform(self.data['Y_vaild'])).to(self.device),
                )
            self.vaild_iter = DataLoader(self.vaild_data,batch_size=self.batch_size)
        
        if self.limit_y_range == False:
            self.train_data = TensorDataset(
                torch.FloatTensor(self.ss_x.transform(self.data['X_train'])).to(self.device),
                torch.FloatTensor(self.data['Y_train'].values).to(self.device),
                )
            self.train_iter = DataLoader(self.train_data,batch_size=self.batch_size)

            # vaild_data_iter
            self.vaild_data = TensorDataset(
                torch.FloatTensor(self.ss_x.transform(self.data['X_vaild'])).to(self.device),
                torch.FloatTensor(self.data['Y_vaild'].values).to(self.device),
                )
            self.vaild_iter = DataLoader(self.vaild_data,batch_size=self.batch_size)

    
    '''
    ## help functions area ##
    '''
    @staticmethod
    def mape(y_true, y_pred, e = 1e-8):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        mask = y_true > e
        y_true, y_pred = y_true[mask], y_pred[mask]
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    def split_data(self,df,x_col,y_col):
        df = shuffle(df).astype('float32')
        X,Y = df[x_col],df[y_col]
        sp1 = int(len(df)*self.cut_way[0])
        sp2 = int(len(df)*self.cut_way[1])
        data = {}
        data['X_train'],data['Y_train'] = X.iloc[:sp1,:],Y.iloc[:sp1,:]
        data['X_vaild'],data['Y_vaild'] = X.iloc[sp1:sp2,:],Y.iloc[sp1:sp2,:]
        data['X_test'],data['Y_test'] = X.iloc[sp2:,:],Y.iloc[sp2:,:]
        return data

    def show_metrics(self,y_real,y_pred,e=1e-8):
        res = pd.DataFrame(index=y_pred.columns,columns=['R2','MSE','MAPE'])
        for i in y_pred.columns:
            res.loc[i,'R2'] = np.clip(r2_score(y_real[i],y_pred[i]),0,1)
            res.loc[i,'MSE'] = mean_squared_error(y_real[i],y_pred[i])
            res.loc[i,'MAPE'] = self.mape(y_real[i],y_pred[i],e)
        res.loc['AVG'] = res.mean(axis=0)
        return res
    
    @staticmethod
    def init_weights(m):
        if hasattr(m,'weight'):
            torch.nn.init.xavier_uniform(m.weight)
        if hasattr(m,'bias'):
            m.bias.data.fill_(0.01)
    
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
  
  # train step
    def train_step(self):
        self.net.train()
        total_loss = 0
        for t,(x,y) in enumerate(self.train_iter):
            y_hat = self.net(x)
            loss = self.loss_fn(y_hat,y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            total_loss += loss.item()
        return total_loss/(t+1)
  
  # valid step
    def valid_step(self):
        self.net.eval()
        total_loss = 0
        for t,(x,y) in enumerate(self.vaild_iter):
            y_hat = self.net(x)
            loss = self.loss_fn(y_hat,y)
            total_loss += loss.item()
        return total_loss/(t+1)

  
    def train(self):
        '''
        train and eval model many epochs
        return best model and plot train_history
        '''   
        history = {
            'train_loss':[],
            'valid_loss':[]
            }
        current_loss = np.inf
        best_model = None
    
        for i in tqdm(range(self.max_epochs)):
            history['train_loss'].append(self.train_step())
            history['valid_loss'].append(self.valid_step())
      
            # print info
            if i % self.log_interval == 0:
                print("epoch:{} train_loss:{:.4f} valid_loss:{:.4f}".format(i,history['train_loss'][-1],history['valid_loss'][-1]))
      
            # keep the best model
            if history['valid_loss'][-1] <= current_loss:
                best_model = deepcopy(self.net.eval())
                current_loss = history['valid_loss'][-1]
    
        # plot history and return best_model
        self.net = deepcopy(best_model.eval())
        plt.plot(history['train_loss'],label='train_loss')
        plt.plot(history['valid_loss'],label='valid_loss')
        plt.legend()
        plt.show()
        return self

    def test(self,e=1e-8):
        '''
        show model metrics
        '''
        predict = self.predict(self.data['X_test'])
        res = self.show_metrics(self.data['Y_test'],predict,e)
        return res

    def predict(self,x):
        '''
        input :pandas.DataFrame()
        return :pandas.DataFrame()
        '''
        self.net.eval()
        predict = self.net(torch.FloatTensor(self.ss_x.transform(x)).to(self.device))
        
        if self.ss_y != None:
            predict = self.ss_y.inverse_transform(predict.detach().cpu().numpy())

        else:
            predict = predict.detach().cpu().numpy()
        
        predict = pd.DataFrame(predict,index=x.index,columns=self.y_col)
    
        # normalize
        if self.normalize_idx_list != None:
            for idx in self.normalize_idx_list:
                predict.iloc[:,idx] = self.normalize(predict.iloc[:,idx])
        
        # The number after the floating point number
        predict = predict.round(self.n_round)
        return predict

    def shrink(self):
        '''
        drop all data ready to save as pkl file
        '''
        self.data = None
        self.train_data = None
        self.vaild_data = None
        self.train_iter = None
        self.vaild_iter = None
        self.optimizer = None
    
    def to(self,device):
        self.device = device
        self.net.to(self.device)
        
if __name__ == '__main__':
    df = pd.DataFrame(np.random.normal(size=(1000,20)))
    x_col = df.columns[:10]
    y_col = df.columns[10:]
    part = PartBulider(df,x_col,y_col,use_robust_Loss=True,limit_y_range=False)
    part.train()
    res = part.test()
    print(res)
    