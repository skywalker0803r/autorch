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
from .loss_function import hingeLoss

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
        lr = 1e-3,
        max_epochs = 300,
        log_interval = 50,
        n_round = 32,# The number after the floating point number
        normalize_idx_list = None,
        device = "cpu",
        hingeLoss = False
        ):
        
        # config
        self.hingeLoss = hingeLoss
        self.n_round = n_round
        self.normalize_idx_list = normalize_idx_list
        self.log_interval = log_interval
        self.device = device
        self.x_col = x_col
        self.y_col = y_col
        self.hidden_size = hidden_size
        self.lr = lr
        self.max_epochs = max_epochs
        self.ss_x = MinMaxScaler().fit(df[x_col])
        self.ss_y = MinMaxScaler().fit(df[y_col])
        
        # net
        self.net = nn.Sequential(
            nn.Linear(len(self.x_col),self.hidden_size),nn.ReLU(),
            nn.Linear(self.hidden_size,self.hidden_size),nn.ReLU(),
            nn.Linear(self.hidden_size,len(self.y_col)),nn.Sigmoid(),
            ).apply(self.init_weights).to(self.device)
        
        # loss function
        if self.hingeLoss == True:
            self.loss_fn = hingeLoss
        else:
            self.loss_fn = nn.SmoothL1Loss()
        
        # optimizer
        self.optimizer = torch.optim.Adam(self.net.parameters(),lr=self.lr)
        
        # dataset
        self.data = self.split_data(df,self.x_col,self.y_col)
        
        # train_data_iter
        self.train_data = TensorDataset(
            torch.FloatTensor(self.ss_x.transform(self.data['X_train'])).to(self.device),
            torch.FloatTensor(self.ss_y.transform(self.data['Y_train'])).to(self.device),
            )
        self.train_iter = DataLoader(self.train_data,batch_size=64)

        # vaild_data_iter
        self.vaild_data = TensorDataset(
            torch.FloatTensor(self.ss_x.transform(self.data['X_vaild'])).to(self.device),
            torch.FloatTensor(self.ss_y.transform(self.data['Y_vaild'])).to(self.device),
            )
        self.vaild_iter = DataLoader(self.vaild_data,batch_size=64)
    
    '''
    ## help functions area ##
    '''
    @staticmethod
    def mape(y_true, y_pred, e = 1e-8):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        mask = y_true > e
        y_true, y_pred = y_true[mask], y_pred[mask]
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    @staticmethod
    def split_data(df,x_col,y_col):
        df = shuffle(df).astype('float32')
        X,Y = df[x_col],df[y_col]
        sp1 = int(len(df)*0.8)
        sp2 = int(len(df)*0.9)
        data = {}
        data['X_train'],data['Y_train'] = X.iloc[:sp1,:],Y.iloc[:sp1,:]
        data['X_vaild'],data['Y_vaild'] = X.iloc[sp1:sp2,:],Y.iloc[sp1:sp2,:]
        data['X_test'],data['Y_test'] = X.iloc[sp2:,:],Y.iloc[sp2:,:]
        return data

    def show_metrics(self,y_real,y_pred,e=1e-8):
        res = pd.DataFrame(index=y_pred.columns,columns=['R2','MSE','MAPE'])
        for i in y_pred.columns:
            res.loc[i,'R2'] = max(r2_score(y_real[i],y_pred[i]),0)
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
            if self.hingeLoss == True:
                loss = self.loss_fn(y_hat,y,self.net)
            else:
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
      
            # pring info
            if i%self.log_interval == 0:
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
        data_index = x.index
        predict = self.net(torch.FloatTensor(self.ss_x.transform(x)).to(self.device))
        predict = self.ss_y.inverse_transform(predict.detach().cpu().numpy())
        predict = pd.DataFrame(predict,index=data_index,columns=self.y_col)
    
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
