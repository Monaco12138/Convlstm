import os
import torch
import torch.nn as nn
from torch.optim import Adam
from models import convlstm

class Model(object):
    def __init__(self, configs):
        self.configs = configs
        self.num_hidden = [int(x) for x in configs.num_hidden.split(',')]       #self.num_hidden=[64,64,64,64]
        self.num_layers = len(self.num_hidden)                                  #隐藏层为4层
        networks_map = {                                                        #将所有模型放入map中供以后选择
            'convlstm': convlstm.RNN
        }

        if configs.model_name in networks_map:
            Network = networks_map[configs.model_name]                           #创建将要使用的模型
            self.network = Network(self.num_layers, self.num_hidden, configs).to(configs.device) #初始化将要使用的模型(隐藏层个数，隐藏层参数，所有参数)
        else:
            raise ValueError('Name of network unknown %s' % configs.model_name)

        self.optimizer = Adam(self.network.parameters(), lr=configs.lr)         #优化器定义
        self.MSE_criterion = nn.MSELoss()                                       #误差为均方误差

    def save(self, itr):
        stats = {}
        stats['net_param'] = self.network.state_dict()
        checkpoint_path = os.path.join(self.configs.save_dir, 'model.ckpt'+'-'+str(itr))
        torch.save(stats, checkpoint_path)
        print("save model to %s" % checkpoint_path)

    def load(self, checkpoint_path):
        print('load model:', checkpoint_path)
        stats = torch.load(checkpoint_path)
        self.network.load_state_dict(stats['net_param'])

    def train(self, frames, mask, itr=1):
        frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        mask_tensor = torch.FloatTensor(mask).to(self.configs.device)
        self.optimizer.zero_grad()
        next_frames = self.network(frames_tensor, mask_tensor, itr)
        loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:])    #(4 , 19 , 140//4 , 140//4 , 4*4*1)
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().numpy()

    def test(self, frames, mask, itr=1):
        frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        mask_tensor = torch.FloatTensor(mask).to(self.configs.device)
        next_frames = self.network(frames_tensor, mask_tensor, itr)
        return next_frames.detach().cpu().numpy()