# import 
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import datasets, transforms

from ResNet import Net
from tictactoe_env import Environment
import numpy as np
import pickle
import random

# parameter
TRAIN_EPOCHS = 100  # 학습 횟수
BATCHSIZE = 64
LEARN_MAX = 0.001

file_name = "model1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state_size = (3,3)
env = Environment(state_size)

CONV_UNITS = 64
model = Net(state_size, env.num_actions, CONV_UNITS).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_MAX)

'''
이게 맞는지 정말 모르겠습니다
정말 정말 정말로요
차원 수정, history 보고 물려야합니다
'''

# loss function 정의
def loss_function(pred_policy, pred_value, y):
    mse = F.mse_loss(pred_policy, y[0])
    cross_entrophy = -torch.mean(y[1] * torch.log(pred_value))
    return mse + cross_entrophy


# dataset을 만드는 함수
def make_dataset(history):
    batch_size = min(BATCHSIZE, len(history))
    mini_batch = random.sample(history, batch_size)
    states, policies, results = zip(*mini_batch) # history에서 어떻게 생겼는지 봐야함

    X = torch.tensor(states, dtype=torch.float32).to(device)
    Y = torch.tensor([policies, results], dtype=torch.float32).to(device)
    return X, Y


# network train하는 함수
def train_network():
    with open(f'{file_name}_history.pkl', 'rb') as f:
        history = pickle.load(f)

    for i in range(TRAIN_EPOCHS):
        X, Y = make_dataset(history)
        # 우선 그냥 현재의 model로 학습한다는 느낌...
        pred_policy, pred_value = model.forward(X)
        loss = loss_function(pred_policy, pred_value, Y)
        # 역전파
        optimizer.zero_grad()
        loss.requires_grad_(True)
        loss.backward()
        optimizer.step()

    # 최근 모델 저장
    with open(f'{file_name}_model_latest.pkl', 'wb') as f:
        pickle.dump(model.state_dict(), f)

    # lr,... epoch,... 조절...
        




   
    
