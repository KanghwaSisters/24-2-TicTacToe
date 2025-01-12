# import
import torch.nn as nn
import torch.nn.functional as F

# parameter
state_size = (3, 3) # env.state_size
action_size = 9 # env.action_size

CONV_UNITS = 64
RESIDUAL_NUM = 16
BATCHSIZE = 64

# ResNet
'''
차원 오류 해결
But 제대로 되는지 모르겠음
'''
# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3,3), bias=False, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        sc = x

        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        
        x = self.conv(x)
        x = self.bn(x)
        x += sc
        x = F.relu(x)
        return x

# main Net
class Net(nn.Module):
    def __init__(self, state_size, action_size, conv_units):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=conv_units, kernel_size=(3,3), bias=False, padding=1)
        self.bn = nn.BatchNorm2d(conv_units)
        self.pool = nn.MaxPool2d(kernel_size=(3,3), stride=1, padding=1)
        self.residual_block = ResidualBlock(conv_units, conv_units)

        self.batch_size = BATCHSIZE

        # self.fc_size = state_size[-1]*state_size[-2]*conv_units
        # self.fc1 = nn.Linear(self.fc_size, action_size)

        # self.fc2 = nn.Linear(state_size[-1]*state_size[-2], 1)
        # self.tanh = nn.Tanh()

        self.policy_head = nn.Sequential(
            nn.Conv2d(conv_units, 2, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2, action_size),
            nn.Softmax(dim=1)
        )
        
        self.value_head = nn.Sequential(
            nn.Conv2d(conv_units, 1, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pool(x)

        # residual block
        for i in range(RESIDUAL_NUM):
            x = self.residual_block(x)

        # pooling
        x = self.pool(x)

        # 전연결 신경망
        # x = x.view(self.batch_size, -1)
        # x = self.fc1(x)

        # # policy: action probability (vector) - softmax
        # policy = F.softmax(x, dim=-1)

        # # value: win probability (scalar) - tanh
        # x = self.fc2(x)
        # value = self.tanh(x)

        policy = self.policy_head(x)
        value = self.value_head(x)

        return policy, value