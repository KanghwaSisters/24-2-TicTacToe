# import
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random

from ResNet import Net
from file_save_load import load_history, load_model, save_history, save_model
from mcts import Mcts
from Environment import Environment
from State import State

# parameter
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LEARN_RATE = 0.001
GAMMA = 0.1
SP_GAME_COUNT = 2  # 셀프 플레이를 수행할 게임 수(오리지널: 25,000)
SP_TEMPERATURE = 1.0

model = Net()

OPTIMIZER = optim.Adam(model.parameters(), lr=LEARN_RATE)
SCHEDULER = optim.lr_scheduler.StepLR(OPTIMIZER, step_size=10, gamma=GAMMA)
CROSS_ENTROPY = torch.nn.CrossEntropyLoss()

BATCHSIZE = 4
TRAIN_EPOCHS = 10
MEM_SIZE = 10000

env = Environment()

file_name = 'test'

# class
class TrainNetwork:
    __slots__ = ('file_name', 'model', 'device', 'optimizer', 'scheduler', 'cross_entropy', 'memory')

    def __init__(self, file_name):
        self.file_name = file_name
        self.device = DEVICE

        self.model = load_model(f'{file_name}_model_latest.pkl')
        self.optimizer = OPTIMIZER
        self.scheduler = SCHEDULER
        self.cross_entropy = CROSS_ENTROPY

        self.memory = load_history(f'{file_name}_history.pkl')


    def _self_play_one_game(self):
        '''
        model을 통한 MCTS 알고리즘에 따라
        1번의 게임을 진행하는 함수
        '''
        history = []

        state = State()
        is_done = False

        while not is_done:
            mcts = Mcts(self.model, temperature = SP_TEMPERATURE)
            # state = state.to(self.device)
            policies = mcts.get_policy(state) # 가능한 행동들의 확률 분포 얻기
            legal_actions = state.get_legal_actions() # 가능한 행동 (index)

            # 전체 행동에 대한 policy
            policy = [0] * env.num_actions
            for action, p in zip(legal_actions, policies):
                policy[action] = p

            total_state = state.get_total_state()
            history.append([total_state, policy, None])

            # 가능한 행동 중 랜덤으로 선택해서 게임 진행
            action = mcts.get_action(state)
            state, is_done, _ = env.step(state, action)

        # first player 기준의 reward로 바꾸기
        reward = env.get_first_reward(state)

        for i in range(len(history)):
            history[i][2] = reward if i % 2 == 0 else -reward

        return history


    def _self_play(self):
        '''
        self-play를 진행해 data를 만드는 함수
        '''
        data = []

        for i in range(SP_GAME_COUNT):
            history = self._self_play_one_game()
            data.extend(history)
            if (i+1) % 10 == 0:
                print(f"{i+1}/{SP_GAME_COUNT}", end=" | ")

        return data


    def save_self_play_history(self):
        '''
        self_play를 진행하고, file 경로에 history를 저장하는 함수
        '''
        data = self._self_play()
        self.memory.extend(data)
        save_history(f'{file_name}_history.pkl', self.memory)


    def _loss_function(self, pred_policy, pred_value, y):
        '''
        Define loss function
        '''
        y_policy = y[:, :-1]
        y_value = y[:, -1:]

        mse = F.mse_loss(pred_policy, y_policy)
        cross_entropy = self.cross_entropy(pred_value, y_value)
        return mse + cross_entropy


    def _make_dataset(self):
        '''
        history에서 랜덤 추출하여 학습 가능한 dataset 형태로 만드는 함수
        '''
        batch_size = min(BATCHSIZE, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)
        states, policies, results = zip(*mini_batch)

        policies = np.array(policies)
        results = np.array(results).reshape(-1, 1)
        Y_array = np.concatenate([policies, results], axis=1)

        X = torch.tensor(states, dtype=torch.float32).to(self.device) # (batch_size, )
        X = X.unsqueeze(1)
        Y = torch.tensor(Y_array, dtype=torch.float32).to(self.device) # (batch_size, 2)

        return X, Y


    def train_network(self):
        '''
        최근 모델을 불러와서 학습하는 함수
        '''
        self.model = self.model.to(self.device)

        for _ in range(TRAIN_EPOCHS):
            X, Y = self._make_dataset()
            pred_policy, pred_value = self.model.forward(X)
            loss = self._loss_function(pred_policy, pred_value, Y)
            # 역전파
            self.optimizer.zero_grad()
            loss.requires_grad_(True)
            loss.backward()
            self.optimizer.step()

        # 최근 모델 저장
        save_model(f'{file_name}_model_latest.pkl', self.model.to('cpu'))

        return loss