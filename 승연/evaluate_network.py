# import
import copy
import numpy as np
import pickle

import torch

from mcts import Mcts
from ResNet import Net

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname('/Users/seungyeonlee/Documents/GitHub/24-2-TicTacToe'))))

from Environment import Environment
from Environment import State

# parameter
NUM_GAME = 10
TEMPERATURE = 1.0 # 볼츠만 분포

file_name = "model1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STATE_SIZE = (3,3)
env = Environment()

CONV_UNITS = 64

##############################
# 1 game play하는 함수
def play_game(mcts_list):
    is_done = False
    state = State()

    while not is_done:
        player = mcts_list[0] if state.check_first_player() else mcts_list[1]

        _, action = player.get_action()
        state, is_done, _ = env.step(state, action)

    is_first_player = state.check_first_player()
    reward = env.get_reward(state) if is_first_player else -env.get_reward(state)

    point = 1 if reward == 1 else 0.5 if reward == 0 else 0
    return point # first player point


# model 파라미터를 가져오는 함수
def load_model(file, model):
    with open(file, 'rb') as f:
        model.load_state_dict(pickle.load(f))

# best model을 저장하는 함수 (train_network와 겹치는 함수)
def save_model(file, model):
    with open(file, 'wb') as f:
        pickle.dump(model, f)


# network 평가하는 함수
def evaluate_network():
    model_latest = Net(env.num_actions, CONV_UNITS).to(device)
    model_best = Net(env.num_actions, CONV_UNITS).to(device)

    load_model(f'{file_name}_model_latest.pkl', model_latest)
    load_model(f'{file_name}_model_best.pkl', model_best)

    mcts_latest = Mcts(model_latest, TEMPERATURE)
    mcts_best = Mcts(model_best, TEMPERATURE)

    mcts_list = [mcts_latest, mcts_best]

    # 대전
    total_point = 0
    for i in range(NUM_GAME):
        # 선 플레이어를 교대하면서 대전
        if i % 2 == 0: # first player: latest
            point = play_game(mcts_list)

        else: # first player: best
            mcts_list[[0, 1]] = mcts_list[[1, 0]]
            point = 1 - play_game(mcts_list) # latest의 point

        total_point += point

    average_point = total_point / NUM_GAME
    print(f"Average point: {average_point}")

    # best player 교체
    if average_point > 0.5:
        save_model(f'{file_name}_model_best.pkl', model_latest)
        return True

    else:
        return False




