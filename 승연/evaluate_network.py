# import
from tictactoe_env import Environment
from mcts import Mcts
from ResNet import Net

import copy
import numpy as np
import pickle

import torch


# parameter
NUM_GAME = 10
TEMPERATURE = 1.0 # 볼츠만 분포

file_name = "model1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state_size = (3,3)
env = Environment(state_size)

CONV_UNITS = 64
model = Net(state_size, env.num_actions, CONV_UNITS).to(device)

# 1 game play하는 함수
def play_game(mcts_list):
    env = Environment(state_size)

    while not env.done:
        state = env.present_state.copy()

        next_player = mcts_list[0] if env.player else mcts_list[1]
        _, action = next_player.get_action()
        _, reward, _, _ = env.step(action)

    point = 1 if reward==1 else 0.5 if reward==-1 else 0
    return point # first player point


# network 평가하는 함수
def evaluate_network():
    model_latest = Net(state_size, env.num_actions, CONV_UNITS).to(device)
    model_best = Net(state_size, env.num_actions, CONV_UNITS).to(device)

    with open(f'{file_name}_model_latest.pkl', 'rb') as f:
        model_latest.load_state_dict(pickle.load(f))

    with open(f'{file_name}_model_best.pkl') as f:
        model_best.load_state_dict(pickle.load(f))

    mcts_latest = Mcts(env, model_latest, env.present_state, TEMPERATURE)
    mcts_best = Mcts(env, model_best, env.present_state, TEMPERATURE)

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

    average_point = total_point/NUM_GAME
    print(f"Average point: {average_point}")

    # best player 교체
    if average_point > 0.5:
        with open(f'{file_name}_model_best.pkl') as f:
            pickle.dump(model_latest, f)

        return True

    else:
        return False




