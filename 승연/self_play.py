# import
import numpy as np
import copy
import pickle

from ResNet import ResidualBlock, Net
from mcts import Mcts

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname('/Users/seungyeonlee/Documents/GitHub/24-2-TicTacToe'))))

from Environment import Environment
from Environment import State

# parameter
SP_GAME_COUNT = 500  # 셀프 플레이를 수행할 게임 수(오리지널: 25,000)
SP_TEMPERATURE = 1.0  # 볼츠만 분포의 온도 파라미터

CONV_UNITS = 64
env = Environment()
state = State()

# 1번의 게임 play 함수
def play_one_game(model):
    history = []

    state = State()

    while not is_done:
        mcts = Mcts(env, model, state, temperature = SP_TEMPERATURE)
        policies, _ = mcts.get_action() # 가능한 행동들의 확률 분포 얻기
        legal_actions = np.where(state.get_legal_actions() == 1) # 가능한 행동 (index)

        policy = [0] * (env.n ** 2)
        for action, policy in zip(legal_actions, policies):
            policy[action] = policy
        
        history.append([[state.state, state.enemy_state], policy, None])

        # 가능한 행동 중 랜덤으로 선택해서 게임 진행
        action = state.get_random_action()
        state, is_done, _ = env.step(state, action)

    reward = env.get_reward(state)
    for i in range(len(history)):
        history[i][2] = reward if i % 2 == 0 else -reward

    return history


# self play 함수
def self_play(model):
    env = copy.deepcopy(env)
    data = []

    for i in range(SP_GAME_COUNT):
        
        history = play_one_game(model)
        data.extend(history)

    return data


#############################
# 동작 확인 및 저장 예시
if __name__ == '__main__':
    file_name = "model1"

    state_size = (3,3)
    env = Environment(state_size)
    model = Net(state_size, env.num_actions, CONV_UNITS)

    # 원래 history 불러오기
    try:
        with open(f'{file_name}_history.pkl', 'rb') as f:
            origin_data = pickle.load(f)
    except FileNotFoundError:
        origin_data = [] # 파일이 비어있는 경우 빈 리스트 생성

    # self-play로 새롭게 얻은 history
    data = self_play(model)
    # 원래 history에 추가
    origin_data.extend(data)

    # 파일에 저장
    with open(f'{file_name}_history.pkl', 'wb') as f:
            pickle.dump(origin_data, f)
