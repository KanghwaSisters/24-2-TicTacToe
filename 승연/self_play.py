# import
import numpy as np
import copy
import pickle

from tictactoe_env import Environment
from ResNet import ResidualBlock, Net
from mcts import Mcts


# parameter
SP_GAME_COUNT = 500  # 셀프 플레이를 수행할 게임 수(오리지널: 25,000)
SP_TEMPERATURE = 1.0  # 볼츠만 분포의 온도 파라미터

CONV_UNITS = 64

# 1번의 게임 play 함수
def play_one_game(model):
    env.reset()
    history = []

    while not env.done:
        state = env.present_state.copy()
        mcts = Mcts(env, model, state, temperature = SP_TEMPERATURE)
        policy, action = mcts.get_action()
        _, reward, _, _ = env.step(action)

        state[[0, 1]] = state[[1, 0]] if not env.player else state[[0, 1]] # player = True 기준, (player_state, enemy_state) 고정
        history.append((state, policy))

    history = [(x[0], x[1], reward) for x in history]

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
