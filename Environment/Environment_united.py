# import
from typing import Tuple
import numpy as np
import random

# parameter
state_size = (3,3)

# class tictactoe environment
'''
- player의 state 구분 (first player: X, second_player:O)
- player를 전환한 후에 state를 판단
- 따라서 lose가 없고 win, draw만 판단
- reward는 player=True 기준으로 제공

- step 외부에서 state를 판단해도 같은 결과를 얻을 수 있음
'''
class Environment:
    def __init__(self, state_size:Tuple):
        # env size
        self.state_size = state_size # (3, 3)
        self.n = self.state_size[0] # 3
        self.num_actions = self.n ** 2 # 9

        # state, action
        self.present_state = np.zeros((2, self.n, self.n)) # (2, 3, 3), present_state[0]: state for first player
        self.action_space = np.arange(self.num_actions) # [0, 1, ..., 8] : action idx

        # reward, done
        self.reward_dict = {'win':1, 'lose':-1, 'draw':0, 'progress':0}
        self.done = False
        
        # 플레이어 
        self.player = True # True: first player


    def step(self, action_idx):
        '''
        Advance the game by one step according to the (input)action_idx 
        output: next_state, reward, done, is_win
        '''
        x, y = divmod(action_idx, self.n)

        self.present_state[0][x, y] = 1

        # check winner of the game
        next_state = self.present_state
        done, is_win = self.is_done(next_state)
        reward = self.get_reward(is_win)
        self.done = done

        # change turn
        self._change_player()
        
        return next_state ,reward, done, is_win


    def reset(self):
        '''
        Reset game.
        '''
        self.present_state = np.zeros((2, self.n, self.n))
        self.done = False
        self.player = True


    def render(self, state):
        '''
        Print the (input)state as a string.
        first player: X / second player: O
        '''
        state = state if self.player else state[[1, 0]]
        state = state.reshape(2,-1)
        board = state[0] - state[1] # -1: player / 1: enemy
        board = board.reshape(-1)
        board_list = list(map(lambda x: 'X' if board[x] == 1 else 'O' if board[x] == -1 else '.', self.action_space))

        # string으로 변환하고 game board 형태로 출력
        board_string = ' '.join(board_list)
        formatted_string = '\n'.join([board_string[i:i+6] for i in range(0, len(board_string), 6)])

        print(formatted_string)
        # print("-"*7)
        

    def get_legal_actions(self, state):
        '''
        Return legal action array(one-hot encoding) in (input)state.
        '''
        state = state.reshape(2,-1)
        board = state[0]+state[1]
        legal_actions = np.array([board[x] == 0 for x in self.action_space], dtype = int)
        return legal_actions # [9]


    def is_done(self, state):
        '''
        Check the winner of the game.
        - is_win= True: win / False: draw
        '''
        is_done, is_win = False, False


        # 무승부 여부 확인
        if np.sum(state) == self.n ** 2:
            is_done, is_win = True, False

        my_state = state[0]
        # 승리 조건 확인
        win_condition = np.concatenate([my_state.sum(axis=0), my_state.sum(axis=1), [my_state.trace()], [np.fliplr(my_state).trace()]]) # (8, )
        if self.n in win_condition:
            is_done, is_win = True, True

        return is_done, is_win


    def _change_player(self):
        '''
        Change the state and the player to next player.
        '''
        self.present_state[[0, 1]] = self.present_state[[1, 0]]
        self.player = not self.player


    def get_reward(self, is_win):
        '''
        Return rewards with consideration for the player.
        - draw, progress: 0 / lose: -1 / win: 1
        '''
        reward = 0

        if is_win:
            reward = self.reward_dict["lose"] if self.player else self.reward_dict["win"]

        return reward

    # 부가적인 메서드
    def get_random_action(self, state):
        '''
        Randomly select one action in legal actions.
        '''
        legal_actions = self.get_legal_actions(state)
        legal_action_idxs = np.where(legal_actions != 0)[0]
        action = np.random.choice(legal_action_idxs)

        return action
