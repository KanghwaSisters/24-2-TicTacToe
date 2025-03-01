# import
import numpy as np
import copy

from Environment import Environment

# parameter
STATE_SIZE = (3, 3)

# class
class State(Environment):
    __slots__ = ('state', 'enemy_state')

    def __init__(self, state=None, enemy_state=None):
        super().__init__()
        self.state = state if state != None else [0] * (self.n ** 2)
        self.enemy_state = enemy_state if enemy_state != None else [0] * (self.n ** 2)

        self.state = np.array(self.state).reshape(STATE_SIZE)
        self.enemy_state = np.array(self.enemy_state).reshape(STATE_SIZE)


    def total_pieces_count(self):
        '''
        이 state의 전체 돌의 개수를 반환한다.
        '''
        total_state = self.state + self.enemy_state
        return np.sum(total_state)


    def get_legal_actions(self):
        '''
        이 state에서 가능한 action idx의 리스트를 반환한다.
        '''
        total_state = (self.state + self.enemy_state).reshape(-1)
        legal_actions = np.where(total_state == 0)[0]
        return legal_actions


    def get_onehot_legal_actions(self):
        '''
        이 state에서 가능한 action을
        one-hot encoding 형식의 array로 반환한다.
        '''
        legal_actions = self.get_legal_actions()
        onehot_legal_actions = np.array([legal_actions[x] == 0 for x in self.action_space], dtype = int)
        return legal_actions


    def check_done(self):
        '''
        이 state의 done, lose 여부를 반환한다.
        note: 상대가 행동한 후, 자신의 행동을 하기 전 이 state를 확인한다.
        따라서 이전 state에서 상대의 행동으로 상대가 이긴 경우는 이 state의 플레이어가 진 경우이다.
        '''
        is_done, is_lose = False, False

        # Check draw
        if self.total_pieces_count() == self.n ** 2:
            is_done, is_lose = True, False

        # Check lose
        lose_condition = np.concatenate([self.enemy_state.sum(axis=0), self.enemy_state.sum(axis=1), [self.enemy_state.trace()], [np.fliplr(self.enemy_state).trace()]])
        if self.n in lose_condition:
            is_done, is_lose = True, True

        return is_done, is_lose


    def next(self, action_idx):
        '''
        주어진 action에 따라 다음 state를 생성한다.
        note: 다음 state는 상대의 차례이므로 state 순서를 바꾼다.
        '''
        x, y =np.divmod(action_idx, self.n)
        state = self.state.copy()
        state[x, y] = 1

        state = list(state.reshape(-1))
        enemy_state = list(copy.copy(self.enemy_state).reshape(-1))

        return State(enemy_state, state)


    def check_first_player(self):
        '''
        이 state의 플레이어가 first player인지 확인한다.
        '''
        return (self.total_pieces_count() % 2) == 0


    def get_random_action(self):
        '''
        이 state에서 가능한 action 중 랜덤으로 action을 반환한다.
        '''
        legal_actions = self.get_legal_actions()
        action = np.random.choice(legal_actions)
        return action


    def get_total_state(self):
        '''
        history에 넣을 전체 게임보드 state
        자신의 수: 1 / 상대의 수: -1 / 빈칸: 0
        '''
        # return (self.state, self.enemy_state)
        return self.state - self.enemy_state


    def render(self):
        '''
        이 state를 렌더링한다.
        '''
        super().render(self)