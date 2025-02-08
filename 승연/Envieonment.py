# import
import numpy as np

from State import State

# parameter
STATE_SIZE = (3, 3)

#class
class Environment:
    __slots__ = ('n', 'num_actions', 'action_space', 'reward_dict')

    def __init__(self):
        self.n = STATE_SIZE[0]
        self.num_actions = self.n ** 2
        self.action_space = np.arange(self.num_actions)
        self.reward_dict = {'win': 1, 'lose': -1, 'draw': 0}


    def step(self, present_state, action_idx :int):
        '''
        present_state에 대해 action_idx의 행동에 따라 게임을 한 턴 진행시키고
        next_state, is_done, is_lose를 반환한다.
        '''
        next_state = present_state.next(action_idx)
        is_done, is_lose = next_state.check_done()

        return next_state, is_done, is_lose


    def get_reward(self, final_state):
        '''
        게임이 종료된 state에 대해 last player의 reward를 반환한다.
        final_state: 게임이 종료된 state
        '''
        _, is_lose = final_state.check_done()
        reward = self.reward_dict['lose'] if is_lose else self.reward_dict['draw']

        return reward


    def get_first_reward(self, final_state):
        '''
        게임이 종료된 state에 대해 first player의 reward를 반환한다.
        final_state: 게임이 종료된 state
        note: final_state가 is_lose라면, 해당 state에서 행동할 차례였던 플레이어가 패배한 것.
        '''
        is_first_player = final_state.check_first_player()
        reward = self.get_reward(final_state) if is_first_player else -self.get_reward(final_state)

        return reward


    def render(self, state):
        '''
        입력받은 state를 문자열로 출력한다.
        X: first_player, O: second_player
        '''
        is_first_player = state.check_first_player()
        board = state.state - state.enemy_state if is_first_player else state.enemy_state - state.state
        board = board.reshape(-1)
        board_list = list(map(lambda x: 'X' if board[x] == 1 else 'O' if board[x] == -1 else '.', self.action_space))

        board_string = ' '.join(board_list)
        formatted_string = '\n'.join([board_string[i:i+6] for i in range(0, len(board_string), 6)])

        print(formatted_string)

    ######## play game method ######
    def play_one_game(self, player_list):
        is_done = False
        state = State()

        while not is_done:
            player = player_list[0] if state.check_first_player() else player_list[1]

            action = player.get_action(state)
            state, is_done, _ = self.step(state, action)

        # 게임 종료 후 first player 기준 reward, point 계산
        reward = self.get_first_reward(state)

        point = 1 if reward == 1 else 0.5 if reward == 0 else 0
        return point # first player point


    def evaluate_first_algorithm(self, label, player_list, num_game):
        inverse_player_list = [None, None]
        inverse_player_list[0], inverse_player_list[1] = player_list[1], player_list[0]
        total_point = 0
        cnt_win = 0
        cnt_draw = 0

        result_list = []

        for i in range(num_game):
            if i % 2 == 0:
                point = self.play_one_game(player_list)
                total_point += point
                cnt_win += 1 if point == 1 else 0
        
            else:
                point = self.play_one_game(inverse_player_list)
                total_point += 1 - point
                cnt_win += 1 if point == 0 else 0

            cnt_draw += 1 if point == 0.5 else 0

            # change player
            player_list[0].player = not player_list[0].player
            player_list[1].player = not player_list[1].player

        average_point = total_point / num_game
        print(f"{label}: {round(average_point, 5)} / win:{cnt_win} / draw:{cnt_draw}")

        return average_point, cnt_win, cnt_draw


    def show_one_game(self, player_list):
        is_done = False
        state = State()

        game_boards = []
        game_boards_string = ['', '', '']

        while not is_done:
            player = player_list[0] if state.check_first_player() else player_list[1]

            action = player.get_action(state)
            state, is_done, _ = self.step(state, action)

            board = state.get_total_state() if state.check_first_player() else -state.get_total_state()
            board = board.reshape(-1)
            board_list = list(map(lambda x: 'X' if board[x] == 1 else 'O' if board[x] == -1 else '.', self.action_space))
            board_string = ' ' + ' '.join(board_list)
            formatted_string = '|'.join([board_string[i:i+6] for i in range(0, len(board_string), 6)]) + '|'

            game_boards_string[0] += formatted_string[:7]
            game_boards_string[1] += formatted_string[7:14]
            game_boards_string[2] += formatted_string[14:]

        game_boards_string = '\n'.join(game_boards_string)
        print(game_boards_string)

        # 게임 종료 후 first player 기준 reward, point 계산
        reward = self.get_first_reward(state)
        print(f"first_reward: {reward}")