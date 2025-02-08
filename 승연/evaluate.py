# import
import torch
import pandas as pd
import numpy as np
import pickle


from mcts import Mcts
from file_save_load import load_model, save_model
from Environment import Environment
from enemy_agents import RandomAgent, AlphaBetaAgent, McsAgent, MctsAgent

# parameter
file_name = 'test'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_GAME = 10
CRITERIA = 0.6
TEMPERATURE = 1.0

env = Environment()

# class
class Evaluate:
    __slots__ = ('file_name', 'result_list')

    def __init__(self, file_name):
        self.file_name = file_name
        # [label, point, win, draw, lose]
        self.result_list = []


    def _save_game_result(self, label, point, win, draw):
        self.result_list.append([label, point, win, draw, NUM_GAME-win-draw])


    def save_result_data(self):
        try:
            with open(f'{self.file_name}_result_data.pkl', 'rb') as f:
                load_data = pickle.load(f)

            df = pd.DataFrame(self.result_list)
            df = pd.concat([load_data, df], axis=0)
            
        except FileNotFoundError:
            df = pd.DataFrame(self.result_list)

        with open(f'{self.file_name}_result_data.pkl', 'wb') as f:
            pickle.dump(df, f)


    def evaluate_network(self):
        model_latest = load_model(f'{self.file_name}_model_latest.pkl').to(DEVICE)
        model_best = load_model(f'{self.file_name}_model_best.pkl').to(DEVICE)

        mcts_latest = Mcts(model_latest, TEMPERATURE)
        mcts_best = Mcts(model_best, TEMPERATURE)

        player_list = [mcts_latest, mcts_best]

        # 대전
        average_point, cnt_win, cnt_draw = env.evaluate_first_algorithm("Evaluate network", player_list, NUM_GAME)
        self._save_game_result('latest vs. best', average_point, cnt_win, cnt_draw)

        player_list = [mcts_latest, mcts_best]
        env.show_one_game(player_list)

        # best player 교체
        if average_point > CRITERIA:
            save_model(f'{self.file_name}_model_best.pkl', model_latest.to('cpu'))
            return True
        else:
            return False


    def evaluate_best_player(self):
        model = load_model(f'{self.file_name}_model_best.pkl').to(DEVICE)

        mcts_best = Mcts(model)
        agents_dict = {'random': RandomAgent(False),
                        'alpha-beta': AlphaBetaAgent(False),
                        'mcs': McsAgent(False),
                        'mcts': MctsAgent(False)}

        count = 0
        for key in agents_dict.keys():
            agent = agents_dict[key]
            player_list = [mcts_best, agent]
            count += 1
            average_point, cnt_win, cnt_draw = env.evaluate_first_algorithm(f"[{count}/{len(agents_dict)}] vs. {key}", player_list, NUM_GAME)
            self._save_game_result(key, average_point, cnt_win, cnt_draw)

            player_list = [mcts_best, agent]
            env.show_one_game(player_list)
            print("="*30)




