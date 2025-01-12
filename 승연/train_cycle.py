# import
from ResNet import Net
from tictactoe_env import Environment
from self_play import self_play
from train_network import train_network
from evaluate_network import evaluate_network
from evaluate_best_player import evaluate_best_player

# parameter
TRAIN_NUM = 10

state_size = (3,3)
env = Environment(state_size)

CONV_UNITS = 64
model = Net(state_size, env.num_actions, CONV_UNITS)


# network train cycle
for i in range(TRAIN_NUM):
    self_play(model)
    train_network()
    update_best_player = evaluate_network()

    if update_best_player:
        evaluate_best_player()
