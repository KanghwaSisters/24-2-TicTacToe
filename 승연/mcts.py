# import
import copy
from math import sqrt
import numpy as np
import torch

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname('/Users/seungyeonlee/Documents/GitHub/24-2-TicTacToe'))))

from Environment import State
from Environment import Environment
from ResNet import ResidualBlock, Net

# parameter
C_PUCT = 1.0
EVAL_CNT = 50

state_size = (3,3)
env = Environment(state_size)


# define Mcts class
class Mcts:
    def __init__(self, env, model, state, temperature):
        self.env = env
        self.model = model
        self.state = state
        self.temperature = temperature
        self.legal_actions = self.env.check_legal_action(self.state)

        # define Node class 
        class Node:
            def __init__(self, mcts, state, p):
                self.mcts = copy.deepcopy(mcts) # Mcts 객체 참조 저장
                self.env = copy.deepcopy(self.mcts.env)

                self.state = copy.deepcopy(state)
                self.p = p # policy
                self.n = 0 # count
                self.w = 0 # cumulate value
                self.child_nodes = None

            def evaluate(self):
                is_done, is_win = self.env.is_done(self.state[0])

                # 게임 종료 시 승패에 따라 value 계산
                if is_done:
                    value = 1 if is_win else 0

                    self.w += value
                    self.n += 1
                    return value

                # child node가 없는 경우 => 확장
                if not self.child_nodes:
                    state = self.state
                    state = torch.tensor(state, dtype = torch.float32)
                    state = state.unsqueeze(0)
                    # model을 통해 policy와 value 얻음
                    policies, value = self.predict(state)

                    self.w += value
                    self.n += 1

                    # expand child node
                    self.child_nodes = []
                    legal_actions = np.where(self.mcts.legal_actions != 0)

                    for action, policy in zip(*legal_actions, *policies):
                        self.env.present_state = copy.deepcopy(self.state)
                        next_state, _, _, _ = self.env.step(action)
                        self.child_nodes.append(Node(self.mcts, next_state, policy))

                    return value

                # end node가 아니고, child node가 있는 경우 => 전개
                else:
                    next_child_node = self.next_child_node()
                    value = -next_child_node.evaluate()

                    self.w += value
                    self.n += 1
                    return value
            

            def next_child_node(self):
                '''
                PUCB에 따라 child node를 선택
                '''
                node_scores = list(map(lambda c: c.n, self.child_nodes))

                scores = sum(node_scores)
                # pucb 값에 따라 정렬한 child nodes list (마지막이 최댓값을 갖는 child node)
                pucb_sorted = sorted(self.child_nodes, key = lambda c: (-c.w / c.n if c.n else 0.0) + C_PUCT * c.p * sqrt(scores) / (1 + c.n))

                return pucb_sorted[-1]


            def predict(self, state):
                '''
                model을 통해 policy와 value 계산
                '''
                x = state # 차원 맞추기
                # x = x.unsqueeze(0) 아마도...
                policies, value = self.mcts.model.forward(x)
                policies = policies.detach().numpy()
                value = value.detach().numpy()
                policies = policies * self.mcts.legal_actions # legal action에 대한 policy
                policies /= np.sum(policies) if np.sum(policies) else 1 # 합계 1의 확률분포로 변환

                return policies, value

        self.Node = Node # Node 객체 생성

    ########################
    # methods of Mcts
    def get_policy(self, state):
        '''
        MCTS에 따라 policy 계산
        '''
        root_node = self.Node(self, state, 0) # Mcts 객체 self 전달

        for i in range(EVAL_CNT):
            root_node.evaluate()

        scores = [c.n for c in root_node.child_nodes]

        if self.temperature == 0: # 최대값인 경우에만 1로 지정
            action = np.argmax(scores)
            scores = np.zeros(len(scores))
            scores[action] = 1

        else: # 볼츠만 분포를 기반으로 분산 추가
            scores = self.boltzman(scores, self.temperature)

        return scores

 
    def boltzman(self, xs, temperature):
        '''
        볼츠만 분포
        '''
        xs = [x ** (1/temperature) for x in xs]
        return [x/sum(xs) for x in xs]


    def get_action(self, state):
        '''
        MCTS를 통해 얻은 policy에 따른 action 선택
        '''
        legal_actions = np.where(self.legal_actions != 0)[0]
        policy = self.get_policy(state)
        action = np.random.choice(legal_actions, p=policy)
        return policy, action