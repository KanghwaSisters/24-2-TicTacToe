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

# parameter
C_PUCT = 1.0
EVAL_CNT = 50

env = Environment()
state = State()

##################################
def predict(state, model):
    '''
    model을 통해 policy와 value 계산
    '''
    x = state # 차원 맞추기
    # x = x.unsqueeze(0) 아마도...
    policies, value = model.forward(x)
    
    policies = policies.detach().numpy()
    value = value.detach().numpy()

    legal_actions = list(np.where(state.get_legal_actions()))
    policies = policies[legal_actions] # legal action에 대한 policy
    policies /= np.sum(policies) if np.sum(policies) else 1 # 합계 1의 확률분포로 변환

    return policies, value


# define Mcts class
class Mcts:
    def __init__(self, model, temperature):
        self.model = model
        self.temperature = temperature

        # define Node class ##################
        class Node:
            def __init__(self, state, p):
                self.state = state
                self.p = p # policy
                self.n = 0 # visit count
                self.w = 0 # cumulative sum of values
                self.child_nodes = None

            def evaluate(self):
                is_done, _ = self.state.check_done()

                # 게임 종료 시 승패 여부에 따라 value 업데이트
                if is_done:
                    value = env.get_reward(self.state)
                    self.w += value
                    self.n += 1
                    return value

                # child node가 없는 경우 확장
                if not self.child_nodes:
                    state = self.state.copy()
                    
                    policies, value = self.predict(state, model)

                    self.w += value
                    self.n += 1

                    # expand child node
                    self.child_nodes = []
                    legal_actions = np.where(state.get_legal_actions() == 1)

                    for action, policy in zip(legal_actions, policies):
                        next_state, _, _ = env.step(state, action)
                        self.child_nodes.append(Node(next_state, policy))

                    return value

                # end node가 아니고, child node가 있는 경우 -> 전개
                else:
                    next_child_node = self.get_next_child_node()
                    value = - next_child_node.evaluate()

                    self.w += value
                    self.n += 1

                    return value

            def get_next_child_node(self):
                '''
                PUCB에 따라 child node를 선택
                '''
                node_scores = list(map(lambda c: c.n, self.child_nodes))

                scores = sum(node_scores)

                def pucb(c):
                    return (-c.w / c.n if c.n else 0.0) + C_PUCT * c.p * sqrt(scores) / (1 + c.n)

                # pucb 값에 따라 정렬한 child nodes list (마지막이 최댓값을 갖는 child node)
                pucb_sorted = sorted(self.child_nodes, key = lambda c: pucb(c))

                return pucb_sorted[-1]
        #########################################

        self.Node = Node # Node 객체 생성

    # methods of Mcts
    def get_policy(self, root_state):
        '''
        MCTS에 따라 policy 계산
        '''
        root_node = self.Node(root_state, 0) # root node 생성

        for _ in range(EVAL_CNT):
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


    def get_action(self, state, policy):
        '''
        MCTS를 통해 얻은 policy에 따른 action 선택
        '''
        legal_actions = np.where(state.get_legal_actions() == 1)
        # policy = self.get_policy(self.state)
        action = np.random.choice(legal_actions, p=policy)
        return action