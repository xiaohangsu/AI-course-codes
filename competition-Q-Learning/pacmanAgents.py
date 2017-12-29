# pacmanAgents.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from pacman import Directions
from game import Agent
import numpy as np
from game import Actions
import random
import math
import json
import os
import random
import json
import os


def getLegalAction(state):
    actions = state.getLegalPacmanActions()
    actions.append("Stop")
    return actions


FEATURE_NUMBER = 4
walls_pos = set()

# Generate wall position
# Only run at beginning of state
def generateWallPos(state):
    pacman_pos = state.getPacmanPosition()
    for i in range(20):
        for j in range(11):
            walls_pos.add((i, j))
    walls_pos.remove(pacman_pos)
    for pos in state.getPellets():
        walls_pos.remove(pos)

    for pos in state.getGhostPositions():
        walls_pos.remove(pos)

    for pos in state.getCapsules():
        walls_pos.remove(pos)

    walls_pos.remove((9, 5))
    walls_pos.remove((9, 6))
    walls_pos.remove((10, 5))
    walls_pos.remove((10, 6))


def featurelize(state, action):
    Direction = {
        "North": (0, 1),
        "South": (0, -1),
        "East":  (1, 0),
        "West":  (-1, 0),
        "Stop": (0, 0)
    }

    pacman_pos = state.getPacmanPosition()
    next_pos = (pacman_pos[0] + Direction[action][0], pacman_pos[1] + Direction[action][1])
    ghosts_pos = state.getGhostPositions()
    pellets_pos = state.getPellets()
    capsules_pos = state.getCapsules()

    def nearestPellet():
        # BFS
        visit_set = set()
        visit_set.add(pacman_pos)
        visit_set.add(next_pos)
        que = [(next_pos, 0)]
        while len(que) != 0:
            temp_que = que
            que = []
            while len(temp_que) != 0:
                top_pos, dist = temp_que.pop()
                visit_set.add(top_pos)
                for action in Direction:
                    top_next_pos = (top_pos[0] + Direction[action][0], top_pos[1] + Direction[action][1])
                    if top_next_pos in walls_pos or top_next_pos in visit_set:
                        continue
                    if top_next_pos in pellets_pos or top_next_pos in capsules_pos:
                        return dist
                    else:
                        que.append((top_next_pos, dist + 1))
        # No food ?
        return 0.0

    def ghostDistNearBy():
        ghost_near_by = 0
        que = []
        for action in Direction:
            if (next_pos[0] + Direction[action][0], next_pos[1] + Direction[action][1]) in ghosts_pos:
                ghost_near_by += 1

        return ghost_near_by

    def eatFood():
        return 1 if ghostDistNearBy() == 0 and next_pos in pellets_pos else 0.0

    return {
        "near-ghosts": ghostDistNearBy() / 10.0,
        "near-food": nearestPellet() / 100.0,
        "eat-food": eatFood() / 10.0,
        "bias": 1.0
    }

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def de_sigmoid(x):
    return x * (1 - x)


def relu(x):
    return np.maximum(0, x)


def de_relu(x):
    return 1. * (x > 0)


class NeuralLayer:
    def __init__(self, input_num, output_num, hidder_node_num):
        self.w1 = np.random.rand(input_num + 1, hidder_node_num) - 0.5
        self.w2 = np.random.rand(hidder_node_num + 1, output_num) - 0.5
        self.input1  = np.array([])
        self.input2  = np.array([])
        self.output1 = np.array([])
        self.output2 = np.array([])
        self.error1  = np.array([])
        self.error2  = np.array([])

    def forward(self, input_list):
        self.input1 = np.append(np.array(input_list), 1.0)
        self.output1 = self.activate(self.input1.dot(self.w1))
        self.input2 = np.append(self.output1, 1.0)
        self.output2 = self.activate(self.input2.dot(self.w2))
        return self.output2.tolist()

    def backPropagation(self, error_vec):
        self.error2 = error_vec
        self.error1 = error_vec.dot(self.w2[:-1].transpose())
        self.error1 = self.deactive(self.output1, self.error1)

    def descentGradient(self, learning_rate):
        self.w1 -= learning_rate * np.outer(self.input1.transpose(), self.error1)
        self.w2 -= learning_rate * np.outer(self.input2.transpose(), self.error2)

    def activate(self, input_vec):
        return sigmoid(input_vec)

    def deactive(self, output, pre_error_vec):
        np_de_sigmoid = np.vectorize(de_sigmoid)
        return np.multiply(pre_error_vec, np_de_sigmoid(output))

    def readWFromFile(self, filename):
        w_pair = np.load(filename)
        self.w1, self.w2 = w_pair[0], w_pair[1]

    def writeWToFile(self, filename):
        np.save(filename, [self.w1, self.w2])

# one of train weights
sample_weights = {"near-ghosts": -119.29632886891633, "near-food": -402.98104848634176, "eat-food": 533.7816809071539, "bias": -304.2004845395081}

class CompetitionAgent(Agent):

    def __init__(self):
        Agent.__init__(self)
        self.w = {"near-ghosts": -119.29632886891633, "near-food": -402.98104848634176, "eat-food": 533.7816809071539, "bias": -304.2004845395081}
        self.weights = NeuralLayer(FEATURE_NUMBER, 1, 40)
        self.win = 0
        self.epsilon = 0.05  # epsilon greedy algorithm
        self.gamma = 0.8
        self.alpha = 0.2  # learning rate
        self.stepFromStart = 0
        self.learning_time = 200000
        self.lose = 0

    def initW(self):
        if os.stat('w.json').st_size != 0:
            with open('w.json', 'r') as file:
                self.w = json.load(file)
                file.close()

    def getW(self, key):
        if key in self.w:
            return self.w[key]
        else:
            self.w[key] = 1.0
            return self.w[key]

    def learning(self, state):
        init_state = state
        for episode in range(self.learning_time):
            action = self.choosePolicy(state, state.getLegalPacmanActions())

            successor = state.generatePacmanSuccessor(action)
            if successor is None:
                return False
            self.stepFromStart += 1

            reward = successor.getScore() - state.getScore() - self.stepFromStart
            self.updateWeight(reward, state, action, successor)
            if successor.isLose() or successor.isWin():
                if successor.isLose():
                    self.lose += 1
                if successor.isWin():
                    self.win += 1
                print self.lose, self.win
                state = init_state
                self.stepFromStart = 0
            else:
                state = successor
        return True

    def updateWeight(self, reward, state, action, successor):
        successor_actions = successor.getLegalPacmanActions()
        features = featurelize(state, action)
        selfQ = self.Q(state, action)
        delta = reward - selfQ
        if not successor.isWin() and not successor.isLose():
            selfQSuccessor = self.Q(successor, self.choosePolicy(successor, successor_actions))
            delta += self.gamma * selfQSuccessor
        for feature in features:
            self.w[feature] = self.getW(feature) + self.alpha * delta * features[feature]

    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        # generateWallPos(state)
        # self.initW()
        # if self.learning(state):
        #     return
        # with open('w.json', 'w') as file:
        #     file.write(json.dumps(self.w))
        #     file.close()

        # After init we can stop trainning
        self.epsilon = 0.0
        return

    # Q function
    def Q(self, state, action):
        q_value = 0
        features = featurelize(state, action)
        for feature in features:
            q_value += self.getW(feature) * features[feature]
        return q_value

    # choose actions policy
    # Open to modify. For now pick max one.
    def choosePolicy(self, state, actions):
        max_q_value = max([self.Q(state, action) for action in actions])
        max_actions= []
        actions_q = []

        for action in actions:
            if self.Q(state, action) == max_q_value:
                max_actions.append(action)
            actions_q.append((self.Q(state, action), action))

        return random.choice(max_actions)

    # epsilon action change action
    def chooseAction(self, state):
        prop = random.random()
        actions = getLegalAction(state)
        if prop < self.epsilon:
            return random.choice(actions)
        else:
            return self.choosePolicy(state, actions)

    # GetAction Function: Called with every frame
    def getAction(self, state):
        return self.chooseAction(state)