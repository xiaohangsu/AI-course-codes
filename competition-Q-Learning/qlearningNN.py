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

from heuristic import scoreEvaluation, normalizedScoreEvaluation

import random
import math
import json
import os
FEATURE_NUMBER = 22


def keyhash(integer, following_str):
    return str(integer) + following_str


def featureHash(features, action):
    return ",".join(features), action


def featurelize(state):
    def manhanttanDistance(point1, point2):
        return math.fabs(point1[0] - point2[0]) + math.fabs(point1[1] - point2[1])

    def euclideanDistance(point1, point2):
        return math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2)

    pacman_pos = state.getPacmanPosition()
    ghosts_pos = state.getGhostPositions()
    pellets_pos = state.getPellets()

    nearest_pellet_pos = ()
    min_dist = 10000.0
    for pellet_pos in pellets_pos:
        if euclideanDistance(pellet_pos, pacman_pos) < min_dist:
            nearest_pellet_pos = pellet_pos
            min_dist = euclideanDistance(pellet_pos, pacman_pos)

    def nearestGhostDistance():
        return min(euclideanDistance(pacman_pos, ghost_pos) for ghost_pos in ghosts_pos)

    def nearestPellet():
        return min_dist

    def ghostDistNearBy():
        return min(euclideanDistance(pacman_pos, ghost_pos) for ghost_pos in ghosts_pos) < 3

    def pelletsOnLeft():
        return len(
            filter(
                (lambda pellet_pos: pacman_pos[0] >= pellet_pos[0]),pellets_pos))

    def pelletsOnRight():
        return len(
            filter(
                (lambda pellet_pos: pacman_pos[0] <= pellet_pos[0]),pellets_pos))

    def pelletsOnTop():
        return len(
            filter(
                (lambda pellet_pos: pacman_pos[1] <= pellet_pos[1]),pellets_pos))

    def pelletsOnBottom():
        return len(
            filter(
                (lambda pellet_pos: pacman_pos[1] >= pellet_pos[1]),pellets_pos))

    def pelletLeft():
        for pellet_pos in pellets_pos:
            if pacman_pos[0] == pellet_pos[0] + 1 and pellet_pos[1] == pacman_pos[1]:
                return True
        return False
    def pelletRight():
        for pellet_pos in pellets_pos:
            if pacman_pos[0] == pellet_pos[0] - 1 and pellet_pos[1] == pacman_pos[1]:
                return True
        return False

    def pelletUp():
        for pellet_pos in pellets_pos:
            if pacman_pos[0] == pellet_pos[0] and pellet_pos[1] - 1 == pacman_pos[1]:
                return True
        return False

    def pelletBottom():
        for pellet_pos in pellets_pos:
            if pacman_pos[0] == pellet_pos[0] and pellet_pos[1] + 1 == pacman_pos[1]:
                return True
        return False

    def nearestPelletOnLeft():
        return pacman_pos[0] > nearest_pellet_pos[0]

    def nearestPelletOnRight():
        return pacman_pos[0] < nearest_pellet_pos[0]

    def nearestPelletOnUp():
        return pacman_pos[1] < nearest_pellet_pos[1]

    def nearestPelletOnBottom():
        return pacman_pos[1] > nearest_pellet_pos[1]

    return [
        len(state.getPellets()) / 100.0,
        nearestPellet() / 20.0,
        nearestGhostDistance() / 20.0,
        1.0 if ghostDistNearBy() else 0.0,

        pelletsOnLeft() / 100.0,
        pelletsOnRight() / 100.0,
        pelletsOnBottom() / 100.0,
        pelletsOnTop() / 100.0,

        1.0 if pelletLeft() else 0.0,
        1.0 if pelletRight() else 0.0,
        1.0 if pelletUp() else 0.0,
        1.0 if pelletBottom() else 0.0,

        1.0 if nearestPelletOnLeft() else 0.0,
        1.0 if nearestPelletOnRight() else 0.0,
        1.0 if nearestPelletOnUp() else 0.0,
        1.0 if nearestPelletOnBottom() else 0.0
    ]


class CompetitionAgent(Agent):

    def __init__(self):
        Agent.__init__(self)
        self.w = {}
        self.epsilon = 0.02  # epsilon greedy algorithm
        self.gamma = 0.8
        self.alpha = 0.1  # learning rate
        self.stepFromStart = 0
        self.learning_time = 2000000

    def initW(self, actions):
        if os.stat('w.json').st_size != 0:
            with open('w.json', 'r') as file:
                self.w = json.load(file)
                file.close()


    def getW(self, key):
        if key in self.w:
            return self.w[key]
        else:
            self.w[key] = 1.0
            return 1.0

    def learning(self, state):
        init_state = state
        for episode in range(self.learning_time):
            action = self.choosePolicy(state, state.getLegalPacmanActions())
            successor = state.generatePacmanSuccessor(action)
            if successor is None:
                print episode
                return False
            reward = normalizedScoreEvaluation(init_state, successor)
            reward -= self.stepFromStart * 0.001
            self.stepFromStart += 1
            self.updateWeight(reward, state, action, successor)

            if successor.isLose() or successor.isWin():
                print "successor is Lose:", successor.isLose()
                state = init_state
                self.stepFromStart = 0
            else:
                state = successor
        return True

    def updateWeight(self, reward, state, action, successor):

        if successor.isLose() or successor.isWin():
            successor_actions = successor.getAllPossibleActions()
        else:
            successor_actions = successor.getLegalPacmanActions()

        features = featurelize(state)

        for i in range(len(features)):
            selfQ = self.Q(state, action)
            delta = reward - selfQ
            if not successor.isWin() and not successor.isLose():
                selfQSuccessor = self.Q(successor, self.choosePolicy(successor, successor_actions))
                delta += self.gamma * selfQSuccessor

            self.w[keyhash(i, str(features[i]) + action)] = self.getW(keyhash(i, str(features[i]) + action)) + self.alpha * delta * features[i]

    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        self.initW(state.getAllPossibleActions())
        if self.learning(state):
            return
        with open('w.json', 'w') as file:
            file.write(json.dumps(self.w))
            file.close()
        return
        # self.initW(state.getAllPossibleActions())
        # self.learning(state)
        # self.neuralNetwork.writeWToFile("w")

    # Q function
    def Q(self, state, action):
        q_value = 0
        features = featurelize(state)

        for i in range(FEATURE_NUMBER):
            q_value += self.getW(keyhash(i, str(features[i]) + action)) * features[i]

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

        # print  actions_q
        return random.choice(max_actions)

    # epsilon action change action
    def chooseAction(self, state):
        prop = random.random()
        actions = state.getLegalPacmanActions()
        if prop < self.epsilon:
            return random.choice(actions)
        else:
            return self.choosePolicy(state, actions)

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write your algorithm Algorithm instead of returning Directions.STOP
        return self.chooseAction(state)