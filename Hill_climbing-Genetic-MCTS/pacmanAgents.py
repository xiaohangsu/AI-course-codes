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
from heuristics import *
import random
import math


class RandomAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # get all legal actions for pacman
        actions = state.getLegalPacmanActions()
        # returns random action from all the valide actions
        return actions[random.randint(0, len(actions) - 1)]


class RandomSequenceAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        self.actionList = [];
        for i in range(0, 10):
            self.actionList.append(Directions.STOP);
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # get all legal actions for pacman
        possible = state.getAllPossibleActions();
        for i in range(0, len(self.actionList)):
            self.actionList[i] = possible[random.randint(0, len(possible) - 1)];
        tempState = state;
        for i in range(0, len(self.actionList)):
            if tempState.isWin() + tempState.isLose() == 0:
                tempState = tempState.generatePacmanSuccessor(self.actionList[i]);
            else:
                break;
        # returns random action from all the valide actions
        return self.actionList[0];


class GreedyAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # get all legal actions for pacman
        legal = state.getLegalPacmanActions()
        # get all the successor state for these actions
        successors = [(state.generatePacmanSuccessor(action), action) for action in legal]
        # evaluate the successor states using scoreEvaluation heuristic
        scored = [(scoreEvaluation(state), action) for state, action in successors]
        # get best choice
        bestScore = max(scored)[0]
        # get all actions that lead to the highest score
        bestActions = [pair[1] for pair in scored if pair[0] == bestScore]
        # return random action from the list of the best actions
        return random.choice(bestActions)


class HillClimberAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        self.actionList = []
        self.actions = state.getAllPossibleActions()
        for i in range(0, 5):
            self.actionList.append(self.actions[random.randint(0, len(self.actions) - 1)])
        return

    # 50% chances to mutate each action in actionList
    def mutate(self):
        secondActionList = self.actionList
        for i in range(0, len(secondActionList)):
            if random.uniform(0, 1) < 0.5:
                secondActionList[i] = self.actions[random.randint(0, len(self.actions) - 1)]

        return secondActionList

    # GetAction Function: Called with every frame
    def getAction(self, state):
        secondActionList = self.mutate()
        bestActions = []  # maximum is two best actions
        state1 = state
        state2 = state
        for action1, action2 in zip(self.actionList, secondActionList):
            state1 = state1.generatePacmanSuccessor(action1)
            state2 = state2.generatePacmanSuccessor(action2)
            if state1.isLose():
                break
            if state2.isLose():
                break
        score1, score2 = scoreEvaluation(state1), scoreEvaluation(state2)
        print self.actionList, secondActionList
        print score1, score2

        if score1 < score2:
            bestActions.append(secondActionList[0])
            self.actionList = secondActionList
        else:
            bestActions.append(self.actionList[0])

        # Pop first action and append new random action
        self.actionList.pop(0)
        self.actionList.append(self.actions[random.randint(0, len(self.actions) - 1)])

        return bestActions[random.randint(0, len(bestActions) - 1)]


class GeneticAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        self.actionLists = []
        self.totalRanks = 0
        self.actions = state.getAllPossibleActions()
        for i in range(0, 8):
            actionList = []
            for i in range(0, 5):
                actionList.append(self.actions[random.randint(0, len(self.actions) - 1)])
            self.actionLists.append(actionList)

        return

    # GetAction Function: Called with every frame
    def getAction(self, state):
        action = self.rankSelection()[0]
        self.generate()
        self.actionLists.sort(key=lambda x: self.getScoreFromActionList(x, state.generatePacmanSuccessor(action)))
        return action

    def generate(self):
        newGeneration = []
        while len(newGeneration) != len(self.actionLists):
            actionList1 = self.rankSelection()
            actionList2 = self.rankSelection()
            if random.uniform(0, 1) <= 0.7:
                actionList1, actionList2 = self.crossover(actionList1, actionList2)
            if random.uniform(0, 1) <= 0.1:
                actionList1 = self.mutate(actionList1)
            if random.uniform(0, 1) <= 0.1:
                actionList2 = self.mutate(actionList2)

            newGeneration.append(actionList1)
            newGeneration.append(actionList2)
        # sort Children
        self.actionLists = newGeneration

    def crossover(self, actionList1, actionList2):
        children1 = []
        children2 = []
        for i in range(0, len(actionList1)):
            if random.uniform(0, 1) < 0.5:
                children1.append(actionList1[i])
            else:
                children1.append(actionList2[i])

        for i in range(0, len(actionList2)):
            if random.uniform(0, 1) < 0.5:
                children2.append(actionList1[i])
            else:
                children2.append(actionList2[i])
        return children1, children2

    def rankSelection(self):
        self.totalRanks = 0
        for i in range(1, len(self.actionLists) + 1):
            self.totalRanks += i

        chance = random.randint(1, self.totalRanks)
        total = 0
        for i in range(1, len(self.actionLists) + 1):
            if total + i <= chance and chance < total + i + i + 1:
                return self.actionLists[i - 1]
            total = total + i
        return []

    def mutate(self, actionList):
        actionList[random.randint(0, len(actionList) - 1)] \
            = self.actions[random.randint(0, len(self.actions) - 1)]
        return actionList

    def getScoreFromActionList(self, actionList, state):
        for action in actionList:
            if state.isLose():
                break
            state = state.generatePacmanSuccessor(action)
            if state.isLose():
                break
        scoreEvaluation(state)


class MCTSNode:
    parent = None

    def __init__(self, pa):
        self.parent = pa
        self.reward = 0.0
        self.visited = 0
        self.children = {}  # 'South': NextNode, 'North': NextNode


def NODE_PRINT(mctsNode, state, nodeName=""):
    if nodeName is not "":
        print "Node Name: ", nodeName
    print "Node reward: ", mctsNode.reward
    print "Node visited: ", mctsNode.visited
    print "Node Children: ", mctsNode.children
    print "Node parent: ", mctsNode.parent
    print "Node state: \n", state


class MCTSAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        self.constant = 1.0  # Use 1 as the constant between exploitation and exploration.
        self.root = None
        self.state = None
        return

    def treePolicy(self, mctsNode):
        v = mctsNode
        while self.state is not None and \
                not self.state.isLose() and \
                not self.state.isWin() and v is not None:
            actions = self.state.getLegalPacmanActions()
            for action in actions:
                if action not in v.children:
                    children = self.expand(v, action)
                    if children is None:
                        continue
                    v.children[action] = children
                    return v.children[action]
            if v.children == {}:
                return None
            if self.state is not None and \
                    not self.state.isLose() and \
                    not self.state.isWin():
                v = self.select(v, self.constant)
        return None

    def expand(self, mctsNode, action):
        nextState = self.state.generatePacmanSuccessor(action)
        if nextState is None or nextState.isLose():
            return None
        self.state = nextState
        return MCTSNode(mctsNode)

    def select(self, mctsNode, constant):
        def selection_formula(parent, children, constant):
            return (children.reward / children.visited) + \
               constant * math.sqrt(2.0 * math.log1p(parent.visited) / children.visited)

        maxNodes = []
        maxActions = []
        currentMaxValue = 0.0
        # print mctsNode.children
        for action in mctsNode.children:
            children = mctsNode.children[action]
            currentValue = selection_formula(mctsNode, children, constant)
            if currentMaxValue < currentValue:
                maxNodes = [children]
                maxActions = [action]
                # print "select action: ", action
                currentMaxValue = currentValue
            elif currentMaxValue == currentValue:
                maxNodes.append(children)
                maxActions.append(action)
        index = random.randint(0, len(maxNodes) - 1)
        action = maxActions[index]
        self.state = self.state.generatePacmanSuccessor(action)
        if self.state is None or self.state.isWin() or self.state.isLose():
            return None
        return maxNodes[index]

    def defaultPolicy(self, state, rootState):
        i = 0
        while not state.isLose() and i != 5:
            actions = state.getLegalPacmanActions()
            successor = state.generatePacmanSuccessor(
                actions[random.randint(0, len(actions) - 1)])
            if successor is None or successor.isLose():
                break
            state = successor
            i += 1
        return normalizedScoreEvaluation(rootState, state)

    def backPropagation(self, v, r):
        while v is not None:
            v.visited += 1
            v.reward += r
            v = v.parent

    # return actionsList
    def bestChildPolicy(self, root):
        bestActions = []
        maxVisited = 0
        for child in root.children:
            if root.children[child] is None:
                continue
            if root.children[child].visited > maxVisited:
                bestActions = [child]
                maxVisited = root.children[child].visited
            elif root.children[child].visited == maxVisited:
                bestActions.append(child)
        return bestActions

    # GetAction Function: Called with every frame
    def getAction(self, state):
        self.root = MCTSNode(None)
        self.state = state

        vl = self.treePolicy(self.root)
        while vl is not None:  # within computational budget
            reward = self.defaultPolicy(self.state, state)
            if self.state is None:
                break
            self.backPropagation(vl, reward)
            self.state = state
            vl = self.treePolicy(self.root)

        bestActions = self.bestChildPolicy(self.root)
        print state
        if bestActions is []:
            actions = state.getLegalPacmanActions()
            return actions[random.randint(0, len(actions) - 1)]
        bestAction = bestActions[random.randint(0, len(bestActions) - 1)]
        # self.root = self.root.children[bestAction]
        # self.root.parent = None
        # self.clearNode(self.root)
        return bestAction
