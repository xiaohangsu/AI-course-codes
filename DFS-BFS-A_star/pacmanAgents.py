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
from heuristics import scoreEvaluation
import random
import bisect

class RandomAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # get all legal actions for pacman
        actions = state.getLegalPacmanActions()
        # returns random action from all the valide actions
        return actions[random.randint(0,len(actions)-1)]


class GreedyAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # get all legal actions for pacman
        legal = state.getLegalPacmanActions()
        # get all the successor state for these actions
        successors = [(state.generateSuccessor(0, action), action) for action in legal]
        # evaluate the successor states using scoreEvaluation heuristic
        scored = [(scoreEvaluation(state), action) for state, action in successors]
        # get best choice
        bestScore = max(scored)[0]
        # get all actions that lead to the highest score
        bestActions = [pair[1] for pair in scored if pair[0] == bestScore]
        # return random action from the list of the best actions
        return random.choice(bestActions)


class BFSAgent(Agent):
    # Initialization Function: Called one time when the game starts

    def registerInitialState(self, state):
        return

    # GetAction Function: Called with every frame
    def getAction(self, state):
        que = [(state, [])]
        depth = 6
        while depth != 0:
            temp_que = que
            que = []
            while temp_que:
                temp_state, actions = temp_que.pop()
                temp_legal = temp_state.getLegalPacmanActions()

                if temp_state.isLose():
                    continue
                for action in temp_legal:
                    successor =  temp_state.generatePacmanSuccessor(action)
                    if successor is None:
                        break
                    que.append((successor, actions + [action]))
            depth -= 1
            if not que:
                print "Too large depth, not result"
                break
        scored = max([(scoreEvaluation(s), actions) for s, actions in que],key=lambda item:item[0])
        best_action = [actions[0] for s, actions in que if scored[0] == scoreEvaluation(s)]
        return random.choice(best_action)


class DFSAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return

    # GetAction Function: Called with every frame
    def getAction(self, state):
        _, action = self.DFSGetAction(state, None, 4)
        return action

    # recursive function for getAction
    def DFSGetAction(self, state, prev_action, depth):
        if depth == 0:
            return scoreEvaluation(state), prev_action
        legal = state.getLegalPacmanActions()
        best_score = 0
        best_action = [prev_action]
        for action in legal:
            successor = state.generatePacmanSuccessor(action)
            if successor is None:
                continue
            if successor.isLose():
                continue
            score, _ = self.DFSGetAction(successor, action, depth - 1)
            if best_score < score:
                best_action = [action]
                best_score = score
            elif best_score == score:
                best_action.append(action)
        return best_score, random.choice(best_action)


class AStarAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return

    def totalCostFunction(self, depth, state_now, root_state):
        return depth - (scoreEvaluation(state_now) - scoreEvaluation(root_state))

    # GetAction Function: Called with every frame
    def getAction(self, state):
        depth = 0
        root_state = state

        que = [(root_state, "", depth)]
        score_list = []

        no_more_successor = False
        while True:
            state_now, init_action, state_now_depth = que.pop(0)
            legal = state_now.getLegalPacmanActions()
            for action in legal:
                successor = state_now.generatePacmanSuccessor(action)
                if successor is None:
                    no_more_successor = True
                    break
                if successor.isLose():
                    continue

                successor_score = self.totalCostFunction(state_now_depth + 1, successor, root_state)
                insert_index = bisect.bisect_left(score_list, successor_score)
                score_list.insert(insert_index, successor_score)
                que.insert(insert_index, (successor, action if init_action == "" else init_action, state_now_depth + 1))

            if no_more_successor:
                break

        best_que = que[0:bisect.bisect_right(score_list, score_list[0])]
        best_actions = [init_action for _, init_action, _ in best_que]
        return random.choice(best_actions)
