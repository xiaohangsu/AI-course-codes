# heuristic for selecting the node
def scoreEvaluation(state):
    return state.getScore() + [0,-100.0][state.isLose()] + [0,100.0][state.isWin()]


def normalizedSingleScoreEvaluation(state):
    return (state.getScore() +
            [0,-200.0][state.isLose()] + [0,200.0][state.isWin()]) / 1000.0


def normalizedScoreEvaluation(rootState, currentState):
    rootEval = scoreEvaluation(rootState)
    currentEval = scoreEvaluation(currentState)
    return (currentEval - rootEval) / 1000.0


# heuristic (remaining cost) for A*
def pelletsHeuristic(state):
    if state.isLose():
        return 1000.0;
    return state.getNumFood() + len(self.state.getCapsules());

# current cost for A*
def pelletsCost(state):
    return Game.totalFoodAndCapsules - pelletsHeuristic(state);
