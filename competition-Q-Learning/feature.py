
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
        "bias": 0.1
    }
