import numpy as np

class grid_world:

    def __init__(self, HEIGHT, WIDTH, GOAL, OBSTACLES):
        self.height = HEIGHT
        self.width = WIDTH
        self.goal = GOAL
        self.obstacles = OBSTACLES

    def is_terminal(self, state):   # Gaol state
        # x, y = state
        return state in self.goal


    def is_out_of_boundary(self, state):
        x, y = state
        if x < 0 or x >= self.height or y < 0 or y >= self.width:
            return True
        else:
            return False


    def is_on_obstacle(self, state):
        if state in self.obstacles:
            return True
        else:
            return False

    def reward(self, state, motion, next_state):
        if self.is_terminal(state):
            return 0
        else:
            return -1


    def interaction(self, state, motion):
        if self.is_terminal(state):
            next_state = state
        else:
            next_state = (np.array(state) + motion).tolist()

        if self.is_out_of_boundary(next_state):
            next_state = state

        if self.is_on_obstacle(next_state):
            next_state = state

        r = self.reward(state,motion,next_state)
        return next_state, r


    def size(self):
        return self.height, self.width
