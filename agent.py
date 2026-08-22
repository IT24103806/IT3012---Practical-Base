import random


class SimpleReflexAgent:
    def sense_and_act(self, percept: dict) -> str:
        if percept.get('food_here', False):
            return 'Up'
        if percept.get('wall_ahead', False):
            return 'Left'
        return 'Up'


class ModelBasedAgent:
    def __init__(self):
        self.visited_cells = set()
        self.last_percept = None
        self.last_action = None
        self.actions_pool = ['Up', 'Right', 'Down', 'Left']

    def sense_and_act(self, percept: dict) -> str:
        percept_state = tuple(sorted(percept.items()))
        if self.last_percept == percept_state:
            next_index = (self.actions_pool.index(self.last_action) + 1) % len(self.actions_pool)
            action = self.actions_pool[next_index]
        elif percept.get('food_here', False):
            action = 'Up'
        elif percept.get('wall_ahead', False):
            action = 'Right'
        else:
            action = 'Up'

        self.last_percept = percept_state
        self.last_action = action
        return action


class GreedyGridAgent:
    """A simple agent that tries to move around systematically to clear the grid."""

    def __init__(self):
        self.actions_pool = ['Up', 'Down', 'Left', 'Right']

    def sense_and_act(self, percept: dict) -> str:
        # If standing directly on food, or just wander / move towards coordinates
        pos = percept['agent_pos']
        # Simple heuristic or fallback random sweep
        return random.choice(self.actions_pool)