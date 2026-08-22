import random
import heapq
from collections import deque


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


class SearchAgent:
    def __init__(self):
        self.plan = []
        self.active_algo = 'BFS'

    def _neighbors(self, position, walls, grid_size):
        width, height = grid_size
        directions = [
            ('Up', (0, 1)),
            ('Right', (1, 0)),
            ('Down', (0, -1)),
            ('Left', (-1, 0))
        ]

        for action, (dx, dy) in directions:
            next_position = (position[0] + dx, position[1] + dy)
            if (
                0 <= next_position[0] < width
                and 0 <= next_position[1] < height
                and next_position not in walls
            ):
                yield action, next_position

    def bfs_search(self, start, goal, walls, grid_size):
        frontier = deque([(start, [])])
        reached = {start}

        while frontier:
            position, path = frontier.popleft()
            if position == goal:
                return path

            for action, next_position in self._neighbors(position, set(walls), grid_size):
                if next_position not in reached:
                    reached.add(next_position)
                    frontier.append((next_position, path + [action]))
        return None

    def dfs_search(self, start, goal, walls, grid_size):
        frontier = [(start, [])]
        reached = {start}

        while frontier:
            position, path = frontier.pop()
            if position == goal:
                return path

            neighbors = list(self._neighbors(position, set(walls), grid_size))
            for action, next_position in reversed(neighbors):
                if next_position not in reached:
                    reached.add(next_position)
                    frontier.append((next_position, path + [action]))
        return None

    def ucs_search(self, start, goal, walls, grid_size):
        frontier = [(0, 0, start, [])]
        reached = {start: 0}
        sequence = 1

        while frontier:
            cost, _, position, path = heapq.heappop(frontier)
            if position == goal:
                return path

            for action, next_position in self._neighbors(position, set(walls), grid_size):
                next_cost = cost + 1
                if next_position not in reached or next_cost < reached[next_position]:
                    reached[next_position] = next_cost
                    heapq.heappush(
                        frontier,
                        (next_cost, sequence, next_position, path + [action])
                    )
                    sequence += 1
        return None

    def sense_and_act(self, percept: dict) -> str:
        if not self.plan:
            start = tuple(percept['agent_pos'])
            foods = [tuple(food) for food in percept['all_food']]
            if not foods:
                return 'Stay'

            target = min(
                foods,
                key=lambda food: abs(food[0] - start[0]) + abs(food[1] - start[1])
            )
            search_methods = {
                'BFS': self.bfs_search,
                'DFS': self.dfs_search,
                'UCS': self.ucs_search
            }
            path = search_methods.get(self.active_algo, self.bfs_search)(
                start,
                target,
                percept['walls'],
                percept['grid_size']
            )
            self.plan = path or []

        return self.plan.pop(0) if self.plan else 'Stay'