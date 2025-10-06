from collections import deque
import math, time
class RabbitEnvironment:
    def __init__(self):
       
        self.start = tuple(['E','E','E','_','W','W','W'])
        self.goal = tuple(['W','W','W','_','E','E','E'])
        self.N = len(self.start)
    def is_goal(self, state):
        """Check if goal reached."""
        return state == self.goal
    def get_neighbors(self, state):
        """Generate valid next states from the current state."""
        neighbors = []
        for i, v in enumerate(state):
            if v == 'E':  
                if i+1 < self.N and state[i+1] == '_': 
                    s = list(state); s[i], s[i+1] = s[i+1], s[i]; neighbors.append(tuple(s))
                if i+2 < self.N and state[i+1] != '_' and state[i+2] == '_':  
                    s = list(state); s[i], s[i+2] = s[i+2], s[i]; neighbors.append(tuple(s))
            elif v == 'W':  
                if i-1 >= 0 and state[i-1] == '_':  
                    s = list(state); s[i], s[i-1] = s[i-1], s[i]; neighbors.append(tuple(s))
                if i-2 >= 0 and state[i-1] != '_' and state[i-2] == '_':  
                    s = list(state); s[i], s[i-2] = s[i-2], s[i]; neighbors.append(tuple(s))
        return list(dict.fromkeys(neighbors))  

    def search_space_size(self):
        """Total unique arrangements of 3E, 3W, and 1 empty."""
        return math.factorial(7)//(math.factorial(3)*math.factorial(3)*math.factorial(1))
class SearchAgent:
    def __init__(self, environment):
        self.env = environment

    def bfs(self):
        start_time = time.time()
        start = self.env.start
        goal_test = self.env.is_goal
        neighbors = self.env.get_neighbors
        frontier = deque([start])
        parent = {start: None}
        nodes_expanded = 0
        max_frontier = 1

        while frontier:
            max_frontier = max(max_frontier, len(frontier))
            state = frontier.popleft()
            nodes_expanded += 1

            if goal_test(state):
                path = []
                cur = state
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                return {
                    "path": path,
                    "nodes_expanded": nodes_expanded,
                    "max_frontier": max_frontier,
                    "time": time.time() - start_time
                }

            for nb in neighbors(state):
                if nb not in parent:
                    parent[nb] = state
                    frontier.append(nb)
        return None

    def dfs(self):
        start_time = time.time()
        start = self.env.start
        goal_test = self.env.is_goal
        neighbors = self.env.get_neighbors
        stack = [start]
        parent = {start: None}
        nodes_expanded = 0
        max_frontier = 1

        while stack:
            max_frontier = max(max_frontier, len(stack))
            state = stack.pop()
            nodes_expanded += 1

            if goal_test(state):
                path = []
                cur = state
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                return {
                    "path": path,
                    "nodes_expanded": nodes_expanded,
                    "max_frontier": max_frontier,
                    "time": time.time() - start_time
                }

            for nb in neighbors(state):
                if nb not in parent:
                    parent[nb] = state
                    stack.append(nb)
        return None
if __name__ == "__main__":
    env = RabbitEnvironment()
    agent = SearchAgent(env)
    print("\nRabbit Leap Problem (Agent–Environment Model)")
    print("Total search space size:", env.search_space_size())
    bfs_res = agent.bfs()
    if bfs_res:
        print("\nBFS Solution:")
        print("Moves:", len(bfs_res['path'])-1,
              "Nodes expanded:", bfs_res['nodes_expanded'],
              "Max frontier:", bfs_res['max_frontier'],
              "Time: {:.6f}s".format(bfs_res['time']))
        for i, state in enumerate(bfs_res['path']):
            print(f"  {i}: {''.join(state)}")
    dfs_res = agent.dfs()
    if dfs_res:
        print("\nDFS Solution:")
        print("Moves:", len(dfs_res['path'])-1,
              "Nodes expanded:", dfs_res['nodes_expanded'],
              "Max frontier:", dfs_res['max_frontier'],
              "Time: {:.6f}s".format(dfs_res['time']))
