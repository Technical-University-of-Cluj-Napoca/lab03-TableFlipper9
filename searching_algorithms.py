from utils import *
from collections import deque
from queue import PriorityQueue
from grid import Grid
from spot import Spot
import math

def bfs(draw: callable, grid: Grid, start: Spot, end: Spot) -> bool:
    """
    Breadth-First Search (BFS) Algorithm.
    Args:
        draw (callable): A function to call to update the Pygame window.
        grid (Grid): The Grid object containing the spots.
        start (Spot): The starting spot.
        end (Spot): The ending spot.
    Returns:
        bool: True if a path is found, False otherwise.
    """
    if start == None or end == None:
        return False

    queue = deque()
    queue.append(start)
    visited = {start}
    came_from = {}

    while queue.count != 0:
        current = queue.popleft()

        if current == end:
            while current in came_from:
                current = came_from[current]
                current.make_path()
                draw()
            end.make_end()
            start.make_start()
            return True
        
        for neighbor in current.neighbors:
            if neighbor not in visited  and not neighbor.is_barrier():
                visited.add(neighbor)
                came_from[neighbor] = current
                queue.append(neighbor)
                neighbor.make_open()

        draw()
        if current != start:
            current.make_closed()
    return False

def dfs(draw: callable, grid: Grid, start: Spot, end: Spot) -> bool:
    """
    Depdth-First Search (DFS) Algorithm.
    Args:
        draw (callable): A function to call to update the Pygame window.
        grid (Grid): The Grid object containing the spots.
        start (Spot): The starting spot.
        end (Spot): The ending spot.
    Returns:
        bool: True if a path is found, False otherwise.
    """
    if start == None or end == None:
        return False

    stack = [start]
    visited = {start}
    came_from = {}

    while len(stack) != 0:
        current = stack.pop()
        if current == end:
            while current in came_from:
                current = came_from[current]
                current.make_path()
                draw()
            end.make_end()
            start.make_start()
            return True
        
        for neighbor in current.neighbors:
            if neighbor not in visited  and not neighbor.is_barrier():
                visited.add(neighbor)
                came_from[neighbor] = current
                stack.append(neighbor)
                neighbor.make_open()

        draw()
        if current != start:
            current.make_closed()
    return False

def h_manhattan_distance(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    """
    Heuristic function for A* algorithm: uses the Manhattan distance between two points.
    Args:
        p1 (tuple[int, int]): The first point (x1, y1).
        p2 (tuple[int, int]): The second point (x2, y2).
    Returns:
        float: The Manhattan distance between p1 and p2.
    """
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)

def h_euclidian_distance(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    """
    Heuristic function for A* algorithm: uses the Euclidian distance between two points.
    Args:
        p1 (tuple[int, int]): The first point (x1, y1).
        p2 (tuple[int, int]): The second point (x2, y2).
    Returns:
        float: The Manhattan distance between p1 and p2.
    """
    x1, y1 = p1
    x2, y2 = p2
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def astar(draw: callable, grid: Grid, start: Spot, end: Spot) -> bool:
    """
    A* Pathfinding Algorithm.
    Args:
        draw (callable): A function to call to update the Pygame window.
        grid (Grid): The Grid object containing the spots.
        start (Spot): The starting spot.
        end (Spot): The ending spot.
    Returns:
        bool: True if a path is found, False otherwise.
    """
    if start == None or end == None:
        return False
    
    count = 0
    open_heap = PriorityQueue()
    open_heap.put((0,count,start))
    came_from = {}
    g_score = {}
    f_score = {}

    for row in grid.grid:
        for col in row:
            g_score[col] = 9999
            f_score[col] = 9999

    g_score[start] = 0
    f_score[start] = h_euclidian_distance(start.get_position(), end.get_position())
    lookup_set ={start}

    while not open_heap.empty():
        current = open_heap.get()[2]
        lookup_set.remove(current)

        if current == end:
            while current in came_from:
                current = came_from[current]
                current.make_path()
                draw()
            end.make_end()
            start.make_start()
            return True
        
        for neighbor in current.neighbors:
            if neighbor.is_barrier():
                continue
            tentative_g = g_score[current] + 1 
            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + h_euclidian_distance(neighbor.get_position(), end.get_position())
                if neighbor not in lookup_set:
                    count += 1
                    open_heap.put((f_score[neighbor],count,neighbor))
                    lookup_set.add(neighbor)
                    neighbor.make_open()
        draw()
        if current != start:
            current.make_closed()
    return False

# and the others algorithms...
# ▢ Depth-Limited Search (DLS)
# ▢ Uninformed Cost Search (UCS)
# ▢ Greedy Search
# ▢ Iterative Deepening Search/Iterative Deepening Depth-First Search (IDS/IDDFS)
# ▢ Iterative Deepening A* (IDA)
# Assume that each edge (graph weight) equalss

def dls(draw: callable, grid: Grid, start: Spot, end: Spot) -> bool:
    if start == None or end == None:
        return False

    limit = 10
    depth = {}
    depth[start] = 0
    stack = [start]
    visited = {start}
    came_from = {}

    while len(stack):
        current = stack.pop()
        if current == end:
            while current in came_from:
                current = came_from[current]
                current.make_path()
                draw()
            end.make_end()
            start.make_start()
            return True
        
        if depth[current] < limit:
            for neighbor in current.neighbors:
                if neighbor not in visited  and not neighbor.is_barrier():
                    visited.add(neighbor)
                    came_from[neighbor] = current
                    depth[neighbor] = depth[current] + 1
                    stack.append(neighbor)
                    neighbor.make_open()

        draw()
        if current != start:
            current.make_closed()
    return False

def ucs(draw: callable, grid: Grid, start: Spot, end: Spot) -> bool:
    if start is None or end is None:
        return False
    
    open_heap = PriorityQueue()
    open_heap.put((0,start))
    came_from = {}
    cost = {}

    for row in grid.grid:
        for col in row:
            cost[col] = 9999

    cost[start] = 0
    visited = {start}

    while not open_heap.empty():
        current_cost, current = open_heap.get()

        if current.get_position() == end.get_position():
            while current in came_from:
                current = came_from[current]
                current.make_path()
                draw()
            end.make_end()
            start.make_start()
            return True
        
        for neighbor in current.neighbors:
            if neighbor.is_barrier():
                continue
            new_cost = cost[current] + 1
            if new_cost < cost[neighbor]:
                cost[neighbor] = new_cost
                came_from[neighbor] = current
                if neighbor not in visited:
                    open_heap.put((new_cost, neighbor))
                    visited.add(neighbor)
                    neighbor.make_open()
        
        draw()
        if current != start:
            current.make_closed()
    return False

def greedy(draw: callable, grid: Grid, start: Spot, end: Spot) -> bool:
    if start is None or end is None:
        return False
    
    heuristic = h_euclidian_distance
    open_heap = PriorityQueue()
    open_heap.put((heuristic(start.get_position(), end.get_position()), start))
    came_from = {}
    visited = {start}

    while not open_heap.empty():
        draw()
        urrent_cost, current = open_heap.get()

        if current == end:
            while current in came_from:
                current = came_from[current]
                current.make_path()
                draw()
            end.make_end()
            start.make_start()
            return True

        for neighbor in current.neighbors:
            if neighbor not in visited and not neighbor.is_barrier():
                visited.add(neighbor)
                came_from[neighbor] = current
                h_val = heuristic(neighbor.get_position(), end.get_position())
                open_heap.put((h_val, neighbor))
                neighbor.make_open()

        draw()
        if current != start:
            current.make_closed()

    return False

def ids(draw: callable, grid: Grid, start: Spot, end: Spot):
    
    def dls_helper(draw: callable, grid: Grid, start: Spot, end: Spot, limit: int) -> bool:
        if start == None or end == None:
            return False

        depth = {}
        depth[start] = 0
        stack = [start]
        visited = {start}
        came_from = {}

        while len(stack):
            current = stack.pop()
            if current == end:
                while current in came_from:
                    current = came_from[current]
                    current.make_path()
                    draw()
                end.make_end()
                start.make_start()
                return True
            
            if depth[current] < limit:
                for neighbor in current.neighbors:
                    if neighbor not in visited  and not neighbor.is_barrier():
                        visited.add(neighbor)
                        came_from[neighbor] = current
                        depth[neighbor] = depth[current] + 1
                        stack.append(neighbor)
                        neighbor.make_open()

            draw()
            if current != start:
                current.make_closed()
        return False
    
    max_depth = 20
    for depth in range(max_depth):
        if dls_helper(draw, grid, start, end, depth):
            return True
    return False

def ida_star(draw: callable, grid: Grid, start: Spot, end: Spot) -> bool:

    heuristic = h_manhattan_distance

    if start is None or end is None:
            return False

    def dfs_with_bound(bound: float) -> tuple[dict | None, float]:
        stack = [(start, 0)] 
        came_from = {}
        visited = {start}
        next_bound = math.inf

        while stack:
            draw()
            current, g = stack.pop()

            f = g + heuristic(current.get_position(), end.get_position())
            if f > bound:
                next_bound = min(next_bound, f)
                continue

            if current == end:
                return came_from, 0

            if current != start:
                current.make_closed()

            for neighbor in current.neighbors:
                if not neighbor.is_barrier() and neighbor not in visited:
                    visited.add(neighbor)
                    came_from[neighbor] = current
                    neighbor.make_open()
                    stack.append((neighbor, g + 1))
                    draw()

        return None, next_bound

    bound = heuristic(start.get_position(), end.get_position())

    while True:
        for row in grid.grid:
            for cell in row:
                if cell.is_closed() or cell.is_open():
                    cell.reset()

        came_from, next_bound = dfs_with_bound(bound)
        draw()

        if came_from is not None:
            current = end
            while current.get_position() != start.get_position():
                current = came_from[current]
                current.make_path()
                draw()
            end.make_end()
            start.make_start()
            return True

        if next_bound == math.inf:
            return False

        bound = next_bound