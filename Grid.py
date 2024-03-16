import random
import math
from Cell import *

class Grid:
    def __init__(self, rows, cols, generation_start_x, generation_start_y, solution_start_x, solution_start_y, solution_goal_x, solution_goal_y):
        self.rows = rows
        self.cols = cols
        #coordinates of the point from which we will start to build the maze:
        self.generation_start_x = generation_start_x
        self.generation_start_y = generation_start_y
        #coordinates of the points from which and to which we will find the solution to the maze
        self.solution_start_x = solution_start_x
        self.solution_start_y = solution_start_y
        self.solution_goal_x = solution_goal_x
        self.solution_goal_y = solution_goal_y

        #fills the 2-D array with cell objects
        self.cells = [[Cell(x, y) for y in range(cols)] for x in range(rows)]

        #adds each cell's neighbours to its neighbours list
        for r in range(self.rows):
            for c in range(self.cols - 1):
                self.cells[r][c].addNeighbor(self.cells[r][c + 1])
        for r in range(self.rows - 1):
            for c in range(self.cols):
                self.cells[r][c].addNeighbor(self.cells[r + 1][c])

        #fills an array with every pair of neighbour cells
        self.possiblePairs = []
        for r in range(self.rows):
            for c in range(self.cols - 1):
                self.addPairToList(self.cells[r][c], self.cells[r][c+1], self.possiblePairs)
        for r in range(self.rows - 1):
            for c in range(self.cols):
                self.addPairToList(self.cells[r][c], self.cells[r+1][c], self.possiblePairs)

    #we need this for the visualization when we change certain paramenters of the maze
    def reinitialize(self, rows, cols, generation_start_x, generation_start_y):
        self.rows = rows
        self.cols = cols
        self.generation_start_x = generation_start_x
        self.generation_start_y = generation_start_y
        self.cells = [[Cell(x, y) for y in range(self.cols)] for x in range(self.rows)]
        for r in range(self.rows):
            for c in range(self.cols - 1):
                self.cells[r][c].addNeighbor(self.cells[r][c + 1])
        for r in range(self.rows - 1):
            for c in range(self.cols):
                self.cells[r][c].addNeighbor(self.cells[r + 1][c])
        self.possiblePairs = []
        for r in range(self.rows):
            for c in range(self.cols - 1):
                self.addPairToList(self.cells[r][c], self.cells[r][c + 1], self.possiblePairs)
        for r in range(self.rows - 1):
            for c in range(self.cols):
                self.addPairToList(self.cells[r][c], self.cells[r + 1][c], self.possiblePairs)

    # we need to be able to only change the starting and goal point for the solution visualization
    def reinitializeSolutionCoord(self, solution_start_x, solution_start_y, solution_goal_x, solution_goal_y):
        self.solution_start_x = solution_start_x
        self.solution_start_y = solution_start_y
        self.solution_goal_x = solution_goal_x
        self.solution_goal_y = solution_goal_y

    # for adding the connection of the two cells to the return list
    def addPairToList(self, current, next, list):
        pair = [current]
        pair.append(next)
        list.append(pair)

    def generateDFSmazeRecursiv(self, x, y, visited, return_list):
        start = self.cells[x][y]
        visited.add(start)
        neighbors_arr = start.neighbors
        random.shuffle(neighbors_arr)
        for n in neighbors_arr:
            if n not in visited:
                start.addConnection(n)
                self.addPairToList(start, n, return_list)
                self.generateDFSmazeRecursiv(n.x, n.y, visited, return_list)

        return return_list

    def generateDFSmazeIterative(self, x, y):
        return_list = []
        start = self.cells[x][y]
        visited = set()
        visited.add(start)
        stack = [start]

        while stack:
            current = stack[-1]
            neighbors_arr = current.neighbors
            unvisited_neighbors = [n for n in neighbors_arr if n not in visited]

            if unvisited_neighbors:
                next_neighbor = random.choice(unvisited_neighbors)
                current.addConnection(next_neighbor)
                self.addPairToList(current, next_neighbor, return_list)
                visited.add(next_neighbor)
                stack.append(next_neighbor)
            else:
                stack.pop()

        return return_list

    def generateKruskalMaze(self):
        return_list = []
        possiblePairs = self.possiblePairs

        groups = []
        for r in range(self.rows):
            for c in range(self.cols):
                groups.append([self.cells[r][c]])

        while len(groups) > 1:
            pair = random.choice(possiblePairs)
            possiblePairs.remove(pair)
            cell1, cell2 = pair

            group1 = None
            group2 = None
            for group in groups:
                if cell1 in group:
                    group1 = group
                if cell2 in group:
                    group2 = group

            if group1 != group2:
                group1.extend(group2)
                groups.remove(group2)
                cell1.addConnection(cell2)
                self.addPairToList(cell1, cell2, return_list)

        return return_list

    def generateAldousBroderMaze(self, x, y):
        return_list = []
        current = self.cells[x][y]
        unvisited = self.rows * self.cols - 1
        while unvisited > 0:
            neighbour = random.choice(current.neighbors)
            if not neighbour.connections:
                current.addConnection(neighbour)
                self.addPairToList(current, neighbour, return_list)
                unvisited -= 1
            current = neighbour

        return return_list

    def generateBinaryTreeMaze(self):
        return_list = []
        for r in range(self.rows):
            for c in range(self.cols):
                current = self.cells[self.rows - r - 1][self.cols - c - 1]
                neighbours = []
                for n in current.neighbors:
                    if n.y == current.y and n.x == current.x - 1:
                        neighbours.append(n)
                    if n.x == current.x and n.y == current.y - 1:
                        neighbours.append(n)
                if neighbours: chosen = random.choice(neighbours)
                current.addConnection(chosen)
                self.addPairToList(current, chosen, return_list)
        return return_list

    def generatePrimsMaze(self, x, y):
        return_list = []
        visited = set()
        start_cell = self.cells[x][y]
        visited.add(start_cell)

        while len(visited) < self.rows * self.cols:
            # Find the minimum weighted edge connecting a visited cell to an unvisited cell (in our case random)
            min_edge_weight = float('inf')
            min_edge = None
            for cell in visited:
                for neighbor in cell.neighbors:
                    if neighbor not in visited:
                        weight = random.randint(1, 100)
                        if weight < min_edge_weight:
                            min_edge_weight = weight
                            min_edge = (cell, neighbor)

            # Mark the selected edge as visited
            cell_a, cell_b = min_edge
            cell_a.addConnection(cell_b)
            self.addPairToList(cell_a, cell_b, return_list)
            visited.add(cell_b)

        return return_list

    def generateHuntAndKillMaze(self, x, y):
        return_list = []
        current = self.cells[x][y]
        while True:
            # Check for unvisited neighbors
            unvisited_neighbors = [neighbor for neighbor in current.neighbors if not neighbor.connections]
            if unvisited_neighbors:
                next_cell = random.choice(unvisited_neighbors)
                current.addConnection(next_cell)
                self.addPairToList(current, next_cell, return_list)

                current = next_cell
            else:
                # Perform the "hunt" phase
                found = False
                for row in self.cells:
                    for cell in row:
                        if not cell.connections:
                            visited_neighbors = [neighbor for neighbor in cell.neighbors if neighbor.connections]
                            if visited_neighbors:
                                next_cell = random.choice(visited_neighbors)
                                cell.addConnection(next_cell)
                                self.addPairToList(cell, next_cell, return_list)
                                current = cell
                                found = True
                                break
                    if found:
                        break
                else:

                    break

        return return_list

    def generateSidewinderMaze(self):
        return_list = []
        for r in range(self.rows):
            run = []
            for c in range(self.cols):
                current = self.cells[r][c]
                run.append(current)

                should_carve_north = r > 0 and (c == self.cols - 1 or random.choice([True, False]))

                if should_carve_north:
                    cell = random.choice(run)
                    if cell.x > 0:
                        cell.addConnection(self.cells[cell.x - 1][cell.y])
                        self.addPairToList(cell, self.cells[cell.x - 1][cell.y], return_list)
                    run = []
                else:
                    if c < self.cols - 1:
                        current.addConnection(self.cells[r][c + 1])
                        self.addPairToList(current, self.cells[r][c + 1], return_list)
        return return_list

    def generateWilsonsMaze(self):
        return_list = []
        unvisited = []
        for r in range(self.rows):
            for c in range(self.cols):
                unvisited.append(self.cells[r][c])
        random_cell = random.choice(unvisited)
        unvisited.remove(random_cell)
        while len(unvisited) > 0:
            current = random.choice(unvisited)
            path = [current]

            while current in unvisited:
                current = random.choice(current.neighbors)
                if current in path:
                    _index = path.index(current)
                    path = path[:_index + 1]
                else:
                    path.append(current)

            if len(path) > 1:
                for i in range(len(path) - 1):
                    path[i].addConnection(path[i+1])
                    self.addPairToList(path[i], path[i+1], return_list)
                    unvisited.remove(path[i])

        return return_list

    def solveDFS(self, start_x, start_y, end_x, end_y):
        start_cell = self.cells[start_x][start_y]
        end_cell = self.cells[end_x][end_y]
        stack = [(start_cell, [start_cell])]

        while stack:
            current, path = stack.pop()

            if current == end_cell:
                return path

            for neighbor in current.connections:
                if neighbor not in path:
                    stack.append((neighbor, path + [neighbor]))

        return None


