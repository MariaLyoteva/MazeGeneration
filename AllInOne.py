from matplotlib import pyplot
import random
import math


class Cell:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.connections = []
        self.neighbors = []
        self.isCurrent = False
        self.isAvailable = True
        self.set_id = None
        self.visited = False

    def printCoordinates(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"

    def addNeighbor(self, another_cell):
        self.neighbors.append(another_cell)
        another_cell.neighbors.append(self)

    def addConnection(self, another_cell):
        if another_cell in self.neighbors:
            self.connections.append(another_cell)
            another_cell.connections.append(self)

    def removeConnection(self, another_cell):
        if another_cell in self.connections:
            self.connections.remove(another_cell)
            another_cell.connections.remove(self)

    def hasUnvisitedNeighbor(self):
        for neighbor in self.neighbors:
            if not neighbor.visited:
                return True
        return False

    def getRandomUnvisitedNeighbor(self):
        unvisited_neighbors = [neighbor for neighbor in self.neighbors if not neighbor.visited]
        if unvisited_neighbors:
            return random.choice(unvisited_neighbors)
        return None

    def hasConnectionToRow(self, row):
        for cell in self.connections:
            if cell.y == row:
                return True
        return False

    def hasConnectionToColumn(self, column):
        for cell in self.connections:
            if cell.x == column:
                return True
        return False







class Grid:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.cells = [[Cell(x, y) for y in range(cols)] for x in range(rows)]

    def getCells(self):
        all_cells = []
        for row in self.cells:
            all_cells.extend(row)
        return all_cells

    def printCells(self):
        for r in range(self.rows):
            for c in range(self.cols):
                print(self.cells[r][c].printCoordinates(), end=" ")
            print()
        print()

    def addNeighbors(self):
        for r in range(self.rows):
            for c in range(self.cols - 1):
                self.cells[r][c].addNeighbor(self.cells[r][c + 1])

        for r in range(self.rows - 1):
            for c in range(self.cols):
                self.cells[r][c].addNeighbor(self.cells[r + 1][c])

    def removeWalls(self):
        for r in range(self.rows):
            for c in range(self.cols - 1):
                self.cells[r][c].addConnection(self.cells[r][c + 1])

        for r in range(self.rows - 1):
            for c in range(self.cols):
                self.cells[r][c].addConnection(self.cells[r + 1][c])

    def printNeighbors(self):
        for r in range(self.rows):
            for c in range(self.cols):
                print("[", end="")
                for n in self.cells[r][c].neighbors:
                    print(n.printCoordinates(), end=",")
                print("] ", end="")
            print()
        print()

    def visualize(self):
        vis = []

        vis.append([1] * (self.cols * 2 + 1))

        for r in range(self.rows - 1):
            curr_row = [1]
            next_row = [1]
            for c in range(self.cols - 1):
                curr_row.append(0)
                if self.cells[r][c] in self.cells[r][c + 1].connections:
                    curr_row.append(0)
                else:
                    curr_row.append(1)
                if self.cells[r][c] in self.cells[r + 1][c].connections:
                    next_row.append(0)
                else:
                    next_row.append(1)
                next_row.append(1)
            curr_row.append(0)
            curr_row.append(1)
            next_row.append(0)
            next_row.append(1)
            vis.append(curr_row)
            vis.append(next_row)

        last_row = [1]
        for n in range(self.cols - 1):
            last_row.append(0)
            if self.cells[self.rows - 1][n] in self.cells[self.rows - 1][n + 1].connections:
                last_row.append(0)
            else:
                last_row.append(1)
        last_row.append(0)
        last_row.append(1)

        vis.append(last_row)
        vis.append([1] * (self.cols * 2 + 1))

        for n in range(self.rows - 1):
            if self.cells[n][self.cols - 1] not in self.cells[n + 1][self.cols - 1].connections:
                vis[2 * (n + 1)][2 * self.cols - 1] = 1

        pyplot.figure(figsize=(5, 5))
        pyplot.imshow(vis, cmap='gray_r')
        pyplot.show()

    def generateDFSmazeRecursiv(self, x, y, visited):
        start = self.cells[x][y]
        visited.add(start)
        neighbors_arr = start.neighbors
        random.shuffle(neighbors_arr)
        for n in neighbors_arr:
            if n not in visited:
                start.addConnection(n)
                self.generateDFSmazeRecursiv(n.x, n.y, visited)

    """
    def generateKruskalMaze(self):
        possiblePairs = set()
        for r in range(self.rows):
            for c in range(self.cols - 1):
                pair = []
                pair.append(self.cells[r][c])
                pair.append(self.cells[r][c+1])
                possiblePairs.add(pair)
        for r in range(self.rows - 1):
            for c in range(self.cols):
                pair = []
                pair.append(self.cells[r][c])
                pair.append(self.cells[r+1][c])
                possiblePairs.add(pair)
    """

    def generateAldousBroderMaze(self, x, y):
        current = self.cells[x][y]
        unvisited = self.rows * self.cols - 1
        while unvisited > 0:
            neighbour = random.choice(current.neighbors)
            if len(neighbour.connections) == 0:
                Cell.addConnection(current, neighbour)
                unvisited -= 1
            current = neighbour
    '''
    def calculate_heuristic_weight(self, cell_a, cell_b,grid):
        self.grid=grid
        # Calculate the Euclidean distance between cell_a and cell_b
        distance = math.sqrt((cell_a.x- cell_b.x) ** 2 + (cell_a.y - cell_b.y) ** 2)
        num_connections_a = len(cell_a.connections)
        num_connections_b = len(cell_b.connections)

        # Calculate the density of connections in the grid
        total_cells = cell_a.grid.rows * cell_a.grid.cols
        density = (num_connections_a + num_connections_b) / (2 * total_cells)

        # Combine the distance, number of connections, and density to calculate the heuristic weight
        weight = distance + num_connections_a + num_connections_b + density

        return weight
   '''

    def ellersAlgorithm(self):
        # Set up initial state
        sets = [{cell} for cell in self.cells[0]]

        # Iterate over remaining rows
        for r in range(1, self.rows):
            new_sets = []
            for cell in self.cells[r]:
                if random.random() < 0.5:
                    neighbor = random.choice(cell.neighbors)
                    sets.append({cell, neighbor})
                    cell.addConnection(neighbor)
                else:
                    new_sets.append({cell})
                    sets[-1].add(cell)

            # Merge sets and create down connections
            for cell_set in sets:
                if len(cell_set) > 1:
                    representative = random.choice(list(cell_set))
                    for cell in cell_set:
                        if cell != representative:
                            cell.addConnection(representative)

            sets = new_sets

        # Clear right connections in the last cell of each set
        for cell_set in sets:
            last_cell = None
            for cell in cell_set:
                if last_cell is not None and last_cell in cell.neighbors:
                    cell.removeConnection(last_cell)
                last_cell = cell

        # Create down connections for cells not connected to the next row
        for r in range(1, self.rows):
            for cell in self.cells[r]:
                if not cell.hasConnectionToRow(r + 1):
                    chosen = random.choice(cell.neighbors)
                    cell.addConnection(chosen)


















    def generateWilsonsMaze(self, x, y):
        end = self.cells[x-1][y-1]
        unvisited = set(self.cells)
        while unvisited:
            curr = self.cells[x][y]
            visited = set()
            current = self.cells[random.randint(0, x-1)][random.randint(0, y-1)]
            visited.add(current)
            while curr != end:
                next = random.choice(current.neighbors)
                if next not in visited:
                    visited.add(next)
                    curr = next
            for cell in visited:
                unvisited.remove(cell)
                cell[0].addConnection(cell[1])

    def generateBinaryTree(self):
        for r in range(self.rows):
            for c in range(self.cols):
                current = self.cells[r][c]
                neighbours = []
                for n in current.neighbors:
                    if n.y == current.y and n.x == current.x + 1:
                        neighbours.append(n)
                    if n.x == current.x and n.y == current.y + 1:
                        neighbours.append(n)
                if neighbours: chosen = random.choice(neighbours)
                current.addConnection(chosen)


    def generationGrowingTree(self):
        for r in range(self.rows):
             for c in range(self.cols):
                current = self.cells[r][c]
                neighbors = current.neighbors

                if neighbors:
                    chosen = random.choice(neighbors)
                    current.addConnection(chosen)

        # Set up initial state



    def wilsonAlgorithm(self):
        unvisited_cells = set(cell for row in self.cells for cell in row)
        start_cell = random.choice(list(unvisited_cells))
        unvisited_cells.remove(start_cell)
        start_cell.visited = True

        while unvisited_cells:
            current_cell = random.choice(list(unvisited_cells))
            path = [current_cell]

            while current_cell.hasUnvisitedNeighbor():
                current_cell = current_cell.getRandomUnvisitedNeighbor()
                path.append(current_cell)

            loop_start = None
            for i, cell in enumerate(path):
                if cell in path[i + 1:]:
                    loop_start = cell
                    break

            if loop_start:
                loop_start.addConnection(path[-1])
                for cell in path[:-1]:
                    cell.visited = True
                    unvisited_cells.remove(cell)
            else:
                for i in range(len(path) - 1):
                    path[i].addConnection(path[i + 1])
                    path[i].visited = True
                    unvisited_cells.remove(path[i])

        #return self.grid


maze= Grid(40, 40)
maze.printCells()
maze.addNeighbors()
#maze.generationGrowingTree()
maze.ellersAlgorithm()
#maze.prim_algorithm(maze)
#grid.printNeighbors()
#grid.generateDFSmazeRecursiv(5, 10, set())
#grid.generateAldousBroderMaze(0, 0)
#maze.generateBinaryTree()
#grid.removeWalls()
#grid.generateWilsonsMaze(0, 0)
maze.visualize()

