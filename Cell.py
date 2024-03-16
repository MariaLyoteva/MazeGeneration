class Cell:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.connections = []
        self.neighbors = []

    #cells next to the current cell
    def addNeighbor(self, another_cell):
        self.neighbors.append(another_cell)
        another_cell.neighbors.append(self)

    #cells to which the current cell has a passage
    def addConnection(self, another_cell):
        if another_cell in self.neighbors:
            self.connections.append(another_cell)
            another_cell.connections.append(self)