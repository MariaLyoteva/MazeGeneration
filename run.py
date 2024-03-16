from Grid import *
from Visualization import *
from colours import *

grid = Grid(10, 10, 0, 0, 0, 0, 9, 9)
visual = Visualization(grid, 600, 0.005, WHITE, BLACK, WHITE)
visual.run()