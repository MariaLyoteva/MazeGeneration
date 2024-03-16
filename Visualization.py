import pygame
import time
from colours import *

class Visualization():
    def __init__(self, grid, screen_height, animation_speed, background_color, maze_background_color, maze_tunnel_color):
        self.grid = grid
        self.screen_height = screen_height
        self.animation_speed = animation_speed
        self.background_color = background_color
        self.maze_background_color = maze_background_color
        self.maze_tunnel_color = maze_tunnel_color

        pygame.init()
        self.screen = 0
        self.screen_width = screen_height * 8/5
        self.square_size = self.screen_height / (2 * grid.rows + 1) if grid.rows > grid.cols else self.screen_height / (
                    2 * grid.cols + 1)
        self.font_size = screen_height / 20
        self.font = pygame.font.Font(None, int(self.screen_height / 20))

        #coordinates of all elements on the GUI
        self.coordinates_animate_generation = [1/4, 1/10, 31/30, 1/30]
        self.coordinates_display_generation = [1/4, 1/10, 79/60, 1/30]
        self.coordinates_animate_solution = [1/4, 1/10, 31/30, 52/75]
        self.coordinates_display_solution = [1/4, 1/10, 79/60, 52/75]
        self.coordinates_dfs = [157/600, 1/20, 31/30, 25/150]
        self.coordinates_kru = [157/600, 1/20, 31/30, 34/150]
        self.coordinates_ab = [157/600, 1/20, 31/30, 43/150]
        self.coordinates_bt = [157/600, 1/20, 31/30, 52/150]
        self.coordinates_p = [157/600, 1/20, 31/30 + 163/600, 25/150]
        self.coordinates_sw = [157/600, 1/20, 31/30 + 163/600, 34/150]
        self.coordinates_hk = [157/600, 1/20, 31/30 + 163/600, 43/150]
        self.coordinates_w = [157/600, 1/20, 31/30 + 163/600, 52/150]

        self.coordinates_speed_text = [31/30, 43/100 + 1/80]
        self.coordinates_rows_text = [31/30, 49/100 + 1/80]
        self.coordinates_cols_text = [31/30, 55/100 + 1/80]
        self.coordinates_startpoint_generation_text = [31/30, 61/100 + 1/80]
        self.coordinates_start_solution_text = [31/30, 124/150 + 1/80]
        self.coordinates_goal_solution_text = [31/30, 133/150 + 1/80]

        self.coordinates_speed_box = [157/600, 1/20, 783/600, 43/100]
        self.coordinates_rows_box = [157/600, 1/20, 783/600, 49/100]
        self.coordinates_cols_box = [157/600, 1/20, 783/600, 55/100]
        self.coordinates_startpoint_generation_box_x = [151/1200, 1/20, 783/600, 61/100]
        self.coordinates_startpoint_generation_box_y = [151/1200, 1/20, 3461/2400, 61/100]
        self.coordinates_solution_start_box_x = [151/1200, 1/20, 783/600, 124/150]
        self.coordinates_solution_start_box_y = [151/1200, 1/20, 3461/2400, 124/150]
        self.coordinates_solution_goal_box_x = [151/1200, 1/20, 783/600, 133/150]
        self.coordinates_solution_goal_box_y = [151/1200, 1/20, 3461/2400, 133/150]

    def draw_button(self, coord, text, color_text, color_button):
        h  = self.screen_height
        pygame.draw.rect(self.screen, color_button, (coord[2]*h, coord[3]*h, coord[0]*h, coord[1]*h))
        font = pygame.font.SysFont(None, int(self.font_size))
        text = font.render(text, True, color_text)
        text_rect = text.get_rect(center=(coord[2]*h + coord[0]/2*h, coord[3]*h + coord[1]/2*h))
        self.screen.blit(text, text_rect)

    def draw_input_box(self, coord, color_box):
        h = self.screen_height
        pygame.draw.rect(self.screen, color_box, (coord[2]*h, coord[3]*h, coord[0]*h, coord[1]*h), 2)

    def draw_text(self, font, text, coord):
        h = self.screen_height
        textField = font.render(text, False, BLACK)
        self.screen.blit(textField, (coord[0]*h, coord[1]*h))

    def initializeWindow(self):
        h = self.screen_height

        # Create the screen
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Maze generation")
        # Fill the screen with white color
        self.screen.fill(self.background_color)

        pygame.draw.rect(self.screen, self.maze_background_color, (0, 0, h, h))

        self.draw_button(self.coordinates_animate_generation, "animate", BLACK, RED)
        self.draw_button(self.coordinates_display_generation, "display", BLACK, RED)
        self.draw_button(self.coordinates_animate_solution, "animate", BLACK, GREEN)
        self.draw_button(self.coordinates_display_solution, "display", BLACK, GREEN)
        self.draw_button(self.coordinates_dfs, "DFS", WHITE, BLACK)
        self.draw_button(self.coordinates_kru, "Kruskal", WHITE, BLACK)
        self.draw_button(self.coordinates_ab, "Aldous-Broder", WHITE, BLACK)
        self.draw_button(self.coordinates_bt, "Binary tree", WHITE, BLACK)
        self.draw_button(self.coordinates_p, "Prims", WHITE, BLACK)
        self.draw_button(self.coordinates_sw, "Sidewinder", WHITE, BLACK)
        self.draw_button(self.coordinates_hk, "Hunt and kill", WHITE, BLACK)
        self.draw_button(self.coordinates_w, "Wilson", WHITE, BLACK)

        self.draw_text(self.font, "Speed: " + str(self.animation_speed), self.coordinates_speed_text)
        self.draw_text(self.font, "Rows: " + str(self.grid.rows), self.coordinates_rows_text)
        self.draw_text(self.font, "Columns: " + str(self.grid.cols), self.coordinates_cols_text)
        self.draw_text(self.font, "Start X, Y: "  + str(self.grid.generation_start_x) + ", " + str(self.grid.generation_start_y), self.coordinates_startpoint_generation_text)
        self.draw_text(self.font, "Start X, Y: " + str(self.grid.solution_start_x) + ", " + str(self.grid.solution_start_y), self.coordinates_start_solution_text)
        self.draw_text(self.font, "Goal X, Y: " + str(self.grid.solution_goal_x) + ", " + str(self.grid.solution_goal_y), self.coordinates_goal_solution_text)

        self.draw_input_box(self.coordinates_speed_box, BLACK)
        self.draw_input_box(self.coordinates_rows_box, BLACK)
        self.draw_input_box(self.coordinates_cols_box, BLACK)
        self.draw_input_box(self.coordinates_startpoint_generation_box_x, BLACK)
        self.draw_input_box(self.coordinates_startpoint_generation_box_y, BLACK)
        self.draw_input_box(self.coordinates_solution_start_box_x, BLACK)
        self.draw_input_box(self.coordinates_solution_start_box_y, BLACK)
        self.draw_input_box(self.coordinates_solution_goal_box_x, BLACK)
        self.draw_input_box(self.coordinates_solution_goal_box_y, BLACK)

        pygame.display.flip()

    # for visualizing the construction of the maze
    def animate(self, x, y, where, type, color):
        if where == "up":
            pygame.draw.rect(self.screen, color, (
                (2 * x + 1) * self.square_size, (2 * y - 1) * self.square_size, self.square_size, 3 * self.square_size),
                             0)  # draw a rectangle twice the width of the cell
            if type == "animation":
                pygame.display.update()  # to animate the wall being removed

        if where == "down":
            pygame.draw.rect(self.screen, color, (
                (2 * x + 1) * self.square_size, (2 * y + 1) * self.square_size, self.square_size, 3 * self.square_size),
                             0)
            if type == "animation":
                pygame.display.update()  # to animate the wall being removed

        if where == "left":
            pygame.draw.rect(self.screen, color, (
                (2 * x - 1) * self.square_size, (2 * y + 1) * self.square_size, 3 * self.square_size, self.square_size),
                             0)
            if type == "animation":
                pygame.display.update()  # to animate the wall being removed

        if where == "right":
            pygame.draw.rect(self.screen, color, (
                (2 * x + 1) * self.square_size, (2 * y + 1) * self.square_size, 3 * self.square_size, self.square_size),
                             0)
            if type == "animation":
                pygame.display.update()  # to animate the wall being removed

    def whereIsTheNeighbor(self, a, b):
        if b in a.neighbors:
            if b.x == a.x:
                if b.y < a.y:
                    return "left"
                else:
                    return "right"
            else:
                if b.x < a.x:
                    return "up"
                else:
                    return "down"

    # to choose which algorithm to use
    def visualizeMaze(self, text, type):
        x = self.grid.generation_start_x
        y = self.grid.generation_start_y
        visualizationArray = []
        if text == "DFS":
            visualizationArray = self.grid.generateDFSmazeIterative(x, y)
        if text == "AB":
            visualizationArray = self.grid.generateAldousBroderMaze(x, y)
        if text == "KRU":
            visualizationArray = self.grid.generateKruskalMaze()
        if text == "BT":
            visualizationArray = self.grid.generateBinaryTreeMaze()
        if text == "P":
            visualizationArray = self.grid.generatePrimsMaze(x, y)
        if text == "HK":
            visualizationArray = self.grid.generateHuntAndKillMaze(x, y)
        if text == "SW":
            visualizationArray = self.grid.generateSidewinderMaze()
        if text == "W":
            visualizationArray = self.grid.generateWilsonsMaze()
        # visualizationArray = grid.generateKruskalMaze()
        for n in visualizationArray:
            current = n[0]
            next = n[1]
            where = self.whereIsTheNeighbor(current, next)
            self.animate(current.y, current.x, where, type, WHITE)
            if type == "animation":
                time.sleep(self.animation_speed)

        pygame.display.update()

    def visualizeSolution(self, start_x, start_y, goal_x, goal_y, type, color, show):
        path = self.grid.solveDFS(start_x, start_y, goal_x, goal_y)
        for n in range(len(path)-1):
            where = self.whereIsTheNeighbor(path[n], path[n+1])
            self.animate(path[n].y, path[n].x, where, type, color)
            if type == "animation":
                time.sleep(self.animation_speed)
        if show:
            pygame.display.update()

    def button_clicked(self, coord, button_text):
        #h = self.screen_height
        self.draw_button(coord, button_text, BLACK, WHITE)
        pygame.display.flip()
        self.draw_button(coord, button_text, WHITE, BLACK)

    def createRect(self, coord):
        h = self.screen_height
        return pygame.Rect(h*coord[2], h*coord[3], h*coord[0], h*coord[1])

    def updateBox(self, inputTextBox, box_rect):
        text_surface = self.font.render(inputTextBox, True, BLACK)
        self.screen.blit(text_surface, (box_rect.x + 5, box_rect.y + 5))
        pygame.display.flip()

    def run(self):
        h = self.screen_height
        self.initializeWindow()

        #initialize default values
        currentMazeAlgorithm = "DFS"
        running = True
        boxSpeedActive = False
        boxRowsActive = False
        boxColsActive = False
        boxGeneStartXactive = False
        boxGeneStartYactive = False
        boxSoluStartXactive = False
        boxSoluStartYactive = False
        boxSoluGoalXactive = False
        boxSoluGoalYactive = False
        inputSpeedBox = ""
        inputTextBoxRows = ""
        inputTextBoxCols = ""
        inputTextBoxGeneX = ""
        inputTextBoxGeneY = ""
        inputTextBoxSoluStartX = ""
        inputTextBoxSoluStartY = ""
        inputTextBoxSoluGoalX = ""
        inputTextBoxSoluGoalY = ""
        new_speed = 0.005
        new_rows = 10
        new_cols = 10
        new_gene_start_x = 0
        new_gene_start_y = 0
        new_solu_start_x = 0
        new_solu_start_y = 0
        new_solu_goal_x = self.grid.rows-1
        new_solu_goal_y = self.grid.cols-1
        button_animate_generation_rect = self.createRect(self.coordinates_animate_generation)
        button_display_generation_rect = self.createRect(self.coordinates_display_generation)
        button_animate_solution_rect = self.createRect(self.coordinates_animate_solution)
        button_display_solution_rect = self.createRect(self.coordinates_display_solution)
        button_DFS_rect = self.createRect(self.coordinates_dfs)
        button_KRU_rect = self.createRect(self.coordinates_kru)
        button_AB_rect = self.createRect(self.coordinates_ab)
        button_BT_rect = self.createRect(self.coordinates_bt)
        button_P_rect = self.createRect(self.coordinates_p)
        button_SW_rect = self.createRect(self.coordinates_sw)
        button_HK_rect = self.createRect(self.coordinates_hk)
        button_W_rect = self.createRect(self.coordinates_w)

        box_speed_rect = self.createRect(self.coordinates_speed_box)
        box_rows_rect = self.createRect(self.coordinates_rows_box)
        box_cols_rect = self.createRect(self.coordinates_cols_box)
        box_start_x_gene_rect = self.createRect(self.coordinates_startpoint_generation_box_x)
        box_start_y_gene_rect = self.createRect(self.coordinates_startpoint_generation_box_y)
        box_start_x_solu_rect = self.createRect(self.coordinates_solution_start_box_x)
        box_start_y_solu_rect = self.createRect(self.coordinates_solution_start_box_y)
        box_goal_x_solu_rect = self.createRect(self.coordinates_solution_goal_box_x)
        box_goal_y_solu_rect = self.createRect(self.coordinates_solution_goal_box_y)

        #main loop
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()
                    if event.button == 1:
                        #what to be executed in case each of the buttons and text boxes is clicked
                        if button_animate_generation_rect.collidepoint(mouse_pos):
                            self.animation_speed = new_speed
                            self.grid.reinitialize(new_rows, new_cols, new_gene_start_x, new_gene_start_y)
                            self.square_size = self.screen_height / (
                                        2 * self.grid.rows + 1) if self.grid.rows > self.grid.cols else self.screen_height / (
                                        2 * self.grid.cols + 1)
                            pygame.draw.rect(self.screen, self.maze_background_color,
                                             (0, 0, self.screen_height,
                                              self.screen_height))
                            self.visualizeMaze(currentMazeAlgorithm, "animation")
                            pygame.display.update()
                        elif button_display_generation_rect.collidepoint(mouse_pos):
                            self.grid.reinitialize(new_rows, new_cols, new_gene_start_x, new_gene_start_y)
                            self.square_size = self.screen_height / (
                                    2 * self.grid.rows + 1) if self.grid.rows > self.grid.cols else self.screen_height / (
                                    2 * self.grid.cols + 1)
                            pygame.draw.rect(self.screen, self.maze_background_color,
                                             (0, 0, self.screen_height,
                                              self.screen_height))
                            self.visualizeMaze(currentMazeAlgorithm, "display")
                        elif button_animate_solution_rect.collidepoint(mouse_pos):
                            self.grid.reinitializeSolutionCoord(new_solu_start_x, new_solu_start_y, new_solu_goal_x, new_solu_goal_y)
                            self.visualizeSolution(new_solu_start_x, new_solu_start_y, new_solu_goal_x, new_solu_goal_y, "animation", GREEN, True)
                            self.visualizeSolution(new_solu_start_x, new_solu_start_y, new_solu_goal_x, new_solu_goal_y, "display", WHITE, False)
                        elif button_display_solution_rect.collidepoint(mouse_pos):
                            self.grid.reinitializeSolutionCoord(new_solu_start_x, new_solu_start_y, new_solu_goal_x, new_solu_goal_y)
                            self.visualizeSolution(new_solu_start_x, new_solu_start_y, new_solu_goal_x, new_solu_goal_y, "display", GREEN, True)
                            self.visualizeSolution(new_solu_start_x, new_solu_start_y, new_solu_goal_x, new_solu_goal_y, "display", WHITE, False)
                        elif button_DFS_rect.collidepoint(mouse_pos):
                            currentMazeAlgorithm = "DFS"
                            self.button_clicked(self.coordinates_dfs, "DFS")
                        elif button_KRU_rect.collidepoint(mouse_pos):
                            currentMazeAlgorithm = "KRU"
                            self.button_clicked(self.coordinates_kru, "Kruskal")
                        elif button_AB_rect.collidepoint(mouse_pos):
                            currentMazeAlgorithm = "AB"
                            self.button_clicked(self.coordinates_ab, "Aldous-Broder")
                        elif button_BT_rect.collidepoint(mouse_pos):
                            currentMazeAlgorithm = "BT"
                            self.button_clicked(self.coordinates_bt, "Binary tree")
                        elif button_P_rect.collidepoint(mouse_pos):
                            currentMazeAlgorithm = "P"
                            self.button_clicked(self.coordinates_p, "Prims")
                        elif button_SW_rect.collidepoint(mouse_pos):
                            currentMazeAlgorithm = "SW"
                            self.button_clicked(self.coordinates_sw, "Sidewinder")
                        elif button_HK_rect.collidepoint(mouse_pos):
                            currentMazeAlgorithm = "HK"
                            self.button_clicked(self.coordinates_hk, "Hunt and kill")
                        elif button_W_rect.collidepoint(mouse_pos):
                            currentMazeAlgorithm = "W"
                            self.button_clicked(self.coordinates_w, "Wilson")
                        elif box_speed_rect.collidepoint(mouse_pos):
                            boxSpeedActive = not boxSpeedActive
                            if boxSpeedActive:
                                self.draw_input_box(self.coordinates_speed_box, RED)
                            else:
                                self.draw_input_box(self.coordinates_speed_box, BLACK)
                            pygame.display.flip()
                        elif box_rows_rect.collidepoint(mouse_pos):
                            boxRowsActive = not boxRowsActive
                            if boxRowsActive:
                                self.draw_input_box(self.coordinates_rows_box, RED)
                            else:
                                self.draw_input_box(self.coordinates_rows_box, BLACK)
                            pygame.display.flip()
                        elif box_cols_rect.collidepoint(mouse_pos):
                            boxColsActive = not boxColsActive
                            if boxColsActive:
                                self.draw_input_box(self.coordinates_cols_box, RED)
                            else:
                                self.draw_input_box(self.coordinates_cols_box, BLACK)
                            pygame.display.flip()
                        elif box_start_x_gene_rect.collidepoint(mouse_pos):
                            boxGeneStartXactive = not boxGeneStartXactive
                            if boxGeneStartXactive:
                                self.draw_input_box(self.coordinates_startpoint_generation_box_x, RED)
                            else:
                                self.draw_input_box(self.coordinates_startpoint_generation_box_x, BLACK)
                            pygame.display.flip()
                        elif box_start_y_gene_rect.collidepoint(mouse_pos):
                            boxGeneStartYactive = not boxGeneStartYactive
                            if boxGeneStartYactive:
                                self.draw_input_box(self.coordinates_startpoint_generation_box_y, RED)
                            else:
                                self.draw_input_box(self.coordinates_startpoint_generation_box_y, BLACK)
                            pygame.display.flip()
                        elif box_start_x_solu_rect.collidepoint(mouse_pos):
                            boxSoluStartXactive = not boxSoluStartXactive
                            if boxSoluStartXactive:
                                self.draw_input_box(self.coordinates_solution_start_box_x, RED)
                            else:
                                self.draw_input_box(self.coordinates_solution_start_box_x, BLACK)
                            pygame.display.flip()
                        elif box_start_y_solu_rect.collidepoint(mouse_pos):
                            boxSoluStartYactive = not boxSoluStartYactive
                            if boxSoluStartYactive:
                                self.draw_input_box(self.coordinates_solution_start_box_y, RED)
                            else:
                                self.draw_input_box(self.coordinates_solution_start_box_y, BLACK)
                            pygame.display.flip()
                        elif box_goal_x_solu_rect.collidepoint(mouse_pos):
                            boxSoluGoalXactive = not boxSoluGoalXactive
                            if boxSoluGoalXactive:
                                self.draw_input_box(self.coordinates_solution_goal_box_x, RED)
                            else:
                                self.draw_input_box(self.coordinates_solution_goal_box_x, BLACK)
                            pygame.display.flip()
                        elif box_goal_y_solu_rect.collidepoint(mouse_pos):
                            boxSoluGoalYactive = not boxSoluGoalYactive
                            if boxSoluGoalYactive:
                                self.draw_input_box(self.coordinates_solution_goal_box_y, RED)
                            else:
                                self.draw_input_box(self.coordinates_solution_goal_box_y, BLACK)
                            pygame.display.flip()
                # when enter is clicked (to save the info from the textboxes)
                elif event.type == pygame.KEYDOWN:
                    if boxSpeedActive:
                        if event.key == pygame.K_RETURN:
                            if inputSpeedBox == "max":
                                new_speed = 0.0000000001
                            elif inputSpeedBox == "normal":
                                new_speed = 0.05
                            elif inputSpeedBox == "slow":
                                new_speed = 0.5
                            else:
                                new_speed = float(inputSpeedBox)
                            inputSpeedBox = ""
                            pygame.draw.rect(self.screen, WHITE,(h*783/600, h*43/100, h*157/600, h/20))
                            pygame.draw.rect(self.screen, WHITE, (h*31/30, h*43/100 + 1/80, h*157/600, h/20))
                            self.draw_input_box(self.coordinates_speed_box, BLACK)
                            self.draw_text(self.font, "Speed: " + str(new_speed), self.coordinates_speed_text)
                            boxSpeedActive = not boxSpeedActive
                        elif event.key == pygame.K_BACKSPACE:
                            inputSpeedBox = ""
                            pygame.draw.rect(self.screen, WHITE, (h * 783 / 600, h * 43 / 100, h * 157 / 600, h / 20))
                            self.draw_input_box(self.coordinates_speed_box, RED)
                        else:
                            inputSpeedBox += event.unicode
                        self.updateBox(inputSpeedBox, box_speed_rect)
                    elif boxRowsActive:
                        if event.key == pygame.K_RETURN:
                            if inputTextBoxRows != "":
                                if 2 <= int(inputTextBoxRows) <= self.screen_height/2-1:
                                    new_rows = int(inputTextBoxRows)
                                    self.grid.rows = new_rows
                                inputTextBoxRows = ""
                                pygame.draw.rect(self.screen, WHITE, (h * 31 / 30 + h * 163 / 600, h * 49 / 100, h * 157 / 600, h / 20))
                                pygame.draw.rect(self.screen, WHITE, (h * 31 / 30, h * 49 / 100, h * 157 / 600, h / 20))
                                self.draw_text(self.font, "Rows: " + str(new_rows), self.coordinates_rows_text)
                                pygame.draw.rect(self.screen, WHITE, (h * 31 / 30, h * 133 / 150 + h * 1 / 80, h * 157 / 600, h * 1 / 20))
                                new_solu_goal_x = new_rows - 1
                                self.draw_text(self.font, "Goal X, Y: " + str(new_solu_goal_x) + ", " + str(new_solu_goal_y), self.coordinates_goal_solution_text)
                            boxRowsActive = not boxRowsActive
                            self.draw_input_box(self.coordinates_rows_box, BLACK)
                        elif event.key == pygame.K_BACKSPACE:
                            inputTextBoxRows = ""
                            pygame.draw.rect(self.screen, WHITE,
                                             (h * 31 / 30 + h * 163 / 600, h * 49 / 100, h * 157 / 600, h / 20))
                            self.draw_input_box(self.coordinates_rows_box, RED)
                        else:
                            if event.unicode.isdigit():
                                inputTextBoxRows += event.unicode
                        self.updateBox(inputTextBoxRows, box_rows_rect)
                    elif boxColsActive:
                        if event.key == pygame.K_RETURN:
                            if inputTextBoxCols != "":
                                if 2 <= int(inputTextBoxCols) <= self.screen_height/2-1:
                                    new_cols = int(inputTextBoxCols)
                                    self.grid.rows = new_cols
                                inputTextBoxCols = ""
                                pygame.draw.rect(self.screen, WHITE, (h * 31 / 30 + h * 163 / 600, h * 55 / 100, h * 157 / 600, h / 20))
                                pygame.draw.rect(self.screen, WHITE, (h * 31 / 30, h * 55 / 100, h * 157 / 600, h / 20))
                                self.draw_input_box(self.coordinates_cols_box, BLACK)
                                self.draw_text(self.font, "Cols: " + str(new_cols), self.coordinates_cols_text)
                                pygame.draw.rect(self.screen, WHITE,
                                             (h * 31 / 30, h * 133 / 150 + h * 1 / 80, h * 157 / 600, h * 1 / 20))
                                new_solu_goal_y = new_cols - 1
                                self.draw_text(self.font, "Goal X, Y: " + str(new_solu_goal_x) + ", " + str(new_solu_goal_y), self.coordinates_goal_solution_text)
                            boxColsActive = not boxColsActive
                            self.draw_input_box(self.coordinates_cols_box, BLACK)
                        elif event.key == pygame.K_BACKSPACE:
                            inputTextBoxCols = ""
                            pygame.draw.rect(self.screen, WHITE,
                                             (h * 31 / 30 + h * 163 / 600, h * 55 / 100, h * 157 / 600, h / 20))
                            self.draw_input_box(self.coordinates_cols_box, RED)

                        else:
                            if event.unicode.isdigit():
                                inputTextBoxCols += event.unicode
                        self.updateBox(inputTextBoxCols, box_cols_rect)
                    elif boxGeneStartXactive:
                        if event.key == pygame.K_RETURN:
                            if inputTextBoxGeneX != "":
                                if 0 <= int(inputTextBoxGeneX) <= new_rows-1:
                                    new_gene_start_x = int(inputTextBoxGeneX)
                                inputTextBoxGeneX = ""
                                pygame.draw.rect(self.screen, WHITE, (h * 783/600, h * 61 / 100, h * 151 / 1200, h / 20))
                                pygame.draw.rect(self.screen, WHITE, (h * 31 / 30, h * 61 / 100, h * 157 / 600, h / 20))
                                self.draw_text(self.font, "Start X, Y: " + str(new_gene_start_x) + ", " + str(new_gene_start_y), self.coordinates_startpoint_generation_text)
                            boxGeneStartXactive = not boxGeneStartXactive
                            self.draw_input_box(self.coordinates_startpoint_generation_box_x, BLACK)
                        elif event.key == pygame.K_BACKSPACE:
                            inputTextBoxGeneX = ""
                            pygame.draw.rect(self.screen, WHITE, (h * 783 / 600, h * 61 / 100, h * 151 / 1200, h / 20))
                            self.draw_input_box(self.coordinates_startpoint_generation_box_x, RED)
                        else:
                            if event.unicode.isdigit():
                                inputTextBoxGeneX += event.unicode
                        self.updateBox(inputTextBoxGeneX, box_start_x_gene_rect)
                    elif boxGeneStartYactive:
                        if event.key == pygame.K_RETURN:
                            if inputTextBoxGeneY != "":
                                if 0 <= int(inputTextBoxGeneY) <= new_cols - 1:
                                    new_gene_start_y = int(inputTextBoxGeneY)
                                inputTextBoxGeneY = ""
                                pygame.draw.rect(self.screen, WHITE, (h * 3461/2400, h * 61 / 100, h * 151 / 1200, h / 20))
                                pygame.draw.rect(self.screen, WHITE, (h * 31 / 30, h * 61 / 100, h * 157 / 600, h / 20))
                                self.draw_text(self.font, "Start X, Y: " + str(new_gene_start_x) + ", " + str(new_gene_start_y), self.coordinates_startpoint_generation_text)
                            boxGeneStartYactive = not boxGeneStartYactive
                            self.draw_input_box(self.coordinates_startpoint_generation_box_y, BLACK)
                        elif event.key == pygame.K_BACKSPACE:
                            inputTextBoxGeneY = ""
                            pygame.draw.rect(self.screen, WHITE,
                                             (h * 3461 / 2400, h * 61 / 100, h * 151 / 1200, h / 20))
                            self.draw_input_box(self.coordinates_startpoint_generation_box_y, RED)
                        else:
                            if event.unicode.isdigit():
                                inputTextBoxGeneY += event.unicode
                        self.updateBox(inputTextBoxGeneY, box_start_y_gene_rect)
                    elif boxSoluStartXactive:
                        if event.key == pygame.K_RETURN:
                            if inputTextBoxSoluStartX != "":
                                if 0 <= int(inputTextBoxSoluStartX) <= new_rows-1:
                                    new_solu_start_x = int(inputTextBoxSoluStartX)
                                inputTextBoxSoluStartX = ""
                                pygame.draw.rect(self.screen, WHITE, (h * 783/600, h * 124 / 150, h * 151 / 1200, h / 20))
                                pygame.draw.rect(self.screen, WHITE, (h * 31 / 30, h*124 / 150 + h*1 / 80, h * 157 / 600, h / 20))
                                self.draw_text(self.font, "Start X, Y: " + str(new_solu_start_x) + ", " + str(new_solu_start_y), self.coordinates_start_solution_text)
                            boxSoluStartXactive = not boxSoluStartXactive
                            self.draw_input_box(self.coordinates_solution_start_box_x, BLACK)
                        elif event.key == pygame.K_BACKSPACE:
                            inputTextBoxSoluStartX = ""
                            pygame.draw.rect(self.screen, WHITE, (h * 783 / 600, h * 124 / 150, h * 151 / 1200, h / 20))
                            self.draw_input_box(self.coordinates_solution_start_box_x, RED)
                        else:
                            if event.unicode.isdigit():
                                inputTextBoxSoluStartX += event.unicode
                        self.updateBox(inputTextBoxSoluStartX, box_start_x_solu_rect)
                    elif boxSoluStartYactive:
                        if event.key == pygame.K_RETURN:
                            if inputTextBoxSoluStartY  != "":
                                if 0 <= int(inputTextBoxSoluStartY) <= new_cols - 1:
                                    new_solu_start_y = int(inputTextBoxSoluStartY)
                                inputTextBoxSoluStartY = ""
                                pygame.draw.rect(self.screen, WHITE, (h * 3461/2400, h * 124 / 150, h * 151 / 1200, h / 20))
                                pygame.draw.rect(self.screen, WHITE, (h * 31 / 30, h*124 / 150 + h*1 / 80, h * 157 / 600, h / 20))
                                self.draw_text(self.font, "Start X, Y: " + str(new_solu_start_x) + ", " + str(new_solu_start_y), self.coordinates_start_solution_text)
                            boxSoluStartYactive = not boxSoluStartYactive
                            self.draw_input_box(self.coordinates_solution_start_box_y, BLACK)
                        elif event.key == pygame.K_BACKSPACE:
                            inputTextBoxSoluStartX = ""
                            pygame.draw.rect(self.screen, WHITE,
                                             (h * 3461 / 2400, h * 124 / 150, h * 151 / 1200, h / 20))
                            self.draw_input_box(self.coordinates_solution_start_box_y, RED)
                        else:
                            if event.unicode.isdigit():
                                inputTextBoxSoluStartY += event.unicode
                        self.updateBox(inputTextBoxSoluStartY, box_start_y_solu_rect)
                    elif boxSoluGoalXactive:
                        if event.key == pygame.K_RETURN:
                            if inputTextBoxSoluGoalX != "":
                                if 0 <= int(inputTextBoxSoluGoalX) <= new_rows-1:
                                    new_solu_goal_x = int(inputTextBoxSoluGoalX)
                                inputTextBoxSoluGoalX = ""
                                pygame.draw.rect(self.screen, WHITE, (h*783 / 600, h*133 / 150, h*151 / 1200, h*1 / 20))
                                pygame.draw.rect(self.screen, WHITE, (h*31 / 30, h*133/150 + h*1/80, h * 157 / 600, h*1 / 20))
                                self.draw_text(self.font, "Goal X, Y: " + str(new_solu_goal_x) + ", " + str(new_solu_goal_y), self.coordinates_goal_solution_text)
                            boxSoluGoalXactive = not boxSoluGoalXactive
                            self.draw_input_box(self.coordinates_solution_goal_box_x, BLACK)
                        elif event.key == pygame.K_BACKSPACE:
                            inputTextBoxSoluGoalX = ""
                            pygame.draw.rect(self.screen, WHITE,
                                             (h * 783 / 600, h * 133 / 150, h * 151 / 1200, h * 1 / 20))
                            self.draw_input_box(self.coordinates_solution_goal_box_x, RED)
                        else:
                            if event.unicode.isdigit():
                                inputTextBoxSoluGoalX += event.unicode
                        self.updateBox(inputTextBoxSoluGoalX, box_goal_x_solu_rect)
                    elif boxSoluGoalYactive:
                        if event.key == pygame.K_RETURN:
                            if inputTextBoxSoluGoalY != "":
                                if 0 <= int(inputTextBoxSoluGoalY) <= new_cols-1:
                                    new_solu_goal_y = int(inputTextBoxSoluGoalY)
                                inputTextBoxSoluGoalY = ""
                                pygame.draw.rect(self.screen, WHITE, (h*3461 / 2400, h*133 / 150, h*151 / 1200, h*1 / 20))
                                pygame.draw.rect(self.screen, WHITE, (h*31 / 30, h*133/150 + h*1/80, h * 157 / 600, h*1 / 20))
                                self.draw_text(self.font, "Goal X, Y: " + str(new_solu_goal_x) + ", " + str(new_solu_goal_y), self.coordinates_goal_solution_text)
                            boxSoluGoalYactive = not boxSoluGoalYactive
                            self.draw_input_box(self.coordinates_solution_goal_box_y, BLACK)
                        elif event.key == pygame.K_BACKSPACE:
                            inputTextBoxSoluGoalY = ""
                            pygame.draw.rect(self.screen, WHITE, (h * 3461 / 2400, h * 133 / 150, h * 151 / 1200, h * 1 / 20))
                            self.draw_input_box(self.coordinates_solution_goal_box_y, RED)
                        else:
                            if event.unicode.isdigit():
                                inputTextBoxSoluGoalY += event.unicode
                        self.updateBox(inputTextBoxSoluGoalY, box_goal_y_solu_rect)