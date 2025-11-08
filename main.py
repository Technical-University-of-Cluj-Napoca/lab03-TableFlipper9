from utils import *
from grid import Grid
from searching_algorithms import *

if __name__ == "__main__":
    # setting up how big will be the display window
    WIN = pygame.display.set_mode((WIDTH, HEIGHT))

    # set a caption for the window
    pygame.display.set_caption("Path Visualizing Algorithm")

    ROWS = 50  # number of rows
    COLS = 50  # number of columns
    grid = Grid(WIN, ROWS, COLS, WIDTH, HEIGHT)

    start = None
    end = None

    # flags for running the main loop
    run = True
    started = False

    algotithms = [        
        ("DFS", dfs),
        ("BFS", bfs),
        ("A*", astar),
        ("UCS", ucs),
        ("Greedy", greedy),
        ("DLS", dls)]
    algo_index = 0

    pygame.font.init()
    font = pygame.font.SysFont("consolas",24)
    def draw_label():
        algo_name = algotithms[algo_index][0]
        text_surface = font.render(f"Algorithm : {algo_name}",True,(255,0,0))
        WIN.blit(text_surface, (WIDTH - text_surface.get_width() -20,20))

    while run:
        grid.draw()  # draw the grid and its spots
        draw_label()
        pygame.display.update()

        for event in pygame.event.get():
            # verify what events happened
            if event.type == pygame.QUIT:
                run = False

            if started:
                # do not allow any other interaction if the algorithm has started
                continue  # ignore other events if algorithm started

            if pygame.mouse.get_pressed()[0]:  # LEFT CLICK
                pos = pygame.mouse.get_pos()
                row, col = grid.get_clicked_pos(pos)

                if row >= ROWS or row < 0 or col >= COLS or col < 0:
                    continue  # ignore clicks outside the grid

                spot = grid.grid[row][col]
                if not start and spot != end:
                    start = spot
                    start.make_start()
                elif not end and spot != start:
                    end = spot
                    end.make_end()
                elif spot != end and spot != start:
                    spot.make_barrier()

            elif pygame.mouse.get_pressed()[2]:  # RIGHT CLICK
                pos = pygame.mouse.get_pos()
                row, col = grid.get_clicked_pos(pos)
                spot = grid.grid[row][col]
                spot.reset()

                if spot == start:
                    start = None
                elif spot == end:
                    end = None

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and not started:
                    # run the algorithm
                    for row in grid.grid:
                        for spot in row:
                            spot.update_neighbors(grid.grid)
                    started = True
                    current_algo = algotithms[algo_index][1]
                    current_algo(lambda: grid.draw(), grid, start, end)
                    # ... and the others?
                    started = False

                if event.key == pygame.K_a:
                    algo_index = (algo_index+1)%len(algotithms)

                if event.key == pygame.K_c:
                    print("Clearing the grid...")
                    start = None
                    end = None
                    grid.reset()
    pygame.quit()
