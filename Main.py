# TODO: Organize Imports
# TODO: Add Physics Monitors and Balances
# TODO: Accelerate/Optimize Calculations
# TODO: Change to Collocated Flow
# TODO: Improve Multigrid F-Cycle
# TODO: Try out Simpler Restriction Operators
# TODO: Suppress Backflow Option


from time import time_ns as tic
from Game import *
from CFD import Flow, Convection

if __name__ == '__main__':
    # Initialize
    pygame.init()
    display = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

    # Create Equations List
    flow = Flow(V_in, 1, W, H, Nx, Ny)
    conv = Convection(flow)
    case = 0
    if case == 0:
        flow.U[:, :] = V_in
        flow.left[:] = 2
        flow.right[:] = 3
        conv.T[:, :] = 0.0
        conv.T[:, 0] = 1.0
        #conv.source[Ny // 4:3 * Ny // 4, Nx // 4:3 * Nx // 4] = True
    elif case == 1:
        flow.bottom[:] = 2
        flow.top[:] = 3
    elif case == 2:
        flow.bottom[:Nx // 4] = 2
        flow.bottom[3*Nx//4:] = 3
        # flow.top[:Nx // 4] = 2
        # flow.top[3*Nx//4:] = 3
        #flow.top[:Nx//4] = 2
        #flow.top[3*Nx//4:] = 3
        skip = Ny // 16
        #flow.solid[skip-1:-skip:skip, Nx//4:3*Nx//4] = True
        #flow.solid[3*Ny//4:, Nx//4:3*Nx//4] = True
    elif case == 3:
        flow.left[:] = 2
        flow.bottom[:] = 2
        flow.right[:] = 3
        flow.top[:] = 3
        flow.U[:, :] = 1.0
        flow.V[:, :] = 1.0
        conv.T[1:-1, 0] = 1.0
        conv.T[0, 1:-1] = 0.0

    flow.initialize()
    conv.initialize()

    # Main Loop
    running = True
    loop_iter = 0
    while running:
        # Iteration Timer
        t0 = tic()

        # Increment Loop Iteration Counter
        loop_iter += 1

        # Detect Quit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Get User Input
        mouse = pygame.mouse.get_pressed()
        if np.any(mouse):
            i, j = get_mouse()
            if mouse[1]:
                flow.reset()
                conv.reset()
            elif not i is None and not j is None:
                if i >= 0 and i < Nx and j >= 0 and j < Ny:
                    if mouse[0]:
                        flow.set_level([[j], [i]], True)
                    elif mouse[2]:
                        flow.set_level([[j], [i]], False)

        # Simulate Physics
        flow.iterate()
        conv.iterate()

        # Refresh Display
        #contour_field = flow.P.copy()
        contour_field = conv.T.copy()
        draw_display(contour_field, rgb, RGB, flow.solid, flow.porous, display)
        quiver_fields = flow.quiver()
        my_quiver_plot(*quiver_fields, display)
        pygame.display.update()

        # Print to Console
        print(f'Total Iteration Time: {(tic() - t0) / 1e6} ms')

        # Stopping Criteria
        print(loop_iter)
        if loop_iter > 100000:
            running = False

    # Quit Window
    pygame.quit()
