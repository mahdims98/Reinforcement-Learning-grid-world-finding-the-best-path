import random
import pprint
import io
import numpy as np
import sys
from gym.envs.toy_text import discrete
from colorama import Fore, Back, Style

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


class GridworldEnv(discrete.DiscreteEnv):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, shape=[10, 10]):
        if not isinstance(shape, (list, tuple)) or not len(shape) == 2:
            raise ValueError('shape argument must be a list/tuple of length 2')

        self.shape = shape

        nS = np.prod(shape)
        nA = 4

        MAX_Y = shape[0]
        MAX_X = shape[1]

        P = {}
        grid = np.arange(nS).reshape(shape)
        it = np.nditer(grid, flags=['multi_index'])

        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            # P[s][a] = (prob, next_state, reward, is_done)
            P[s] = {a: [] for a in range(nA)}

            # Reaching target
            is_done = lambda s: s == 0

            # Due to existance of walls, these moves are not available
            no_up = lambda s: s in [44, 45, 63, 64]
            no_down = lambda s: s in [23, 24, 25, 44]
            no_left = lambda s: s in [44, 36, 55]
            no_right = lambda s: s in [32, 42, 52]

            # Barries in the path
            pit = lambda s: s in [13, 62, 41]
            wall = lambda s: s in [43, 33, 53, 34, 35, 54]

            # Rewards of each state
            reward = 1.0 if is_done(s) else 0.0
            reward = -10.0 if pit(s) else reward

            ns_up = s if y == 0 else s - MAX_X
            ns_right = s if x == (MAX_X - 1) else s + 1
            ns_down = s if y == (MAX_Y - 1) else s + MAX_X
            ns_left = s if x == 0 else s - 1

            P[s][UP] = [(1, ns_up, reward, is_done(ns_up))]
            P[s][RIGHT] = [(1.0, ns_right, reward, is_done(ns_right))]
            P[s][DOWN] = [(1, ns_down, reward, is_done(ns_down))]
            P[s][LEFT] = [(1, ns_left, reward, is_done(ns_left))]

            # Terminal state
            if is_done(s):
                P[s][UP] = [(1.0, s, reward, True)]
                P[s][RIGHT] = [(1.0, s, reward, True)]
                P[s][DOWN] = [(1.0, s, reward, True)]
                P[s][LEFT] = [(1.0, s, reward, True)]

            # Encountering walls
            if no_up(s):
                P[s][UP] = [(1.0, s, -3.0, False)]
            if no_down(s):
                P[s][DOWN] = [(1.0, s, -3.0, False)]
            if no_right(s):
                P[s][RIGHT] = [(1.0, s, -3.0, False)]
            if no_left(s):
                P[s][LEFT] = [(1.0, s, -3.0, False)]

            it.iternext()

            # Initial state distribution is uniform
        isd = np.zeros(100)
        isd[44] = 1
        self.P = P

        super(GridworldEnv, self).__init__(nS, nA, P, isd)

    def _render(self, mode='human', close=False):
        if close:
            return

        outfile = io.StringIO() if mode == 'ansi' else sys.stdout

        grid = np.arange(self.nS).reshape(self.shape)
        it = np.nditer(grid, flags=['multi_index'])

        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            if self.s == s and not s == 0:
                output = Back.CYAN + Style.BRIGHT + Fore.WHITE + "  웃 " + Style.RESET_ALL if s not in [13, 62,
                                                                                                       41] else Back.BLUE + Style.BRIGHT + Fore.WHITE + "  웃 " + Style.RESET_ALL
            elif s == 0:
                output = Back.GREEN + Style.BRIGHT + Fore.WHITE + "     " + Style.RESET_ALL if not self.s == 0 else Back.GREEN + Style.BRIGHT + Fore.WHITE + "  웃 " + Style.RESET_ALL
            elif s in [13, 62, 41]:
                output = Back.BLUE + "     " + Style.RESET_ALL
            elif s in [43, 33, 53, 34, 35, 54]:
                output = Back.CYAN + Style.BRIGHT + Fore.BLACK + "  █  " + Style.RESET_ALL
            else:

                output = Back.CYAN + Style.BRIGHT + Fore.BLACK + "  .  " + Style.RESET_ALL

            if x == 0:
                output = output.lstrip()
            if x == self.shape[1] - 1:
                output = output.rstrip()

            outfile.write(output)

            if x == self.shape[1] - 1:
                outfile.write("\n")
            it.iternext()


