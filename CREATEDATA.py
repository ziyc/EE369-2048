from game2048.game import Game
import numpy as np
from game2048.displays import Display

import time
time_start = time.time()

def single_run(size, score_to_win, AgentClass, **kwargs):
    game = Game(size, score_to_win)
    agent = AgentClass(game, display=None, **kwargs)
    agent.play(verbose=False)


if __name__ == '__main__':
    GAME_SIZE = 4
    SCORE_TO_WIN = 
    N_TESTS = 1

    '''====================
    Use your own agent here.'''
    from game2048.agents import ExpectiMaxAgent as TestAgent
    from game2048.agents import jiluer
    '''===================='''

    #scores = []
    for aa in range(N_TESTS):
        single_run(GAME_SIZE, SCORE_TO_WIN,
                           AgentClass=TestAgent)
        #scores.append(score)
    print(jiluer)
    print(len(jiluer))
    import pandas as pd

    #data1 = pd.DataFrame(jiluer)
    #data1.to_csv('0to64.csv',index=0)

    time_end = time.time()
    print('totally cost', time_end - time_start)
    #print("Average scores: @%s times" % N_TESTS, sum(scores) / len(scores))
