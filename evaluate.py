from game2048.game import Game
from game2048.displays import Display
import torch

def single_run( model,size,score_to_win, AgentClass, **kwargs):
    game = Game(size, score_to_win)
    agent = AgentClass(model,game, display=Display(), **kwargs)
    agent.play(verbose=True)
    return game.score


if __name__ == '__main__':
    GAME_SIZE = 4
    SCORE_TO_WIN = 2048
    N_TESTS =10

    '''====================
    Use your own agent here.'''
   # from game2048.agents import ExpectiMaxAgent as TestAgent
    from game2048.agents import CZY as TestAgent
    from game2048.Model2 import Net
    '''===================='''

    model = Net()
    model.load_state_dict(torch.load("./trainmodel/0610/para061013_31.pkl", map_location='cpu'))
    model.eval()
    scores = []
    for _ in range(N_TESTS):
        score = single_run(model,GAME_SIZE, SCORE_TO_WIN,
                           AgentClass=TestAgent)
        scores.append(score)
        #print(score)

    print("Average scores: @%s times" % N_TESTS, sum(scores) / len(scores))
    print("times of 2048: %s" % scores.count(2048))
    print(scores)
