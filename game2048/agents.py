import numpy as np
import torch
import time
class Agent:
    '''Agent Base.'''

    def __init__(self, model,game, display=None):
        self.game = game
        self.display = display
        self.model = model

    def play(self, max_iter=np.inf, verbose=False):
        n_iter = 0
        while (n_iter < max_iter) and (not self.game.end):
            direction = self.step()
            self.game.move(direction)
            n_iter += 1
            if verbose:
                print("Iter: {}".format(n_iter))
                print("======Direction: {}======".format(
                    ["left", "down", "right", "up"][direction]))
                if self.display is not None:
                    self.display.display(self.game)

    def step(self):
        direction = int(input("0: left, 1: down, 2: right, 3: up = ")) % 4
        return direction


class RandomAgent(Agent):

    def step(self):
        direction = np.random.randint(0, 4)
        return direction


class ExpectiMaxAgent(Agent):

    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        from .expectimax import board_to_move
        self.search_func = board_to_move

    def step(self):
        direction = self.search_func(self.game.board)
        print(type(self.game.board),self.game.board)
        return direction

class CZY(Agent):

    def __init__(self,model, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(model,game, display)
        self.model =model
        self.search_func = model


    def step(self):
        board = self.game.board
        board[board== 0] = 1
        board=np.log2(board)
        board=np.reshape(board,(1,16))
        X = np.int64(board)
        X = grid_ohe(X)
        X = torch.FloatTensor(X)
        output = self.search_func(X)
        direction = output.data.max(1, keepdim=True)[1]
        return direction

def grid_ohe(input):
    output=[]
    onelayer=[]
    for counter in range(len(input)):
        oneofinput = input[counter, :]
        for layer in range(12):
            ret=np.zeros(shape=(4,4),dtype=int)
            for r in range(4):
                for c in range(4):
                    if layer==oneofinput[r*4+c]:
                        ret[r,c]=1
            onelayer.append(ret)
        output.append(onelayer)
        onelayer = []
    return output
