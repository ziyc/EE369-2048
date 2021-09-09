import json
import numpy as np
import torch
from game2048.game import Game


def generate_fingerprint(model,AgentClass, **kwargs):
    with open("board_cases.json") as f:
        board_json = json.load(f)

    game = Game(size=4, enable_rewrite_board=True)
    agent = AgentClass(model=model,game=game, **kwargs)

    trace = []
    for board in board_json:
        game.board = np.array(board)
        direction = agent.step()
        direction = int(direction)
        trace.append(direction)
    fingerprint = "".join(str(i) for i in trace)
    return fingerprint


if __name__ == '__main__':
    from collections import Counter

    '''====================
    Use your own agent here.'''
    from game2048.agents import CZY as TestAgent
    from game2048.Model2 import Net
    '''===================='''

    model = Net()
    model.load_state_dict(torch.load("./trainmodel/0610/para061013_31.pkl", map_location='cpu'))
    model.eval()

    fingerprint = generate_fingerprint(model,TestAgent)

    with open("EE369_fingerprint.json", 'w') as f:        
        pack = dict()
        pack['fingerprint'] = fingerprint
        pack['statstics'] = dict(Counter(fingerprint))
        f.write(json.dumps(pack, indent=4))
