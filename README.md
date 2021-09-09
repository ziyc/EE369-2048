# 代码详情-CZY
* [`game2048/`](game2048/): the main package.
   * [`agents.py`](game2048/agents.py): 其中的`CZY` class是自主定义的Agent
   * [`model2.py`](game2048/model2.py): 保存了CNN模型框架
* [`trainmoel/`](trainmodel/): 保存了训练网络的py文件和参数文件
   * [`train2.py`](trainmodel/train2.py): 训练CNN网络
* [`CREATEDATA.py`](CREATEDATA.py): 用于生成训练数据的程序
* [`evaluate.py`](evaluate.py): evaluate my self-defined agent.点开此程序运行代码即可。

# 2048-api
A 2048 game api for training supervised learning (imitation learning) or reinforcement learning agents

# Code structure
* [`game2048/`](game2048/): the main package.
    * [`game.py`](game2048/game.py): the core 2048 `Game` class.
    * [`agents.py`](game2048/agents.py): the `Agent` class with instances.
    * [`displays.py`](game2048/displays.py): the `Display` class with instances, to show the `Game` state.
    * [`expectimax/`](game2048/expectimax): a powerful ExpectiMax agent by [here](https://github.com/nneonneo/2048-ai).
* [`explore.ipynb`](explore.ipynb): introduce how to use the `Agent`, `Display` and `Game`.
* [`static/`](static/): frontend assets (based on Vue.js) for web app.
* [`webapp.py`](webapp.py): run the web app (backend) demo.
* [`evaluate.py`](evaluate.py): evaluate your self-defined agent.

# Requirements
* code only tested on linux system (ubuntu 16.04)
* Python 3 (Anaconda 3.6.3 specifically) with numpy and flask

# To define your own agents
```python
from game2048.agents import Agent

class YourOwnAgent(Agent):

    def step(self):
        '''To define the agent's 1-step behavior given the `game`.
        You can find more instance in [`agents.py`](game2048/agents.py).
        
        :return direction: 0: left, 1: down, 2: right, 3: up
        '''
        direction = some_function(self.game)
        return direction

```

# To compile the pre-defined ExpectiMax agent

```bash
cd game2048/expectimax
bash configure
make
```

# To run the web app
```bash
python webapp.py
```
![demo](preview2048.gif)

# LICENSE
The code is under Apache-2.0 License.

# For EE369 / EE228 students from SJTU
Please read course project [requirements](EE369.md) and [description](https://docs.qq.com/slide/DS05hVGVFY1BuRVp5). 
