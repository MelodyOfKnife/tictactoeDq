import numpy as np
import torch as T
from torch import nn, optim
from torch.nn import functional as F

class TicTacToe:

    position = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [0, 3, 6],
        [1, 4, 7],
        [2, 5, 8],
        [0, 4, 8],
        [2, 4, 6],
    ]
    
    signs = [' ', 'o', 'x']
	
    def __init__(self):
		self.board = np.zeros([3, 3], dtype=int)
    
    def reset(self):
        self.board = np.zeros([3, 3], dtype=int)
        return self.board.flatten()

    def step(self, pos, sign):
        self.board[pos//3][pos%3] = sign
        reward = self.check()
        return self.board.flatten(), reward, (reward != 0)

    def check(self):
        for p in self.position:
            r = 0
            for i in p:
                r += self.board[i//3][i%3]

            if abs(r) == 3:
                return r//3
        return 0

    def __str__(self):
        s = "_______\n"
        for i in range(3):
            for j in range(3):
                s += "|" + self.signs[self.board[i][j]]
            s += "|\n"

        return s



class QModel(nn.Module):
    def __init__(self, state_size=9, action_size=9):
        super(QModel, self).__init__()
        self.d1 = nn.Linear(state_size, 32)
        self.d3 = nn.Linear(32, 32)
        self.d4 = nn.Linear(32, 32)
        self.d5 = nn.Linear(32, 32)
        self.out = nn.Linear(32, action_size)

    def forward(self, x):
        x = F.relu(self.d1(x))
        x = F.relu(self.d2(x))
        x = F.relu(self.d3(x))
        x = F.relu(self.d4(x))
        x = F.relu(self.d5(x))
        x = self.out(x)
        return x

class Agent:
    TAU = .99
    LR = 1e-3
    GAMMA = .99

    def __init__(self, state_size, action_size, epl=1., end_epl=0.01, decay=0.999):
        self.state_size = state_size
        self.action_size = action_size
        self.tQ = QModel(state_size, action_size).cuda()
        self.lQ = QModel(state_size, action_size).cuda()
        self.ops = optim.Adam(self.lQ.parameters(), lr=self.LR)
        self.epl = epl
        self.end_epl = end_epl
        self.decay = decay

    def act(self, state, training=False):
        th = 1.
        if training:
            th = np.random.random()

        if self.epl > th:
             return np.random.randint(0, self.action_size)
        return np.argmax(self.lQ(T.Tensor(state).cuda()), axis=1)

    def step(self, states, actions, rewards, next_states, dones):

        targetY = T.Tensor(rewards).cuda() + self.GAMMA * self.tQ(T.Tensor(next_states).cuda()) * T.Tensor(1 - dones).cuda()
        expectY = self.lQ(T.Tensor(states).cuda())
        loss = F.mse_loss(targetY, expectY)
        self.ops.zero_grads()
        loss.backward()
        self.ops.step()

        self.epl = max(self.epl * self.decay, self.min_epl)

env = TicTacToe()
state = env.reset()
#print(env)
#env.step(3, 1)
#print(env)

agent = Agent(9, 9)
action = agent.act(state, training=True)
next_state, reward, done = env.step(action, 1)
agent.step(state, action, reward, next_state, done)


