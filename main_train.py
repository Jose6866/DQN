from environment import grid_world
from agent import AGENT
from replaybuffer import ReplayBuffer
from qnet import Qnet
import torch.optim as optim
import torch

WORLD_HEIGHT = 5
WORLD_WIDTH = 10
learning_rate = 0.0005

env = grid_world(WORLD_HEIGHT,WORLD_WIDTH,
                 GOAL = [[WORLD_HEIGHT-1, WORLD_WIDTH-1]],
                 OBSTACLES=[[0,2], [1,2], [2,2], [0,4], [2,4], [4,4], [2, 6],
                            [3, 6],[4, 6], [2,7], [2,8]])

#USE_CUDA = torch.cuda.is_available()
#device = torch.device('cuda:0' if USE_CUDA else 'cpu')
#print('학습을 진행하는 기기:', device)

q = Qnet()
q_target = Qnet()
q_target.load_state_dict(q.state_dict())
memory = ReplayBuffer()
optimizer = optim.Adam(q.parameters(), lr=learning_rate)

agent = AGENT(env, q, q_target, memory, optimizer, is_upload=False)
agent.Q_learning(decay_period=10000, decay_rate=0.8).cuda()





