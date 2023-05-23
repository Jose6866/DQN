import torch.nn.functional as F
import numpy as np

# left, right, up, down
ACTIONS = [np.array([0, -1]),
           np.array([0, 1]),
           np.array([-1, 0]),
           np.array([1, 0])]

TRAINING_EPOCH_NUM = 10000
gamma         = 0.98
batch_size    = 32

class AGENT:
    def __init__(self, env, q, q_target, memory, optimizer, is_upload=False):

        self.ACTIONS = ACTIONS
        self.env = env
        self.q = q
        self.q_target = q_target
        self.memory = memory
        self.optimizer = optimizer
#        HEIGHT, WIDTH = env.size()
        self.state = [0,0]
        self.loss_avg = 0

        if is_upload:
            qlearning_results = np.load('./result/qlearning.npz')
            self.q = qlearning_results['Q']
            self.q_target = qlearning_results['Q_TAGET']
            self.memory = qlearning_results['MEMORY']
            self.optimizer = qlearning_results['OPTIMIZER']
#        else:
#            self.V_values = np.zeros((HEIGHT, WIDTH))
#            self.Q_values = np.zeros((HEIGHT, WIDTH, len(self.ACTIONS)))
#            self.policy = np.zeros((HEIGHT, WIDTH,len(self.ACTIONS)))+1./len(self.ACTIONS)

    def initialize_episode(self):
        HEIGHT, WIDTH = self.env.size()
        while True:
            i = np.random.randint(HEIGHT)
            j = np.random.randint(WIDTH)
            state = [i, j]
            if (state in self.env.goal) or (state in self.env.obstacles):
                continue
            break
            # if (state not in self.env.goal) and (state not in self.env.obstacles):
            #     break
        return state

    def train(self, q, q_target, memory, optimizer):
        loss_list = []

        for i in range(100):
            s, a, r, s_prime, done_mask = memory.sample(32)

            q_out = self.q(s)
            q_a = q_out.gather(1, a)
            max_q_prime = self.q_target(s_prime).max(1)[0].unsqueeze(1)
            target = r + gamma * max_q_prime * done_mask
            loss = F.smooth_l1_loss(q_a, target)
            loss_list.append(loss)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        loss_avg = sum(loss_list) / len(loss_list)
        return loss_avg



    def Q_learning(self, discount=1.0, alpha=0.01, max_seq_len=500,
                             decay_period=20000, decay_rate=0.9):

        for episode in range(TRAINING_EPOCH_NUM):
            epsilon = max(0.01, 0.08 - 0.01 * (episode / 200))
            state = self.initialize_episode()
            done = False
            timeout = False
            seq_len = 0
            cum_reward = 0
            while not (done or timeout):
                # Next state and action generation
                action = self.get_action(state, epsilon)
                movement = ACTIONS[action]
                next_state, reward = self.env.interaction(state, movement)

                if self.env.is_terminal(next_state):
                    done_mask = 0.0
                else:
                    done_mask = 1.0

                self.memory.put((state, action, reward, next_state, done_mask))
                state = next_state  # agent를 다음 state로 이동
                cum_reward = cum_reward + reward
                seq_len += 1
                if (seq_len >= max_seq_len):
                    timeout = True
                done = self.env.is_terminal(state)

            if self.memory.size() > 2000:
                self.loss_avg = self.train(self.q, self.q_target, self.memory, self.optimizer)


            if episode % 100 == 0:
                print("Num of episodes = {:}, epsilon={:.4f}, loss avg={: .8f}".format(episode, epsilon, self.loss_avg))

            if episode % decay_period == 0:
                epsilon *= decay_rate

            print(self.q)



        np.savez('./result/qlearning.npz', Q=self.q, Q_TARGET=self.q_target, MEMORY=self.memory, OPTIMIZER=self.optimizer)
        return self.q, self.q_target, self.memory, self.optimizer


    def get_action(self, state, epsilon):  # epsilon-greedy
        i, j = state
        action = np.random.choice(len(ACTIONS))
        return action
