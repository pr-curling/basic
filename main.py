import torch
from model import ResNet, ResidualBlock, load_model, save_model
from collections import deque
import random
import numpy as np

from Simulator import Simulator as sim


def MCTS(model, state):
    prob = 0
    return prob


# def best_shot_parm(prob):
#     turn = prob // 1024
#     rows = (prob % 1024) // 32
#     cols = (prob % 1024) % 32
#
#     print(rows, cols, turn)

def best_shot_parm(prob):
    index = prob

    if prob - 1024 <= 0:
        turn = 0
    else:
        turn = 1

    rows = 0
    tmp = index
    while True:

        tmp = tmp - 32

        if tmp < 0:
            break
        else:
            rows += 1

    cols = index - rows * 32

    x = 4.75 / 32 * cols
    y = 11.28 / 32 * rows
    return [x, y, turn]


def coordinates_to_plane(coordinates):
    # x: 0.14 4.61
    # y: 11.135 2.906
    number_of_coor = len(coordinates)
    coors = []
    for coordinate in coordinates:
        coors.append([])
        for x, y in zip(coordinate[::2], coordinate[1::2]):
            if x == 0 and y ==0:
                coors[-1].append(None)
            else:
                coors[-1].append([int(round((x-0.14)/4.47 * 31)), int(round((y-2.906)/8.229 * 31))])

    plane = torch.zeros((number_of_coor, 2, 32, 32))
    for bat, coor in enumerate(coors):
        for i, c in enumerate(coor):
            if c is None:
                continue
            x, y = c
            if i % 2 == 0:
                plane[bat][0][y][x] = 1
            else:
                plane[bat][1][y][x] = 1

    return plane


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, data):
        self.buffer.append(data)

    def sample(self, batch_size):
        pass

    def __len__(self):
        return len(self.buffer)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    learning_rate = 0.001
    gen = 1
    uncertatinty = 0.145
    batch_size = 32
    epoch = 1

    model = ResNet(ResidualBlock, [2, 2, 2, 2]).to(device)

    model_file_name = None
    if model_file_name is not None:
        load_model(model, model_file_name)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)


    #----------------------TRAIN--------------------------

    num_of_game = 1

    for i in range(gen):
        mem = [] # state turn prob reward
        state = torch.zeros((1, 32))

        for _ in range(num_of_game):
            for turn in range(16):
                state_plane = state.numpy()
                print(state_plane)
                state_plane = coordinates_to_plane(state_plane).to(device)
                print(state_plane.shape)
                prob, _ = model(state_plane)
                print(prob[0].shape)
                action = best_shot_parm(prob)
                state = sim.simulate(state, turn, action[0], action[1], action[3], uncertatinty)
                mem.append([state, turn, prob, 0])

            score = get_score(state)
            for m in mem[-16:]:
                m[3] = score

        for e in range(epoch):

            for i in range(int(len(mem)/batch_size)):
                samples = np.asarray(random.sample(mem, batch_size))

                states = np.vstack(samples[:, 0])
                turns = np.vstack(samples[:, 1])
                probs = np.vstack(samples[:, 2])
                scores = torch.tensor(samples[:, 3]).to(device)

                p_out, v_out = model(state_plane)

                one = torch.sum(- scores * torch.log(v_out)) / batch_size
                two = torch.sum(- probs * torch.log(p_out)) / batch_size

                loss = one + two

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if i % 100 == 0:
                    print("loss", loss)
                if i % 500 == 1:
                    save_model(model, "zero"+str(i))
