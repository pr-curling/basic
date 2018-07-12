import torch
from torch import nn
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
    index = torch.argmax(prob)
    print("max", index)

    if index - 1024 < 0:
        turn = 0
    else:
        turn = 1

    if turn == 1:
        rows = 0
        tmp = index-1024
        rows = tmp.item() // 32
        cols = (index-1024) - rows * 32

    else:
        rows = 0
        tmp = index
        rows = tmp.item() // 32
        cols = index - rows * 32

    x = 4.75 / 31 * cols.item()
    y = 11.28 / 31 * rows
    return [x, y, turn]

def get_score(state, turn):

    score = 0

    t_coor = np.array([2.375, 4.88])
    coors = [np.array([state[i], state[i+1]]) for i in range(0, 32, 2)]
    dists = [np.linalg.norm(t_coor-coor) for coor in coors]

    my = sorted(dists[turn::2])
    op = sorted(dists[1-turn::2])

    if my[0] < op[0]:
        for my_dist in my:
            if my_dist < op[0] and my_dist < 1.97:
                score += 1
            else:
                break
    else:
        for op_dist in op:
            if op_dist < my[0] and op_dist < 1.97:
                score -= 1
            else:
                break

    return score

def coordinates_to_plane(coordinates):
    # x: 0.14 4.61
    # y: 11.135 2.906
    if len(coordinates.shape) == 1:
        coordinates = [coordinates]
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

    learning_rate = 0.0001
    gen = 3
    uncertatinty = 0.145
    batch_size = 1
    epoch = 3

    model = ResNet(ResidualBlock, [2, 2, 2, 2]).to(device)

    model_file_name = None
    if model_file_name is not None:
        load_model(model, model_file_name)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss()


    #----------------------TRAIN--------------------------

    num_of_game = 20
    epsilon = 0.99
    for i in range(gen):
        mem = [] # state turn prob reward

        for _ in range(num_of_game):

            state = torch.zeros((1, 32))
            for turn in range(16):
                state_plane = state
                state_plane = coordinates_to_plane(state_plane).to(device)
                prob, _ = model(state_plane)
                if epsilon > random.random():
                    action = (random.random()*4.75, random.random()*11.28, random.randint(0,1))
                else:
                    action = best_shot_parm(prob)

                # print("state, action ", state, action)
                state = sim.simulate(state, turn, action[0], action[1], action[2], uncertatinty)[0]
                mem.append([state, turn, prob, 0])
            prob_np = prob.detach().cpu().numpy()
            score = get_score(state, 0)
            for m in mem[-16:]:
                m[3] = score

            epsilon *= 0.98
        print("mem", len(mem))

        for e in range(epoch):
            last_loss = 0
            #for i in range(int(len(mem)/batch_size)):
            for i in range(400):
                #samples = np.asarray(random.sample(mem, batch_size))

                # states = np.vstack(samples[:, 0])
                # turns = np.vstack(samples[:, 1])
                # probs = torch.tensor(np.vstack(samples[:, 2])).to(device)
                # scores = torch.tensor(samples[:, 3]).to(device)

                samples = random.sample(mem, 1)[0]

                states = samples[0]
                turns = samples[1]
                probs = torch.empty(1, dtype=torch.long).to(device)
                probs[0] = torch.argmax(samples[2])
                scores = torch.empty(1, dtype=torch.long).to(device)
                scores[0] = samples[3]+8

                state_plane = coordinates_to_plane(states).to(device)
                state_plane.requires_grad_()
                p_out, v_out = model(state_plane)

                # print(probs.item())

                # one = torch.sum(- scores * torch.log(v_out))
                # two = torch.sum(- probs * torch.log(p_out))

                one = criterion(v_out, scores)
                two = criterion(p_out, probs)

                loss = one + two

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


                if i % 100 == 0:
                    # print(state)
                    # state_plane = state
                    # state_plane = coordinates_to_plane(state_plane).to(device)
                    # prob, v = model(state_plane)
                    # print("ho", max(prob[0]), v[0])
                    # a = prob.detach().cpu().numpy()[0][:1024]
                    # a= np.reshape(a, (1,32,32))
                    # print(best_shot_parm(prob))
                    print("loss " +str(e) + " " + str(i), one.item(), two.item(), loss.item())
                if i % 500 == 1:
                    save_model(model, "zero"+str(i))

        save_model(model, "zero_final")
