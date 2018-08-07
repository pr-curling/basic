import torch
import numpy as np
from tqdm import trange
import random
from Simulator import Simulator as sim


# def best_shot_parm(prob):
#     turn = prob // 1024
#     rows = (prob % 1024) // 32
#     cols = (prob % 1024) % 32
#
#     print(rows, cols, turn)

def shot_to_onehot_prob(shot):
    prob = torch.zeros((1, 2048))

    x, y, curl = shot

    idx = round(y/11.28*31) *32 + round(x/4.75 *31) + curl*1024
    prob[0][idx] = 1

    return prob


def best_shot_parm(prob):
    index = torch.argmax(prob)
    # print("max", index)

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


def clip(x):
    return max(0, min(31, round(x)))


def coordinates_to_plane(coordinates, turns, order=0):
    # x: 0.14 4.61
    # y: 11.135 2.906

    #--------------- must be modified -------------
    if len(coordinates.shape) == 1 and coordinates.shape[0] == 32:
        coordinates = [coordinates]
    # ---------------------------------------------
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
    ones_plane = torch.ones((number_of_coor,1,32,32))
    plane = torch.cat((plane, ones_plane), 1)
    zeros_plane = torch.ones((number_of_coor,1,32,32))

    if order == 0:
        plane = torch.cat((plane, ones_plane), 1)
        plane = torch.cat((plane, zeros_plane), 1)
    else:
        plane = torch.cat((plane, zeros_plane), 1)
        plane = torch.cat((plane, ones_plane), 1)

    plane = torch.cat((plane, torch.zeros(number_of_coor, 8, 32, 32)), 1)


    for bat, coor in enumerate(coors):
        for i, c in enumerate(coor):
            if c is None:
                continue

            x, y = [clip(cc) for cc in c]

            if i % 2 == order:
                plane[bat][0][y][x] = 1
            else:
                plane[bat][1][y][x] = 1

        if type(turns) is int:
            plane[bat][5 + turns // 2] = torch.ones((32, 32))
        else:
            plane[bat][5 + turns[bat] // 2] = torch.ones((32, 32))

    return plane

def lose_to_win_action(lose_mem, win_mem_size, num_of_turn=16, order=0):
    uncertatinty = 0
    mem = []
    for start_idx in trange(0, len(lose_mem), 8):

        actions = [best_shot_parm(x[2]) for x in lose_mem[start_idx:start_idx+8]]
        while True:
            state = np.zeros((1, 32))
            mem_tmp = []
            for turn in range(num_of_turn):
                if turn % 2 == order:
                    action = (random.random() * 4.75, random.random() * 11.28, random.randint(0, 1))
                    prob = shot_to_onehot_prob(action)
                    mem_tmp.append([state, turn, prob, 0])
                else:
                    action = actions[turn//2]
                state = sim.simulate(state, turn, action[0], action[1], action[2], uncertatinty)[0]

            score = get_score(state, order)
            if score > 0:
                for m in mem_tmp:
                    m[3] = score
                break

        if win_mem_size <= len(mem):
            break

        mem.extend(mem_tmp)

    return mem