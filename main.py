import torch
from torch import nn

import random
import numpy as np

from Simulator import Simulator as sim
from model import ResNet, ResidualBlock, load_model, save_model
from utils import shot_to_onehot_prob, best_shot_parm, get_score, coordinates_to_plane

from tqdm import trange
import time


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    learning_rate = 0.00001
    gen = 1
    uncertatinty = 0.145
    batch_size = 16
    epoch = 30

    model = ResNet(ResidualBlock, [2, 2, 2, 2]).to(device)

    model_file_name = None
    if model_file_name is not None:
        load_model(model, model_file_name)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss()


    #----------------------TRAIN--------------------------

    num_of_game = 3000
    epsilon = 1
    num_of_turn = 16
    order = 0
    for i_gen in range(gen):
        # (state, turn, prob, reward)
        mem = []

        for _ in trange(num_of_game):
            s = time.time()
            state = np.zeros((1, 32))
            for turn in range(num_of_turn):

                if turn % 2 == order:
                    action = (random.random()*4.75, random.random()*11.28, random.randint(0,1))
                    prob = shot_to_onehot_prob(action)
                else:
                    s = time.time()
                    state_plane = coordinates_to_plane(state, turn % 2).to(device)
                    v = None
                    with torch.no_grad():
                        prob, v = model(state_plane)
                    action = best_shot_parm(prob)
                    prob = shot_to_onehot_prob(action)
                # print(action)

                next_state = sim.simulate(state, turn, action[0], action[1], action[2], uncertatinty)[0]

                if turn % 2 == order:
                    mem.append([state, turn, prob, 0])
                state = next_state

            score = get_score(state, order)
            # print(score)
            if score > 0:
                for m in mem[-int(num_of_turn/2):]: # should be changed to -8
                    m[3] = score
            else:
                del(mem[-int(num_of_turn/2):])

            #epsilon *= 0.999
        print("mem", len(mem))

        for e in range(epoch):
            last_loss = 0
            for i in range(int(len(mem)/batch_size)):

                samples = random.sample(mem, batch_size)
                states = np.asarray([x[0] for x in samples])
                turns = [x[1] for x in samples]

                probs = torch.argmax(torch.stack([x[2] for x in samples]), 2).view(-1).to(device)

                scores = torch.LongTensor([x[3]+8 for x in samples]).to(device)

                state_plane = coordinates_to_plane(states).to(device)
                state_plane.requires_grad_()
                # scores.requires_grad_()
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
                    print("loss " +str(e) + " " + str(i), one.item(), two.item(), loss.item())
                #if i % 500 == 1:
                #    save_model(model, "zero"+str(i))

        save_model(model, "zero_final_" + str(i_gen))
