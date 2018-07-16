import torch
from model import ResNet, ResidualBlock, load_model
import random
import numpy as np

from Simulator import Simulator as sim

from main import coordinates_to_plane, best_shot_parm, get_score


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ResNet(ResidualBlock, [2, 2, 2, 2]).to(device)

model_file_name = "zero_final"
load_model(model, model_file_name)

num_of_game = 100
num_of_win = 0
for _ in range(num_of_game):
    state = np.zeros((1, 32))
    for turn in range(16):
        if turn % 2 == 0:
            state_plane = coordinates_to_plane(state).to(device)

            prob, _ = model(state_plane)
            action = best_shot_parm(prob)
        else:
            action = (random.random() * 4.75, random.random() * 11.28, random.randint(0, 1))

        state = sim.simulate(state, turn, action[0], action[1], action[2], 0.145)[0]
    if get_score(state, 0) > 0:
        num_of_win += 1

print(num_of_win / 100)
