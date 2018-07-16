import numpy as np


def get_score(state, turn):

    score = 0

    t_coor = np.array([2.375, 4.88])
    coors = [np.array([state[i], state[i+1]]) for i in range(0, 32, 2)]
    dists = [np.linalg.norm(t_coor-coor) for coor in coors]
    print(dists)

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


turn = 0
a = "0.          0.          0.75884885  4.016514    0.9810853   8.065675 0.          0.          3.467804    6.3805842   4.4806557   6.5864105 0.          0.          1.1077763   8.327171    2.3687203   8.225878 0.61222416  7.590076    1.6262761   4.843929    3.6802986  10.758456 2.8274627   5.480879    3.247639    8.134254    0.58973074  7.9238253 0.          0.        "
a = a.split()
a = [float(x) for x in a]
print(get_score(a, turn))

import matplotlib.pyplot as plt
a = [[a[i], a[i+1]] for i in range(0, 32, 2)]
my = np.asarray(a[turn::2])
op = np.asarray(a[1-turn::2])

cir = plt.Circle((2.375, 4.88),1.83, color='g', alpha=0.2)
fig, ax = plt.subplots()
ax.add_artist(cir)
plt.scatter(my[:,0],my[:,1], color='r')
plt.scatter(op[:,0],op[:,1], color='y')
plt.axis('equal')
plt.show()