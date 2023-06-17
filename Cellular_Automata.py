import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(1234)

def build_road(iters, L, N_cars):
    road = np.zeros([iters, L])
    road[0, :N_cars] = 1
    rng.shuffle(road[0, :])
    # print(road[0, :])

    return road

def find_empty_cells_in_front_of_car(d, lane, L):
    for i in range(L):
        if lane[i] == 0:
            continue
        else:
            # print(i, (i+1) % L, (i+2) % L)
            if (lane[int((i+1) % L)] == 0) & (lane[int((i+2) % L)] == 0):
                d[i] = 2
            elif lane[int((i+1) % L)] == 0:
                d[i] = 1
            else:
                d[i] = 0
    return d

def move_cars(lane, v):
    new_lane = np.zeros([len(lane)])
    new_v = np.zeros([len(lane)])
    for i in range(len(lane)):
        if lane[i] == 0:
            continue
        else:
            # lane[i] = 0
            # print((i+v[i]) % len(lane))
            new_lane[int((i+v[i]) % len(lane))] = 1
            new_v[int((i+v[i]) % len(lane))] = v[i]
            # v[i] = 0
    # return lane, v
    return new_lane, new_v

def NS_Algo(L, N_cars, v_max, iterations,rand_slow):
    road = build_road(iterations, L, N_cars)
    d = np.copy(road)
    d[0, :] = find_empty_cells_in_front_of_car(d[0, :], road[0, :], L)
    v = np.copy(d)

    for t in range(1, iterations):
        ## Acceleration
        v[t, :] = np.minimum(v[t-1, :] + 1, v_max)
        ## Slowing down
        v[t, :] = np.minimum(v[t, :], d[t-1, :])
        ## Randomization
        randomize = rng.random([L])
        v[t, randomize < rand_slow] = np.maximum(v[t, randomize < rand_slow] - 1, 0)
        ## Car motion
        # print(road[t-1, :])
        road[t, :], v[t, :] = move_cars(road[t-1, :], v[t, :])
        d[t, :] = find_empty_cells_in_front_of_car(d[t, :], road[t, :], L)

    return road

if __name__ == '__main__':
    print("Lets go!")
    L = 1000
    N_cars = [100]#, 200, 333, 500]
    v_max = 2
    iterations = 1000
    rand_slow = 0.01
    for N in N_cars:
        road = NS_Algo(L, N, v_max, iterations,rand_slow)
        plt.figure()
        plt.pcolormesh(road)
        plt.title("N = " + str(N) + " cars")
        plt.ylabel("Time")
        plt.xlabel("Road")
        plt.savefig(str(N) + "_cars.png")
        plt.close()
