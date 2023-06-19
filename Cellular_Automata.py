import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson

rng = np.random.default_rng()

def NS_Algo(L, N_cars, v_max, iterations, rand_slow=0):
    def build_road(iters, L, N_cars):
        road = np.zeros([iters, L])
        road[0, :N_cars] = 1
        rng.shuffle(road[0, :])

        return road

    def find_empty_cells_in_front_of_car(d, lane, L):
        for i in range(L):
            if lane[i] == 0:
                continue
            else:
                if (lane[int((i + 1) % L)] == 0) & (lane[int((i + 2) % L)] == 0):
                    d[i] = 2
                elif lane[int((i + 1) % L)] == 0:
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
                new_lane[int((i + v[i]) % len(lane))] = 1
                new_v[int((i + v[i]) % len(lane))] = v[i]
        return new_lane, new_v

    road = build_road(iterations, L, N_cars)
    # print(road)
    flux = np.zeros([iterations])
    d = np.copy(road)
    d[0, :] = find_empty_cells_in_front_of_car(d[0, :], road[0, :], L)
    v = np.copy(d)
    flux[0] = (np.sum(v[0, :] == 1) + 2 * np.sum(v[0, :] == 2)) / L

    for t in range(1, iterations):
        ## Acceleration
        v[t, :] = np.minimum(v[t-1, :] + 1, v_max)
        ## Slowing down
        v[t, :] = np.minimum(v[t, :], d[t-1, :])
        ## Randomization
        randomize = rng.random([L])
        v[t, randomize < rand_slow] = np.maximum(v[t, randomize < rand_slow] - 1, 0)
        ## Car motion
        road[t, :], v[t, :] = move_cars(road[t-1, :], v[t, :])
        d[t, :] = find_empty_cells_in_front_of_car(d[t, :], road[t, :], L)
        ## Finding flux
        flux[t] = (np.sum(v[t, :] == 1) + 2 * np.sum(v[t, :] == 2)) / L

    return road, flux

def power_distribution(y_min, alpha, N=1):
    ## This returns a power distribution with a minimum value of y_min and a power of alpha

    k = alpha * y_min**alpha
    x = rng.random(N)
    y = (1 / (y_min**(-alpha) - (alpha/k) * x))**(1/alpha)

    return y

def test_power_distribution(y_min, alpha, N=10000):
    ## This tests the power distribution function by plotting a histogram of the results

    k = alpha * y_min**alpha

    y = power_distribution(y_min, alpha, N)
    x = np.linspace(y_min, np.max(y), N)
    y_theory = k * x ** (-(alpha + 1))
    print(simpson(y_theory, x))

    plt.figure()
    plt.hist(y, bins=100, density=True, label="Simulation")
    plt.plot(x, y_theory, 'r')
    plt.title("Power Distribution Histogram")
    plt.ylabel("Count")
    plt.xlabel("y")
    # plt.savefig("Power_Distribution_Histogram.png")
    plt.show()
    # plt.close()

def simulate_RW(L, iterations):
    # moves = rng.choice([-1, 1], size=iterations)
    t = np.zeros(iterations)

    for i in range(iterations):
        print("Iteration: " + str(i+1))
        lattice = np.zeros(L)
        lattice[0] = 1
        is_in_sys = True

        while is_in_sys:
            if lattice[0]:
                lattice = np.roll(lattice, 1)
                t[i] += 1
            else:
                move = rng.choice([-1, 1])
                if not ((lattice[-1]) and (move == 1)):
                    lattice = np.roll(lattice, move)
                    t[i] += 1
                else:
                    is_in_sys = False
                    t[i] += 1

    return t

if __name__ == '__main__':
    print("Lets go!")
    # ## Exercise 1:
    L_lane = 1000
    N_cars = np.array([100, 200, 333, 500, 666, 800, 900])
    v_max = 2
    iter_traffic = 2500
    rand_slow = 0.05
    means_no_slowing = np.zeros(len(N_cars))
    means_slowing = np.zeros(len(N_cars))
    for N in N_cars:
        road, flux = NS_Algo(L_lane, N, v_max, iter_traffic)
        means_no_slowing[np.where(N_cars == N)] = np.mean(flux)

        plt.figure()
        plt.pcolormesh(road)
        plt.title("N = " + str(N) + " cars")
        plt.ylabel("Time")
        plt.xlabel("Road")
        plt.savefig(str(N) + "_cars_no_slowing.png")
        plt.close()

        plt.figure()
        plt.plot(flux)
        plt.title("flux vs time, N = " + str(N) + " cars, no slowing")
        plt.ylabel("flux")
        plt.xlabel("time")
        plt.savefig(str(N) + "_cars_flux_no_slowing.png")
        plt.close()

        road, flux = NS_Algo(L_lane, N, v_max, iter_traffic, rand_slow)
        means_slowing[np.where(N_cars == N)] = np.mean(flux)

        plt.figure()
        plt.pcolormesh(road)
        plt.title("N = " + str(N) + " cars")
        plt.ylabel("Time")
        plt.xlabel("Road")
        plt.savefig(str(N) + "_cars_rand_" + str(rand_slow) + ".png")
        plt.close()

        plt.figure()
        plt.plot(flux)
        plt.title("flux vs time, N = " + str(N) + " cars, rand_slow = " + str(rand_slow))
        plt.ylabel("flux")
        plt.xlabel("time")
        plt.savefig(str(N) + "_cars_flux_rand_" + str(rand_slow) + ".png")
        plt.close()

    plt.figure()
    plt.plot(N_cars/L_lane, means_no_slowing, label="No slowing")
    plt.plot(N_cars/L_lane, means_slowing, label="Slowing")
    plt.title("flux vs density")
    plt.ylabel("flux")
    plt.xlabel("density")
    plt.legend()
    plt.savefig("flux_vs_density.png")
    plt.show()

    print("Done with traffic jams!")

    ## Exercise 2:
    y_min = 0.1
    alpha = 1.5
    N = 5000
    y = test_power_distribution(y_min, alpha, N)

    print("Done with power distribution!")

    ## Exercise 3:
    L_RW = 50
    iter_RW = 10
    time = simulate_RW(L_RW, iter_RW)
    mean_time = np.mean(time)
