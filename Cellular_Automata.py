import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from scipy.linalg import lstsq
from scipy.optimize import nnls, curve_fit

rng = np.random.default_rng()


def NS_Algo(L, N_cars, v_max, iterations, P=.0):
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
        v[t, randomize < P] = np.maximum(v[t, randomize < P] - 1, 0)
        ## Car motion
        road[t, :], v[t, :] = move_cars(road[t-1, :], v[t, :])
        d[t, :] = find_empty_cells_in_front_of_car(d[t, :], road[t, :], L)
        ## Finding flux
        flux[t] = (np.sum(v[t, :] == 1) + 2 * np.sum(v[t, :] == 2)) / L

    return road, flux


def power_distribution(y_min, alpha, N=1):
    ## This returns a power distribution with a minimum value of y_min and a power of alpha
    ## N is the number of samples to return

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


def sand_pile_model(size, max_grain=3, iterations=1000):
    ## This function simulates the sand pile model on a lattice of size LxL

    def topple(lattice, y, x, it):
        toppled[it] += 1
        # print("topple")
        ## This function topples the lattice at a given site with open boundary conditions
        if not ((x == L-1) or (x == 0) or (y == L-1) or (y == 0)):
            lattice[y+1, x] += 1
            if lattice[y+1, x] > max_grain:
                lattice[y+1, x] = 0
                topple(lattice, y+1, x, it)
            lattice[y-1, x] += 1
            if lattice[y-1, x] > max_grain:
                lattice[y-1, x] = 0
                topple(lattice, y-1, x, it)
            lattice[y, x+1] += 1
            if lattice[y, x+1] > max_grain:
                lattice[y, x+1] = 0
                topple(lattice, y, x+1, it)
            lattice[y, x-1] += 1
            if lattice[y, x-1] > max_grain:
                lattice[y, x-1] = 0
                topple(lattice, y, x-1, it)
        elif (x == L-1) and (y == L-1):
            lattice[y-1, x] += 1
            if lattice[y-1, x] > max_grain:
                lattice[y-1, x] = 0
                topple(lattice, y-1, x, it)
            lattice[y, x-1] += 1
            if lattice[y, x-1] > max_grain:
                lattice[y, x-1] = 0
                topple(lattice, y, x-1, it)
        elif (x == L-1) and (y == 0):
            lattice[y+1, x] += 1
            if lattice[y+1, x] > max_grain:
                lattice[y+1, x] = 0
                topple(lattice, y+1, x, it)
            lattice[y, x-1] += 1
            if lattice[y, x-1] > max_grain:
                lattice[y, x-1] = 0
                topple(lattice, y, x-1, it)
        elif (x == 0) and (y == L-1):
            lattice[y-1, x] += 1
            if lattice[y-1, x] > max_grain:
                lattice[y-1, x] = 0
                topple(lattice, y-1, x, it)
            lattice[y, x+1] += 1
            if lattice[y, x+1] > max_grain:
                lattice[y, x+1] = 0
                topple(lattice, y, x+1, it)
        elif (x == 0) and (y == 0):
            lattice[y+1, x] += 1
            if lattice[y+1, x] > max_grain:
                lattice[y+1, x] = 0
                topple(lattice, y+1, x, it)
            lattice[y, x+1] += 1
            if lattice[y, x+1] > max_grain:
                lattice[y, x+1] = 0
                topple(lattice, y, x+1, it)
        elif (x == L-1):
            lattice[y+1, x] += 1
            if lattice[y+1, x] > max_grain:
                lattice[y+1, x] = 0
                topple(lattice, y+1, x, it)
            lattice[y-1, x] += 1
            if lattice[y-1, x] > max_grain:
                lattice[y-1, x] = 0
                topple(lattice, y-1, x, it)
            lattice[y, x-1] += 1
            if lattice[y, x-1] > max_grain:
                lattice[y, x-1] = 0
                topple(lattice, y, x-1, it)
        elif (x == 0):
            lattice[y+1, x] += 1
            if lattice[y+1, x] > max_grain:
                lattice[y+1, x] = 0
                topple(lattice, y+1, x, it)
            lattice[y-1, x] += 1
            if lattice[y-1, x] > max_grain:
                lattice[y-1, x] = 0
                topple(lattice, y-1, x, it)
            lattice[y, x+1] += 1
            if lattice[y, x+1] > max_grain:
                lattice[y, x+1] = 0
                topple(lattice, y, x+1, it)
        elif (y == L-1):
            lattice[y-1, x] += 1
            if lattice[y-1, x] > max_grain:
                lattice[y-1, x] = 0
                topple(lattice, y-1, x, it)
            lattice[y, x+1] += 1
            if lattice[y, x+1] > max_grain:
                lattice[y, x+1] = 0
                topple(lattice, y, x+1, it)
            lattice[y, x-1] += 1
            if lattice[y, x-1] > max_grain:
                lattice[y, x-1] = 0
                topple(lattice, y, x-1, it)
        elif (y == 0):
            lattice[y+1, x] += 1
            if lattice[y+1, x] > max_grain:
                lattice[y+1, x] = 0
                topple(lattice, y+1, x, it)
            lattice[y, x+1] += 1
            if lattice[y, x+1] > max_grain:
                lattice[y, x+1] = 0
                topple(lattice, y, x+1, it)
            lattice[y, x-1] += 1
            if lattice[y, x-1] > max_grain:
                lattice[y, x-1] = 0
                topple(lattice, y, x-1, it)

        return lattice

    toppled = np.zeros(iterations)
    lattice = np.zeros([L, L])
    dump_sites = rng.integers(0, L, size=[iterations, 2])  # First column is y, second column is x

    for i in range(iterations):
        # print(i, 'out of', iterations)
        lattice[tuple(dump_sites[i])] += 1
        if lattice[tuple(dump_sites[i])] > max_grain:
            lattice[tuple(dump_sites[i])] -= (max_grain + 1)
            lattice = topple(lattice, *dump_sites[i], i)
    return lattice, toppled

def func_powerlaw(x, tau, c):
    return c * x**(-tau)

if __name__ == '__main__':
    # print("Lets go!")
    # # ## Exercise 1:
    # L_lane = 1000
    # N_cars = np.array([100, 200, 333, 500, 666, 800, 900])
    # v = np.linspace(0, 2, 3)
    # v_max = 2
    # iter_traffic = 1500
    # P = 0.05
    #
    # means_no_slowing = np.zeros(len(N_cars))
    # means_slowing = np.zeros(len(N_cars))
    # flux_no_slowing = np.zeros([len(N_cars), iter_traffic])
    # flux_slowing = np.zeros([len(N_cars), iter_traffic])
    # flux_linalg = np.zeros(len(N_cars))
    # conc = np.zeros([len(N_cars), 3])
    #
    # for N in N_cars:
    #     road, flux_no_slowing[np.where(N_cars == N), :] = NS_Algo(L_lane, N, v_max, iter_traffic)
    #     means_no_slowing[np.where(N_cars == N)] = np.mean(flux_no_slowing[np.where(N_cars == N)])
    #
    #     plt.figure()
    #     plt.pcolormesh(road)
    #     plt.title("N = " + str(N) + " cars")
    #     plt.ylabel("Time")
    #     plt.xlabel("Road")
    #     plt.savefig(str(N) + "_cars_no_slowing.png")
    #     plt.close()
    #
    #     plt.figure()
    #     plt.plot(flux_no_slowing[np.where(N_cars == N)][0])
    #     plt.title("flux vs time, N = " + str(N) + " cars, no slowing")
    #     plt.ylabel("flux")
    #     plt.xlabel("time")
    #     plt.savefig(str(N) + "_cars_flux_no_slowing.png")
    #     plt.close()
    #
    #     road, flux_slowing[np.where(N_cars == N)] = NS_Algo(L_lane, N, v_max, iter_traffic, P)
    #     means_slowing[np.where(N_cars == N)] = np.mean(flux_slowing[np.where(N_cars == N)])
    #
    #     plt.figure()
    #     plt.pcolormesh(road)
    #     plt.title("N = " + str(N) + " cars")
    #     plt.ylabel("Time")
    #     plt.xlabel("Road")
    #     plt.savefig(str(N) + "_cars_rand_" + str(P) + ".png")
    #     plt.close()
    #
    #     plt.figure()
    #     plt.plot(flux_slowing[np.where(N_cars == N)][0])
    #     plt.title("flux vs time, N = " + str(N) + " cars, P = " + str(P))
    #     plt.ylabel("flux")
    #     plt.xlabel("time")
    #     plt.savefig(str(N) + "_cars_flux_rand_" + str(P) + ".png")
    #     plt.close()
    #
    #     ## Calculate expected from mean-field theory (Schreckenberg, 1995, Phys. Rev, E 51, 2939)
    #     c = N / L_lane
    #     d = 1 - c
    #     q = 1 - P
    #
    #     conc[np.where(N_cars == N), 0] = c**2 * (1 + P * d)/(1 - P * d**2)
    #     conc[np.where(N_cars == N), 1] = q *c**2 *d * (1 + d + P * d**2)/((1 - P * d**3) * (1 - P * d**2))
    #     conc[np.where(N_cars == N), 2] = q**2 * c**2 * d**3 * (1 + d + d**2 * P)/((1 - q * d**2) * (1 - P * d**3) *
    #                                                                                (1 - P * d**2))
    #
    # flux_linalg = np.sum(conc * v, axis=1)
    #
    # plt.figure()
    # plt.plot(N_cars/L_lane, means_no_slowing, label="No slowing")
    # plt.plot(N_cars/L_lane, means_slowing, label="Slowing")
    # plt.plot(N_cars/L_lane, flux_linalg, label="Rate equations")
    # plt.title("flux vs density")
    # plt.ylabel("flux")
    # plt.xlabel("density")
    # plt.legend()
    # plt.savefig("flux_vs_density.png")
    # plt.show()
    #
    # print("Done with traffic jams!")
    #
    # ## Exercise 2:
    # y_min = 0.01
    # alpha = 1.5
    # N = 5000
    # test_power_distribution(y_min, alpha, N)
    #
    # print("Done with power distribution!")
    #
    # ## Exercise 3:
    # L_RW = 50
    # iter_RW = 10
    # time = simulate_RW(L_RW, iter_RW)
    # mean_time = np.mean(time)

    ## Exercise 4:
    print("Lets go!")
    L = 50

    lat, top = sand_pile_model(L, iterations=int(1e6))
    histo, bins = np.histogram(top, 'sqrt')    ## I have to make sure in bins there are is 0
    popt, pcov = curve_fit(func_powerlaw, bins[1:], histo, p0=[1, np.max(histo)], bounds=([0.5,np.max(histo)/1e4],
                                                                                          [1.5, np.max(histo)*1e2]))


    # plt.figure()
    # plt.pcolormesh(lat)
    # plt.title("Sand pile model")
    # plt.ylabel("y")
    # plt.xlabel("x")
    # # plt.savefig("sand_pile_model.png")
    # plt.close()
    #
    # plt.figure()
    # plt.plot(top)
    # histogram = plt.hist("Topples")
    # plt.ylabel("Number of topples")
    # plt.xlabel("Iteration")
    # # plt.savefig("toppling.png")
    # plt.close()

    plt.figure()
    plt.loglog(bins[1:], histo, label="Data")
    plt.loglog(bins[1:], func_powerlaw(bins[1:], *popt), label="Fit")
    plt.title("Power law fit")
    plt.xlabel("Number of topples")
    plt.ylabel("Occurences")
    plt.legend()
    plt.savefig("power_law_fit.png")
    plt.show()


    print("Done with sand pile model!")
