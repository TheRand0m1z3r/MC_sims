import numpy as np
import sys
import matplotlib
matplotlib.use('TkAgg')
from scipy.integrate import simpson
from scipy.linalg import lstsq
from scipy.optimize import nnls, curve_fit
import matplotlib.pyplot as plt

sys.setrecursionlimit(100000)
rng = np.random.default_rng()

def ex_1():
    L_lane = 1000
    N_cars = np.linspace(0, 1000, 26, dtype=int)
    v = np.linspace(0, 2, 3)
    v_max = 2
    iter_traffic = 1500
    P = 0.05
    P_po = 0.005

    means_no_slowing = np.zeros(len(N_cars))
    means_slowing = np.zeros(len(N_cars))
    means_po = np.zeros(len(N_cars))
    flux_no_slowing = np.zeros([len(N_cars), iter_traffic])
    flux_slowing = np.zeros([len(N_cars), iter_traffic])
    flux_po = np.zeros([len(N_cars), iter_traffic])
    conc = np.zeros([len(N_cars), 3])
    conc_pull_over = np.zeros([len(N_cars), 3])

    for N in N_cars:
        print(N)
        road_no_slowing, flux_no_slowing[np.where(N_cars == N), :] = NS_Algo(L_lane, N, v_max, iter_traffic)
        means_no_slowing[np.where(N_cars == N)] = np.mean(flux_no_slowing[np.where(N_cars == N)])

        road_slowing, flux_slowing[np.where(N_cars == N)] = NS_Algo(L_lane, N, v_max, iter_traffic, P)
        means_slowing[np.where(N_cars == N)] = np.mean(flux_slowing[np.where(N_cars == N)])

        road_po, flux_po[np.where(N_cars == N)] = NS_Algo(L_lane, N, v_max, iter_traffic, P, P_po)
        means_po[np.where(N_cars == N)] = np.mean(flux_po[np.where(N_cars == N)])

        # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
        # ax1.imshow(road_no_slowing)
        # ax1.set_xlabel("Road")
        # ax1.set_ylabel("Time")
        # ax2.imshow(road_slowing)
        # ax2.set_xlabel("Road")
        # ax3.imshow(road_po)
        # ax3.set_xlabel("Road")
        # plt.savefig(str(N) + "_cars_all.png")
        # plt.close()
        #
        # fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
        # ax1.plot(flux_no_slowing[np.where(N_cars == N)][0])
        # ax1.set_ylabel("flux")
        # ax2.plot(flux_slowing[np.where(N_cars == N)][0])
        # ax2.set_ylabel("flux")
        # ax3.plot(flux_po[np.where(N_cars == N)][0])
        # ax3.set_ylabel("flux")
        # ax3.set_xlabel("time")
        # plt.savefig(str(N) + "_cars_flux_all.png")
        # plt.close()


        ## Calculate expected from mean-field theory (Schreckenberg, 1995, Phys. Rev, E 51, 2939)
        c = N / L_lane
        d = 1 - c
        q = 1 - P

        conc[np.where(N_cars == N), 0] = c ** 2 * (1 + P * d) / (1 - P * d ** 2)
        conc[np.where(N_cars == N), 1] = q * c ** 2 * d * (1 + d + P * d ** 2) / ((1 - P * d ** 3) * (1 - P * d ** 2))
        conc[np.where(N_cars == N), 2] = q ** 2 * c ** 2 * d ** 3 * (1 + d + d ** 2 * P) / (
                (1 - q * d ** 2) * (1 - P * d ** 3) *
                (1 - P * d ** 2))

        conc_pull_over[np.where(N_cars == N), 0] = c**2 * ((1-P_po) + P * d)/(1 - P * d**2)
        conc_pull_over[np.where(N_cars == N), 1] = (1 - (1 - P) * d**2) * c * (1 - (1 - P_po) * c -P*d) / (1 - P * d**2)
        conc_pull_over[np.where(N_cars == N), 2] = (1 - (1 - P_po) * c - P*d) * d**2 * (1 - P) * c / (1 - P * d**2)

    flux_linalg = np.sum(conc * v, axis=1)
    flux_linalg_pull_over = np.sum(conc_pull_over * v, axis=1)

    plt.figure()
    plt.plot(N_cars / L_lane, means_no_slowing, label="No slowing")
    plt.plot(N_cars / L_lane, means_slowing, label="Slowing")
    plt.plot(N_cars / L_lane, means_po, label="Slowing and pull over")
    plt.plot(N_cars / L_lane, flux_linalg, label="Rate equations")
    plt.plot(N_cars / L_lane, flux_linalg_pull_over, label="Rate equations, pull over")
    # plt.title("flux vs density")
    plt.ylabel("flux")
    plt.xlabel("density")
    plt.legend()
    plt.savefig("flux_vs_density_P_" + str(P) + "_P_po_" + str(P_po) + ".png")
    plt.show()

    print("Done with traffic jams!")


def ex_2():
    y_min = 1
    alpha = 5
    test_power_distribution(y_min, alpha)
    print("Done with power distribution!")


def ex_3():
    print("Lets go ex. 3!")
    L_RW = 25
    iter_RW = 10000
    time = simulate_RW(L_RW, iter_RW)
    mean_time = np.mean(time)
    print("Mean time for random walk with L = %i and %i iterations is %.3f" % (L_RW, iter_RW, mean_time))
    abs_err = np.abs(mean_time - L_RW ** 2)/L_RW ** 2 * 100
    print("Absolute error is %.3f percent" % abs_err)
    return mean_time


def ex_4():
    print("Lets go ex. 4!")
    L = 128
    lat, top = sand_pile_model(L, iterations=int(1e5))
    histo, bins = np.histogram(top, np.linspace(1e-10, np.max(top), int(np.max(top))+1))

    new_histo = histo[np.nonzero(histo)]  # Remove zeros so we can curve fit
    new_bins = bins[1:][np.nonzero(histo)]
    log_bins = np.log(new_bins)
    log_histo = np.log(new_histo)
    log_bins[np.where(log_bins == 0)] += 1e-10
    log_histo[np.where(log_histo == 0)] += 1e-10

    popt, pcov = curve_fit(func_powerlaw, log_bins[log_bins < 5], log_histo[log_bins < 5],
                           p0=[2, np.max(log_histo)])

    plt.figure()
    plt.pcolormesh(lat, cmap='Greys')
    plt.ylabel("y")
    plt.xlabel("x")
    plt.colorbar()
    plt.savefig("sand_pile_model.png")
    plt.show()

    # plt.figure()
    # plt.loglog(new_bins, new_histo, '.', label="Data")
    # plt.loglog(new_bins, np.exp(func_powerlaw(log_bins, *popt)), label="Fit - tau = %.3f $\pm$ %.3f" % (popt[0],
    #                                                                                                     np.diag(pcov)[0]))
    # plt.xlabel("Number of topples")
    # plt.ylabel("Occurences")
    # plt.legend()
    plt.figure()
    plt.loglog(new_bins[log_bins < 5], new_histo[log_bins < 5], '.', label="Data")
    plt.loglog(new_bins[log_bins < 5], np.exp(func_powerlaw(log_bins[log_bins < 5], *popt)),
               label="Fit - tau = %.3f $\pm$ %.3f" % (popt[0], np.diag(pcov)[0]))
    plt.xlabel("Number of topples")
    plt.ylabel("Occurences")
    plt.legend()
    plt.savefig("power_law_fit.png")
    plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.pcolormesh(lat, cmap='Greys')
    ax1.set_ylabel("y")
    ax1.set_xlabel("x")

    ax2.loglog(new_bins[log_bins < 5], new_histo[log_bins < 5], '.', label="Data")
    ax2.loglog(new_bins[log_bins < 5], np.exp(func_powerlaw(log_bins[log_bins < 5], *popt)),
                label="Fit - tau = %.3f $\pm$ %.3E" % (popt[0], np.diag(pcov)[0]))
    ax2.set_xlabel("Number of topples")
    ax2.set_ylabel("Occurences")
    ax2.legend()
    plt.savefig("sand_pile_model_and_fit.png")
    plt.show()


    print('Done w. 4')

    return new_bins, new_histo, popt, pcov, log_histo, log_bins, lat


def NS_Algo(L, N_cars, v_max, iterations, P=.0, pull_over=0.):
    def build_road(iters, L, N_cars):
        road = np.zeros([iters, L])
        road[0, :N_cars] = 1
        rng.shuffle(road[0, :])

        return road

    def find_empty_cells_in_front_of_car(lane, L):
        d = np.zeros([L])
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
    # print(np.sum(road))
    flux = np.zeros([iterations])
    d = np.copy(road)
    d[0, :] = find_empty_cells_in_front_of_car(road[0, :], L)
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
        ## Pull over
        if pull_over:
            randomize = rng.random([L])
            is_smaller = randomize < pull_over
            v_0 = v[t, :] == 0
            pullcars = is_smaller & v_0
            road[t-1, pullcars] = 0
            # d[t-1, pullcars] = 0
            # print(np.sum(road[t-1, :]))
        ## Car motion
        road[t, :], v[t, :] = move_cars(road[t-1, :], v[t, :])
        d[t, :] = find_empty_cells_in_front_of_car(road[t, :], L)
        ## Finding flux
        flux[t] = (np.sum(v[t, :] == 1) + 2 * np.sum(v[t, :] == 2)) / L

    return road, flux


def power_distribution(y_min, alpha, N=1):
    ## This returns a power distribution with a minimum value of y_min and a power of alpha
    ## N is the number of samples to return

    k = alpha * y_min**alpha
    x = rng.random(N)
    y = y_min / np.power(1 - x, 1/alpha) #  (1 / (y_min**(-alpha) - (alpha/k) * x))**(1/alpha)

    return y


def test_power_distribution(y_min, alpha):
    ## This tests the power distribution function by plotting a histogram of the results

    k = alpha * y_min**alpha
    numbers = np.logspace(3, 6, 4)
    mean_x = np.zeros(4)
    mean_theory = alpha*y_min/(alpha-1)
    for i in range(4):
        y = power_distribution(y_min, alpha, int(numbers[i]))
        mean_x[i] = np.mean(y)
    x = np.linspace(y_min, np.max(y), int(numbers[-1]))
    y_theory = k * x ** (-(alpha + 1))
    print(simpson(y_theory, x))
    print(mean_x)
    print(mean_theory)

    plt.figure()
    plt.hist(y, bins=100, density=True, label="Simulation")
    plt.plot(x, y_theory, 'r', label='Expected')
    # plt.title("Power Distribution Histogram")
    plt.ylabel("Count")
    plt.xlabel("y")
    plt.legend()
    plt.savefig("Power_Distribution_Histogram_alpha_" + str(alpha) + "_ymin_" + str(y_min) + ".png")
    plt.show()
    # plt.close()


def simulate_RW(L, iterations):
    # moves = rng.choice([-1, 1], size=iterations)
    t = np.zeros(iterations)

    for i in range(iterations):
        # print("Iteration: " + str(i+1))
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


def sand_pile_model(size_L, max_grain=3, iterations=1000):
    ## This function simulates the sand pile model on a lattice of size size_L x size_L

    def topple(lattice, y, x, it):
        toppled[it] += 1
        try:
            lattice[y+1, x] += 1
            if lattice[y+1, x] > max_grain:
                lattice[y+1, x] = 0
                topple(lattice, y+1, x, it)
        except IndexError:
            pass
        try:
            lattice[y-1, x] += 1
            if lattice[y-1, x] > max_grain:
                lattice[y-1, x] = 0
                topple(lattice, y-1, x, it)
        except IndexError:
            pass
        try:
            lattice[y, x+1] += 1
            if lattice[y, x+1] > max_grain:
                lattice[y, x+1] = 0
                topple(lattice, y, x+1, it)
        except IndexError:
            pass
        try:
            lattice[y, x-1] += 1
            if lattice[y, x-1] > max_grain:
                lattice[y, x-1] = 0
                topple(lattice, y, x-1, it)
        except IndexError:
            pass
        # print("topple")
        ## This function topples the lattice at a given site with open boundary conditions
        # if not ((x == L-1) or (x == 0) or (y == L-1) or (y == 0)):
        #     lattice[y+1, x] += 1
        #     if lattice[y+1, x] > max_grain:
        #         lattice[y+1, x] = 0
        #         topple(lattice, y+1, x, it)
        #     lattice[y-1, x] += 1
        #     if lattice[y-1, x] > max_grain:
        #         lattice[y-1, x] = 0
        #         topple(lattice, y-1, x, it)
        #     lattice[y, x+1] += 1
        #     if lattice[y, x+1] > max_grain:
        #         lattice[y, x+1] = 0
        #         topple(lattice, y, x+1, it)
        #     lattice[y, x-1] += 1
        #     if lattice[y, x-1] > max_grain:
        #         lattice[y, x-1] = 0
        #         topple(lattice, y, x-1, it)
        # elif (x == L-1) and (y == L-1):
        #     lattice[y-1, x] += 1
        #     if lattice[y-1, x] > max_grain:
        #         lattice[y-1, x] = 0
        #         topple(lattice, y-1, x, it)
        #     lattice[y, x-1] += 1
        #     if lattice[y, x-1] > max_grain:
        #         lattice[y, x-1] = 0
        #         topple(lattice, y, x-1, it)
        # elif (x == L-1) and (y == 0):
        #     lattice[y+1, x] += 1
        #     if lattice[y+1, x] > max_grain:
        #         lattice[y+1, x] = 0
        #         topple(lattice, y+1, x, it)
        #     lattice[y, x-1] += 1
        #     if lattice[y, x-1] > max_grain:
        #         lattice[y, x-1] = 0
        #         topple(lattice, y, x-1, it)
        # elif (x == 0) and (y == L-1):
        #     lattice[y-1, x] += 1
        #     if lattice[y-1, x] > max_grain:
        #         lattice[y-1, x] = 0
        #         topple(lattice, y-1, x, it)
        #     lattice[y, x+1] += 1
        #     if lattice[y, x+1] > max_grain:
        #         lattice[y, x+1] = 0
        #         topple(lattice, y, x+1, it)
        # elif (x == 0) and (y == 0):
        #     lattice[y+1, x] += 1
        #     if lattice[y+1, x] > max_grain:
        #         lattice[y+1, x] = 0
        #         topple(lattice, y+1, x, it)
        #     lattice[y, x+1] += 1
        #     if lattice[y, x+1] > max_grain:
        #         lattice[y, x+1] = 0
        #         topple(lattice, y, x+1, it)
        # elif (x == L-1):
        #     lattice[y+1, x] += 1
        #     if lattice[y+1, x] > max_grain:
        #         lattice[y+1, x] = 0
        #         topple(lattice, y+1, x, it)
        #     lattice[y-1, x] += 1
        #     if lattice[y-1, x] > max_grain:
        #         lattice[y-1, x] = 0
        #         topple(lattice, y-1, x, it)
        #     lattice[y, x-1] += 1
        #     if lattice[y, x-1] > max_grain:
        #         lattice[y, x-1] = 0
        #         topple(lattice, y, x-1, it)
        # elif (x == 0):
        #     lattice[y+1, x] += 1
        #     if lattice[y+1, x] > max_grain:
        #         lattice[y+1, x] = 0
        #         topple(lattice, y+1, x, it)
        #     lattice[y-1, x] += 1
        #     if lattice[y-1, x] > max_grain:
        #         lattice[y-1, x] = 0
        #         topple(lattice, y-1, x, it)
        #     lattice[y, x+1] += 1
        #     if lattice[y, x+1] > max_grain:
        #         lattice[y, x+1] = 0
        #         topple(lattice, y, x+1, it)
        # elif (y == L-1):
        #     lattice[y-1, x] += 1
        #     if lattice[y-1, x] > max_grain:
        #         lattice[y-1, x] = 0
        #         topple(lattice, y-1, x, it)
        #     lattice[y, x+1] += 1
        #     if lattice[y, x+1] > max_grain:
        #         lattice[y, x+1] = 0
        #         topple(lattice, y, x+1, it)
        #     lattice[y, x-1] += 1
        #     if lattice[y, x-1] > max_grain:
        #         lattice[y, x-1] = 0
        #         topple(lattice, y, x-1, it)
        # elif (y == 0):
        #     lattice[y+1, x] += 1
        #     if lattice[y+1, x] > max_grain:
        #         lattice[y+1, x] = 0
        #         topple(lattice, y+1, x, it)
        #     lattice[y, x+1] += 1
        #     if lattice[y, x+1] > max_grain:
        #         lattice[y, x+1] = 0
        #         topple(lattice, y, x+1, it)
        #     lattice[y, x-1] += 1
        #     if lattice[y, x-1] > max_grain:
        #         lattice[y, x-1] = 0
        #         topple(lattice, y, x-1, it)

        return lattice

    toppled = np.zeros(iterations)
    lattice = np.zeros([size_L, size_L])
    dump_sites = rng.integers(0, size_L, size=[iterations, 2])  # First column is y, second column is x

    for i in range(iterations):
        if i % 12500 == 0:
            print(i, 'out of', iterations)
        lattice[tuple(dump_sites[i])] += 1
        if lattice[tuple(dump_sites[i])] > max_grain:
            lattice[tuple(dump_sites[i])] = 0
            lattice = topple(lattice, *dump_sites[i], i)
    return lattice, toppled


def func_powerlaw(x, tau, c):
    return - tau * x + c


if __name__ == '__main__':
    ## Exercise 1:
    ex_1()
    # ## Exercise 2:
    # ex_2()
    # # Exercise 3:
    # mean_t = ex_3()
    # Exercise 4:
    # new_bins, new_histo, popt, pcov, log_histo, log_bins, pile = ex_4()
