import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
from scipy.integrate import simpson
from scipy.optimize import curve_fit
import dace as dc
import matplotlib.ticker as mtick

rng = default_rng()

##Question 2
# def cantor_cut(array, L, it):
#
#     # delta_true = L * (2 / 5)**it
#     delta_false = L * (1 / 5)**it
#     # print(int(L / delta_false) * int(delta_false))
#     mat = np.resize(array, [int(L / delta_false), int(delta_false)])
#     for i in range(int(L / delta_false)):
#         if not (i+1) % 3:
#             mat[i, :] = 0
#         else:
#             continue
#     new_array = np.resize(mat, L)
#     return new_array
def cantor_cut(array, len_array):
    seg_len = 2 * len_array // 5
    return_array = np.zeros([len_array])
    return_array[:seg_len] = array[:seg_len]
    return_array[-seg_len:] = array[-seg_len:]

    return return_array

def cantor_split(array):
    len_array = len(array)
    start_index = 0
    end_index = 1
    return_array = np.zeros([len_array])
    while end_index < len_array:
        # for i in array:
        if (array[end_index]) & (end_index < len_array - 1):
            end_index += 1
        else:
            if end_index == len_array - 1:
                end_index += 1
            return_array[start_index:end_index] = cantor_cut(array[start_index:end_index], end_index - start_index)
            start_index = end_index
            if end_index == len_array:
                break

            while not array[start_index]:
                start_index += 1
            end_index = start_index
    return return_array

def cantor_set(L, depth=5):
    cantor_matrix = np.zeros([depth, L], int)
    cantor_matrix[0, :] = 1
    for i in range(1, depth):
        cantor_matrix[i, :] = cantor_split(cantor_matrix[i-1, :])

    dim = np.sum(cantor_matrix) / (L * depth)

    plt.figure()
    plt.gca().invert_yaxis()
    plt.pcolormesh(cantor_matrix, cmap='bwr')
    plt.axis('off')
    plt.savefig(r'./Cantor_Set_L_{}.png'.format(L))
    plt.show()

    return cantor_matrix, dim

# def cantor_set(L, depth=5):
#     cantor_matrix = np.ones([depth, L], int)
#     for i in range(1, L):
#         cantor_matrix[i, :] = cantor_cut(cantor_matrix[i-1], L, i)
#         if not np.sum(cantor_matrix[i, :]):
#             final_cantor_matrix = cantor_matrix[:i, :]
#             dim = np.sum(final_cantor_matrix) / L ** 2
#             # dim = np.sum(cantor_matrix) / L ** 2
#             return final_cantor_matrix, dim
#             # return cantor_matrix, dim
#
#     dim = np.sum(cantor_matrix, axis=1) / L**2
#
#     return cantor_matrix, dim


# def cantor_ex():
#     num_l = 5
#     Ls = np.logspace(1, 4, num_l)
#     dims = np.zeros(num_l)
#     for L in range(num_l):
#         cantor_matrix, dims[L] = cantor_set(int(Ls[L]))
#         # print(cantor_matrix)
#
#         plt.figure()
#         plt.imshow(cantor_matrix)
#         plt.title('Cantor Set with L = {}'.format(Ls[L]))
#         plt.savefig(r'./Cantor_Set_L_{}.png'.format(int(Ls[L])))
#         plt.show()

#Question 3:
def exp_distribution(y_min, alpha, N=1):
    ## This returns a power distribution with a minimum value of y_min and a power of alpha
    ## N is the number of samples to return

    # k = alpha * np.exp(alpha * y_min)
    x = rng.random(N)
    y = -1 / alpha * np.log(1 - x) + y_min

    return y


def test_exp_distribution(y_min, alpha, N=10000):
    ## This tests the power distribution function by plotting a histogram of the results

    k = alpha * np.exp(alpha * y_min)

    y = exp_distribution(y_min, alpha, N)
    x = np.linspace(y_min, np.max(y), N)

    y_theory = k * np.exp(-alpha * x)

    print(simpson(y_theory, x))

    plt.figure()
    plt.hist(y, bins=100, density=True, label="Simulation")
    plt.plot(x, y_theory, 'r', label="Expected")
    plt.ylabel("Count")
    plt.xlabel("y")
    plt.legend()
    plt.savefig("Exponential_Distribution_Histogram_alpha_" + str(alpha) + "_ymin_" + str(y_min) + ".png")
    plt.show()


## Question 5:
def question_5():
    L_vec = 25
    iterations = np.array([500, 750, 1000, 2000, 5000])
    len_it = len(iterations)
    rec_it = 1 / iterations
    mean_exit_time, mean_return_time = np.zeros(len_it), np.zeros(len_it)
    prob_fr, prob_fe = np.zeros(len_it), np.zeros(len_it)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[12, 5], sharey=True)

    for j in range(len_it):
        print('it = {}'.format(iterations[j]))
        return_time, exit_time, tot_steps, mean_exit_time[j], mean_return_time[j], prob_fr[j], prob_fe[j] = \
            random_walker(L_vec, iterations[j])

    popt, pcov = curve_fit(linear, rec_it, prob_fr)

    ax1.plot(iterations, prob_fr, label='$P_{}$, L = {}'.format('fr', L_vec))
    ax1.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Probability')
    ax1.legend()

    ax2.plot(rec_it, prob_fr, 'r*', label='$P_{fr}$')
    ax2.plot(rec_it, linear(rec_it, *popt), 'b-', label='fit: a=%5.3f, b=%5.3f' % tuple(popt))
    ax2.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    ax2.set_xlabel('1/Iterations')
    ax2.set_ylabel('Probability')
    ax2.legend()

    plt.savefig('Probabilities_2.png')
    plt.show()

def random_walker(L: int, it: int):
    # rw_array = np.zeros([it, L], int)
    # rw_array[0, 0] = 1
    return_time = np.zeros(it)
    exit_time = np.zeros(it)
    tot_steps = np.zeros(it)
    for i in range(1, it):
        # print('Iteration: {}'.format(i))
        has_returned = False
        rw_index = 0
        steps = 0
        # choices = rng.choice([-1, 1], it)
        while rw_index < L:
            steps += 1
            rw_index += rng.choice([-1, 1])
            # rw_index += choices[steps % it]
            if (rw_index > 0) & (rw_index < L):
                continue
            elif rw_index < 0:
                rw_index = 1
            elif (steps != 0) & (rw_index == 0) & (not has_returned):
                has_returned = True
                return_time[i] = steps
            elif rw_index == L:
                exit_time[i] = steps
                tot_steps[i] = steps

    mean_exit_time = np.mean(exit_time)
    mean_return_time = np.mean(return_time)

    prob_fr = np.sum(return_time < exit_time) / it
    prob_fe = 1 - prob_fr

    return return_time, exit_time, tot_steps, mean_exit_time, mean_return_time, prob_fr, prob_fe

def linear(x, a, b):
    return a * x + b
##Question 6:
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
        if P > 0:
            randomize = rng.random([L])
            v[t, randomize < P] = np.maximum(v[t, randomize < P] - 1, 0)
        ## Car motion
        road[t, :], v[t, :] = move_cars(road[t-1, :], v[t, :])
        d[t, :] = find_empty_cells_in_front_of_car(d[t, :], road[t, :], L)
        ## Finding flux
        flux[t] = (np.sum(v[t, :] == 1) + 2 * np.sum(v[t, :] == 2)) / L

    return road, flux

def question_6():
    L_lane = 1000
    N_cars = np.array([100, 200, 333, 500, 666, 800, 900])
    v = np.linspace(0, 2, 3)
    v_max = 2
    iter_traffic = 1500
    P = 0

    means_no_slowing = np.zeros(len(N_cars))
    flux_no_slowing = np.zeros([len(N_cars), iter_traffic])
    # flux_linalg = np.zeros(len(N_cars))
    conc = np.zeros([len(N_cars), 3])

    for N in N_cars:
        road, flux_no_slowing[np.where(N_cars == N), :] = NS_Algo(L_lane, N, v_max, iter_traffic, P)
        means_no_slowing[np.where(N_cars == N)] = np.mean(flux_no_slowing[np.where(N_cars == N)])

        plt.figure()
        plt.pcolormesh(road)
        # plt.title("N = " + str(N) + " cars")
        plt.ylabel("Time")
        plt.xlabel("Road")
        plt.savefig(str(N) + "_cars_no_slowing_exam.png")
        plt.close()

        plt.figure()
        plt.plot(flux_no_slowing[np.where(N_cars == N)][0])
        # plt.title("flux vs time, N = " + str(N) + " cars, no slowing")
        plt.ylabel("flux")
        plt.xlabel("time")
        plt.savefig(str(N) + "_cars_flux_no_slowing_exam.png")
        plt.close()

        ## Calculate expected from mean-field theory (Schreckenberg, 1995, Phys. Rev, E 51, 2939)
        c = N / L_lane
        d = 1 - c
        # q = 1 - P

        conc[np.where(N_cars == N), 0] = c**2
        conc[np.where(N_cars == N), 1] = c**2 * d * (1 + d / (2 - d))
        conc[np.where(N_cars == N), 2] = c * d**2 / (2 - d)

    flux_linalg = np.sum(conc * v, axis=1)

    plt.figure()
    plt.plot(N_cars/L_lane, means_no_slowing, label="No slowing")
    plt.plot(N_cars/L_lane, flux_linalg, label="Rate equations")
    # plt.title("flux vs density")
    plt.ylabel("flux")
    plt.xlabel("density")
    plt.legend()
    plt.savefig("flux_vs_density_exam.png")
    plt.show()


if __name__ == '__main__':
    cantor_matrix, dim = cantor_set(12500, 5)
    D = np.log(2) / np.log(5/2)
    error = np.abs(dim - D) / D
    # question_5()
    # test_exp_distribution(0.001, 0.5, 100000)
    # test_exp_distribution(0.001, 10, 100000)
    # question_6()



    print('Done!')
