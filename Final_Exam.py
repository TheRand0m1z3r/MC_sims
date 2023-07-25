import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
from scipy.integrate import simpson
from scipy.optimize import curve_fit
import dace as dc
import matplotlib.ticker as mtick

rng = default_rng()

# Question 2
def cantor_cut(array, L, it):
    # Reflection about a line: s_n = s - 2(s \dot r)r

    # delta_true = L * (2 / 5)**it
    delta_false = L * (1 / 5)**it
    # print(int(L / delta_false) * int(delta_false))
    mat = np.resize(array, [int(L / delta_false), int(delta_false)])
    for i in range(int(L / delta_false)):
        if not (i+1) % 3:
            mat[i, :] = 0
        else:
            continue
    new_array = np.resize(mat, L)
    return new_array


def cantor_set(L):
    cantor_matrix = np.ones([L, L], int)
    for i in range(1, L):
        cantor_matrix[i, :] = cantor_cut(cantor_matrix[i-1], L, i)
        if not np.sum(cantor_matrix[i, :]):
            final_cantor_matrix = cantor_matrix[:i, :]
            dim = np.sum(final_cantor_matrix) / L ** 2
            return final_cantor_matrix, dim

    dim = np.sum(cantor_matrix, axis=1) / L**2

    return cantor_matrix, dim


def cantor_ex():
    num_l = 5
    Ls = np.logspace(1, 4, num_l)
    dims = np.zeros(num_l)
    for L in range(num_l):
        cantor_matrix, dims[L] = cantor_set(int(Ls[L]))
        print(cantor_matrix)

        plt.figure()
        plt.imshow(cantor_matrix)
        plt.title('Cantor Set with L = {}'.format(Ls[L]))
        plt.savefig(r'./Cantor_Set_L_{}.png'.format(int(Ls[L])))
        plt.show()

##Question 3:
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


if __name__ == '__main__':
    # cantor_ex()
    # question_5()
    test_exp_distribution(0.001, 0.5, 100000)
    test_exp_distribution(0.001, 10, 100000)



    print('Done!')
