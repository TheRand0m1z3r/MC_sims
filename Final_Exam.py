import numpy as np
import matplotlib.pyplot as plt


# Question 2
def cantor_cut(array, L, it):
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


if __name__ == '__main__':
    cantor_ex()
    print('Done!')