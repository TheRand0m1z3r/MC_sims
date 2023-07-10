import numpy as np
from numba import prange
from numpy.random import default_rng
import matplotlib.pyplot as plt
# from scipy.constants import k

rng = default_rng()
k = 1

class IsingModel:
    def __init__(self, T, J, L, it):
        # self.starting_model = (rng.random([int(L), int(L)]) < 0.5) * 2 - 1
        self.starting_model = np.ones([L, L]) * -1
        self.model = np.copy(self.starting_model)
        self.M_start = np.sum(self.starting_model)
        self.magnet = np.copy(self.M_start)
        self.J = J
        self.L = L
        self.T_c = 2*J / (k * np.log(1 + np.sqrt(2)))
        # self.t = (T - self.T_c) / self.T_c
        self.t = T
        # self.E_start = total_energy(self.starting_model, L, J)
        self.energy = 0
        self.flip_cells = np.zeros([1, 2])
        if it:
            self.iter = it
        else:
            self.iter = int(1e6)

    def total_energy(self):
        tot_e = 0
        for i in prange(self.L):
            for j in prange(self.L):
                nbr = (np.array([[1, 0], [-1, 0], [0, 1], [0, -1]]) + [i, j]) % self.L
                tot_e += - self.J * self.model[i, j] * np.sum(self.model[nbr[:, 0], nbr[:, 1]])
        self.energy = tot_e / 2

# def find_equil_time(starting_model, J, T):
#     return

    def flipper(self, spin, acceptor):
        vec = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        nn = (spin+vec) % self.L
        e_diff = np.sum(self.model[nn[:, 0], nn[:, 1]]) * self.model[*spin] * 2
        # print(e_diff)
        if e_diff <= 0:
            self.model[*spin] *= -1
            self.energy += e_diff
            self.magnet += self.model[*spin]
        else:
            boltz_prob = np.exp(-e_diff / (k * self.t))
            if boltz_prob > acceptor:
                self.model[*spin] *= -1
                self.energy += e_diff
                self.magnet += self.model[*spin]
            else:
                pass
                # print("Didn't flip")

    def metropolis(self):
        magnetiz, energyz = np.zeros(self.iter), np.zeros(self.iter)
        # sys_energy = total_energy()
        ## Finding equilibration time
        equilibrated = False
        # while not equilibrated:
        #     equilibrated = find_equil_time(starting_model, J, T)
        spin_choice = rng.integers(L, size=[it, 2])
        accept_values = rng.random(it)
        # eq_model, sys_energy, M =
        # self.flipper(spin_choice[0], accept_values[0], sys_energy, T, M_start)
        loop = 0
        for spin in spin_choice:
            # eq_model, sys_energy, M =
            self.flipper(spin, accept_values[loop])
            magnetiz[loop] = self.magnet
            energyz[loop] = self.energy
            loop += 1
        return magnetiz, energyz

    def update_E_M(self):
        ## Updating magnetization
        k = len(self.flip_cells)
        self.magnet += k * self.model[*self.flip_cells[0]]
        ## Updating energy
        self.total_energy()
        # m, n = 0, 0
        # for cell in self.flip_cells:
        #     nbr = (np.array([[1, 0], [-1, 0], [0, 1], [0, -1]]) + cell) % self.L
        #     count_same = np.sum(self.model[*cell] == self.model[nbr[:, 0], nbr[:, 1]])
        #     if count_same == 4:
        #         continue
        #     else:
        #         m += count_same
        #         n += 4 - count_same
        # self.energy += 2 * self.J * (m - n)
        return

    def flip_wolff(self, cell):
        P_add = 1 - np.exp(-2 * self.J / (k * self.t))
        nbr = (np.array([[1, 0], [-1, 0], [0, 1], [0, -1]]) + cell) % self.L
        P_check = rng.random(4)
        for i in prange(4):
            if (P_check[i] <= P_add) & (self.model[nbr[i, 0], nbr[i, 1]] == self.model[cell[0], cell[1]])\
                    & (nbr not in self.flip_cells):
                self.flip_cells = np.vstack((self.flip_cells, nbr[i]))
        # if self.model[(cell[0] + 1) % self.L, cell[1]] == self.model[cell[0], cell[1]]:
        #     if P_check[0] < P_add[0]:
        #         self.flip_cells = np.vstack((self.flip_cells, [(cell[0] + 1) % self.L, cell[1]]))
        #     # else:
        #     #     print("Didn't flip")
        # if self.model[(cell[0] - 1) % self.L, cell[1]] == self.model[cell[0], cell[1]]:
        #     P_add[1] = 1 - np.exp(-2 * self.J / (k * self.t))
        #     if P_check[1] < P_add[1]:
        #         self.flip_cells = np.vstack((self.flip_cells, [(cell[0] - 1) % self.L, cell[1]]))
        #     # else:
        #     #     print("Didn't flip")
        # if self.model[cell[0], (cell[1] + 1) % self.L] == self.model[cell[0], cell[1]]:
        #     P_add[2] = 1 - np.exp(-2 * self.J / (k * self.t))
        #     if P_check[2] < P_add[2]:
        #         self.flip_cells = np.vstack((self.flip_cells, [cell[0], (cell[1] + 1) % self.L]))
        #     # else:
        #     #     print("Didn't flip")
        # if self.model[cell[0], (cell[1] - 1) % self.L] == self.model[cell[0], cell[1]]:
        #     P_add[3] = 1 - np.exp(-2 * self.J / (k * self.t))
        #     if P_check[3] < P_add[3]:
        #         self.flip_cells = np.vstack((self.flip_cells, [cell[0], (cell[1] - 1) % self.L]))
        #     # else:
        #     #     print("Didn't flip")
        return

    def wolff(self):
        energies = np.zeros(self.iter)
        magneties = np.zeros(self.iter)
        cell_choice = rng.integers(L, size=[self.iter, 2])
        for it in prange(self.iter):
        # for cell in cell_choice:
            self.flip_cells = np.array([cell_choice[it]])
            for flip in self.flip_cells:
                self.flip_wolff(flip)

            self.model[self.flip_cells[:, 0], self.flip_cells[:, 1]] *= -1
            self.update_E_M()
            energies[it] = self.energy
            magneties[it] = self.magnet


        return energies, magneties, np.mean(energies), np.mean(magneties)


if __name__ == '__main__':

    L = int(20)
    # starting_model = (rng.random([L, L]) < 0.5) * 2 - 1
    # M_start = np.sum(starting_model)
    J = 1
    it = int(1e4)
    timer = np.linspace(0, it, it)
    ## T_C is somewhere around 2.269
    # # Tc = 2J / (k * np.log(1 + np.sqrt(2)))
    T = np.linspace(2, 3, 6)
    # # t = (T - Tc) / Tc
    # E_start = total_energy(starting_model, L, J)
    # eq_model, E_tot, M = metropolis(starting_model, L, J, T, it, M_start)
    for t in T:
        print(t)
        model_ising = IsingModel(t, J, L, it)
        mag_ising, ene_ising = model_ising.metropolis()

        # plt.figure()
        # plt.plot(timer, mag_ising, label='magnet')
        # plt.title(f'T = {t}, Ising')
        # plt.legend()
        # # plt.show()
        # #
        # # plt.figure()
        # plt.plot(timer, ene_ising, label='energy')
        # plt.title(f'T = {t}, Ising')
        # plt.legend()
        # plt.show()

        # plt.figure()
        # plt.pcolormesh(model_ising.model)
        # plt.title(f'T = {t}, Ising')
        # plt.show()

        model_wolff = IsingModel(t, J, L, it)
        Es, Ms, E, M = model_wolff.wolff()
        print(f'E = {E}, M = {M}')

        plt.figure()
        plt.plot(timer, Ms, label='magnet')
        plt.title(f'T = {t}, Wolff')
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(timer, Es, label='energy')
        plt.title(f'T = {t}, Wolff')
        plt.legend()
        plt.show()

        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.pcolormesh(model_wolff.model)
        ax2.pcolormesh(model_wolff.starting_model)
        plt.suptitle(f'T = {t}, Wolff')
        plt.show()
