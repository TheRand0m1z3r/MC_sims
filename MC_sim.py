import numpy as np
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
        self.t = (T - self.T_c) / self.T_c
        # self.E_start = total_energy(self.starting_model, L, J)
        self.energy = 0
        if it:
            self.iter = it
        else:
            self.iter = int(1e6)

    def total_energy(self):
        tot_e = 0
        for i in range(self.L):
            for j in range(self.L):
                tot_e += - J * self.model[i, j] * (self.model[i-1, j] + self.model[i, j-1])
        self.energy = tot_e


# def find_equil_time(starting_model, J, T):
#     return


    def flipper(self, spin, acceptor):
        vec = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        nn = spin+vec
        e_diff = 0
        for n in nn:
            try:
                e_diff += self.model[*n]
            except IndexError:
                n[n == self.L] = 0
                e_diff += self.model[*n]
        e_diff *= self.model[*spin] * 2
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
                print("Didn't flip")


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


if __name__ == '__main__':

    L = int(1e2)
    # starting_model = (rng.random([L, L]) < 0.5) * 2 - 1
    # M_start = np.sum(starting_model)
    J = 1
    it = int(1000)
    timer = np.linspace(0, it, it)
    # # Tc = 2J / (k * np.log(1 + np.sqrt(2)))
    T = 500
    # # t = (T - Tc) / Tc
    # E_start = total_energy(starting_model, L, J)
    # eq_model, E_tot, M = metropolis(starting_model, L, J, T, it, M_start)
    model = IsingModel(T, J, L, it)
    mag, ene = model.metropolis()

    plt.plot(timer, mag, label='magnet')
    plt.legend()
    plt.show()
    plt.figure()
    plt.plot(timer, ene, label='energy')
    plt.legend()
    plt.show()
