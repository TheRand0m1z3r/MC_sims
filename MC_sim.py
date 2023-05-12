import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
# from scipy.constants import k

rng = default_rng()
k = 1

class IsingModel:
    def __init__(self, T, J, L, it):
        self.starting_model = (rng.random([int(L), int(L)]) < 0.5) * 2 - 1
        self.model = np.zeros([int(L), int(L)])
        self.M_start = np.sum(self.starting_model)
        self.magnet = 0
        self.J = J
        self.T_c = 2*J / (k * np.log(1 + np.sqrt(2)))
        self.t = (T - self.T_c) / self.T_c
        # self.E_start = total_energy(self.starting_model, L, J)
        self.energy = 0
        if it:
            self.iter = it
        else:
            self.iter = int(1e6)

def total_energy(model, L, J=1):
    tot_e = 0
    for i in range(L):
        for j in range(L):
            tot_e += - J * model[i, j] * (model[i-1, j] + model[i, j-1])
    return tot_e


def find_equil_time(starting_model, J, T):
    return


def flipper(model, spin, acceptor, energy, T, M):
    vec = [[1, 0], [-1, 0], [0, 1], [0, -1]]
    nn = spin+vec
    e_diff = 0
    for n in nn:
        try:
            e_diff += model[*n]
        except IndexError:
            n[n == L] = 0
            e_diff += model[*n]
    e_diff *= model[*spin] * 2
    if e_diff <= 0:
        # temp_model = np.copy(model)
        model[*spin] *= -1
        return model, energy + e_diff, M + model[*spin]
    else:
        boltz_prob = np.exp(-e_diff / (k * T))
        if boltz_prob > acceptor:
            # temp_model = np.copy(model)
            model[*spin] *= -1
            return model, energy + e_diff, M + model[*spin]
        else:
            return model, energy, M


def metropolis(starting_model, L, J, T, it, M_start):
    sys_energy = total_energy(starting_model, L, J)
    ## Finding equilibration time
    equilibrated = False
    while not equilibrated:
        equilibrated = find_equil_time(starting_model, J, T)
    spin_choice = rng.integers(L, size=[it, 2])
    accept_values = rng.random(it)
    eq_model, sys_energy, M = flipper(starting_model, spin_choice[0], accept_values[0], sys_energy, T, M_start)
    loop = 1
    for spin in spin_choice[1:]:
        eq_model, sys_energy, M = flipper(eq_model, spin, accept_values[loop], sys_energy, T, M)
        loop += 1
    return eq_model, sys_energy, M


if __name__ == '__main__':
    L = int(1e2)
    starting_model = (rng.random([L, L]) < 0.5) * 2 - 1
    M_start = np.sum(starting_model)
    J = 1
    it = int(1e7)
    # Tc = 2J / (k * np.log(1 + np.sqrt(2)))
    T = 300
    # t = (T - Tc) / Tc
    E_start = total_energy(starting_model, L, J)
    eq_model, E_tot, M = metropolis(starting_model, L, J, T, it, M_start)
