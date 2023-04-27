import numpy as np
from numpy.random import default_rng
from scipy.constants import k

rng = default_rng()


def total_energy(model, L, J=1):
    tot_e = 0
    for i in range(L):
        for j in range(L):
            tot_e += - J * model[i, j] * (model[i-1, j] + model[i, j-1])
    return tot_e


def flipper(model, spin, acceptor, energy, T):
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
        return model, energy + e_diff
    else:
        boltz_prob = np.exp(-e_diff / (k * T))
        if boltz_prob > acceptor:
            # temp_model = np.copy(model)
            model[*spin] *= -1
            return model, energy + e_diff
        else:
            return model, energy



def metropolis(starting_model, L, J, T, it):
    sys_energy = total_energy(starting_model, L, J)
    spin_choice = rng.integers(L, size=[it, 2])
    accept_values = rng.random(it)
    eq_model, sys_energy = flipper(starting_model, spin_choice[0], accept_values[0], sys_energy, T)
    loop = 1
    for spin in spin_choice[1:]:
        eq_model, sys_energy = flipper(eq_model, spin, accept_values[loop], sys_energy, T)
        loop += 1
    return eq_model, sys_energy


if __name__ == '__main__':
    L = int(1e2)
    starting_model = (rng.random([int(L), int(L)]) < 0.5) * 2 - 1
    J = 1
    it = int(1e7)
    T = 300
    E_start = total_energy(starting_model, L, J)
    eq_model, E_tot = metropolis(starting_model, L, J, T, it)
