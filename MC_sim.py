import numpy as np
from numba import prange, jit
from numpy.random import default_rng
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
# from scipy.constants import k

rng = default_rng()
k = 1

# Reflection about a line: s_n = s - 2(s \dot r)r

class IsingModel:
    def __init__(self, T, J, L, it, xy=False):
        # self.starting_model = (rng.random([int(L), int(L)]) < 0.5) * 2 - 1
        if not xy:
            self.starting_model = np.ones([L, L])
            self.M_start = np.sum(self.starting_model)
        else:
            self.starting_model = rng.random([int(L), int(L)]) * 2 * np.pi
            self.M_start = np.sum(np.sin(self.starting_model))

        self.model = np.copy(self.starting_model)
        self.magnet = np.copy(self.M_start)
        self.J = J
        self.L = L
        self.T_c = 2*J / (k * np.log(1 + np.sqrt(2)))
        # self.t = (T - self.T_c) / self.T_c
        self.t = T
        self.energy = 0
        self.flip_cells = np.zeros([1, 2])
        if it:
            self.iter = it
        else:
            self.iter = int(1e6)

    def total_energy(self, xy=False):
        for i in prange(self.L):
            for j in prange(self.L):
                nbr = (np.array([[1, 0], [0, 1]]) + [i, j]) % self.L
                if not xy:
                    self.energy -= self.J * self.model[i, j] * np.sum(self.model[nbr[:, 0], nbr[:, 1]])
                else:
                    self.energy -= self.J * np.sum(np.matmul([np.cos(self.model[i, j]), np.sin(self.model[i, j])],
                                                             [np.cos(self.model[nbr[:, 0], nbr[:, 1]]),
                                                              np.sin(self.model[nbr[:, 0], nbr[:, 1]])]))

        return

    def flipper(self, spin, acceptor, xy=False):
        vec = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        nn = (spin+vec) % self.L
        if not xy:
            e_diff = self.model[*spin] * np.sum(self.model[nn[:, 0], nn[:, 1]]) * 2 * self.J
        else:
            theta_plane = rng.random() * 2 * np.pi
            old_e = self.J * np.sum(np.matmul([np.cos(self.model[*spin]), np.sin(self.model[*spin])],
                                    [np.cos(self.model[nn[:, 0], nn[:, 1]]), np.sin(self.model[nn[:, 0], nn[:, 1]])]))
            delta_theta = theta_plane - self.model[*spin]
            new_theta = self.model[*spin] + 2 * delta_theta
            new_e = self.J * np.sum(np.matmul([np.cos(new_theta), np.sin(new_theta)],
                                    [np.cos(self.model[nn[:, 0], nn[:, 1]]), np.sin(self.model[nn[:, 0], nn[:, 1]])]))
            e_diff = new_e - old_e
        boltz_prob = np.exp(-e_diff / (k * self.t))
        if e_diff <= 0 or acceptor < boltz_prob:
            if not xy:
                self.model[*spin] *= -1
                self.magnet += self.model[*spin]
            else:
                self.magnet += np.sin(new_theta) - np.sin(self.model[*spin])
                self.model[*spin] = new_theta
            self.energy += e_diff

    def metropolis(self, xy=False):
        magnetiz, energyz = np.zeros(int(self.iter*0.9)), np.zeros(int(self.iter*0.9))
        spin_choice = rng.integers(self.L, size=[self.iter, 2])
        accept_values = rng.random(self.iter)
        # eq_model, sys_energy, M =
        # self.flipper(spin_choice[0], accept_values[0], sys_energy, T, M_start)
        self.total_energy(xy)
        loop = 0
        ## equilibrate
        for spin in spin_choice[:int(0.1 * self.iter)]:
            self.flipper(spin, accept_values[loop], xy)

        ## simulate
        for spin in spin_choice[int(0.1 * self.iter):]:
            self.flipper(spin, accept_values[loop], xy)
            magnetiz[loop] = self.magnet
            energyz[loop] = self.energy
            loop += 1

        energyz /= self.L**2
        magnetiz /= self.L**2
        U = np.mean(energyz)
        M = np.mean(magnetiz)
        C_v = np.var(energyz) / (k * self.t**2)
        chi = np.var(magnetiz) / (k * self.t)
        return energyz, magnetiz, U, M, C_v, chi

    def update_E_M(self):
        ## Updating magnetization
        k = len(self.flip_cells)
        self.magnet = np.abs(self.magnet + k * self.model[*self.flip_cells[0]])
        ## Updating energy
        m, n = 0, 0
        for cell in self.flip_cells:
            nbr = (np.array([[1, 0], [-1, 0], [0, 1], [0, -1]]) + cell) % self.L
            for i in prange(4):
                # print((nbr[i, 0] == self.flip_cells[:, 0]) & (nbr[i, 1] == self.flip_cells[:, 1]), (nbr[i,
                # 0] == self.flip_cells[:, 0]), (nbr[i, 1] == self.flip_cells[:, 1]))
                if not (np.any(nbr[i, 0] == self.flip_cells[:, 0]) & np.any(nbr[i, 1] == self.flip_cells[:, 1])):
                    if self.model[*nbr[i]] == self.model[*cell]:
                        m += 1
                    else:
                        n += 1
                else:
                    continue
        self.energy += 2 * self.J * (m - n)
        return

    def flip_wolff(self, cell):
        P_add = 1 - np.exp(-2 * self.J * self.T_c / (k * (self.t - self.T_c)))
        nbr = (np.array([[1, 0], [-1, 0], [0, 1], [0, -1]]) + cell) % self.L
        P_check = rng.random(4)
        for i in prange(4):
            if (P_check[i] <= P_add) and (self.model[*nbr[i]] == self.model[*cell])\
                    and not (np.any(nbr[i, 0] == self.flip_cells[:, 0]) & np.any(nbr[i, 1] == self.flip_cells[:, 1])):
                self.flip_cells = np.vstack((self.flip_cells, nbr[i]))
        return

    def wolff(self):
        energies = np.zeros(int(self.iter * 0.9))
        magneties = np.zeros(int(self.iter * 0.9))
        cell_choice = rng.integers(self.L, size=[self.iter, 2])
        for it_equilibrate in prange(int(self.iter*0.1)):
            self.flip_cells = np.array([cell_choice[it_equilibrate]])
            for flip in self.flip_cells:
                self.flip_wolff(flip)

            self.model[self.flip_cells[:, 0], self.flip_cells[:, 1]] *= -1

        self.total_energy()

        for it in prange(int(self.iter*0.9)):
            self.flip_cells = np.array([cell_choice[it]])
            for flip in self.flip_cells:
                self.flip_wolff(flip)

            self.model[self.flip_cells[:, 0], self.flip_cells[:, 1]] *= -1
            self.update_E_M()
            energies[it] = self.energy
            magneties[it] = self.magnet

        energies /= self.L**2
        magneties /= self.L**2
        U = np.mean(energies)
        M = np.mean(magneties)
        C_v = np.var(energies) * self.T_c**2 / (k * (self.t - self.T_c) ** 2)
        chi = np.var(magneties) * self.T_c / (k * (self.t - self.T_c))
        return energies, magneties, U, M, C_v, chi

def simulate_Ising(J, it, Ts = 26):
    L = [20, 50, 75, 100, 150]
    Ls = len(L)
    Tc = np.zeros(Ls)
    T = np.linspace(0.1, 10, Ts)
    Ts = len(T)
    spec_ene, spec_mag = np.zeros([Ls, Ts, int(it * 0.9)]), np.zeros([Ls, Ts, int(it * 0.9)])
    Es, Ms, Cs, Chis = np.zeros([Ls,Ts]), np.zeros([Ls,Ts]), np.zeros([Ls,Ts]), np.zeros([Ls,Ts])
    for l_ind in prange(Ls):
        for t_ind in prange(Ts):
            print(T[t_ind])
            model_ising = IsingModel(T[t_ind], J, L[l_ind], it)
            spec_ene[l_ind, t_ind], spec_mag[l_ind, t_ind], Es[l_ind, t_ind], Ms[l_ind, t_ind], Cs[l_ind, t_ind], \
            Chis[l_ind, t_ind] = model_ising.metropolis()
            print(f'E = {Es[l_ind, t_ind]}, M = {Ms[l_ind, t_ind]}, C_v = {Cs[l_ind, t_ind]}, Chi = {Chis[l_ind, t_ind]}')
        Tc[l_ind] = T[np.nonzero(Chis[l_ind] == np.max(Chis[l_ind]))]
    return spec_ene, spec_mag, Es, Ms, Cs, Chis, Tc

def simulate_wolff(J, it, Ts = 26):
    L = [20, 50, 75, 100, 150]
    Ls = len(L)
    Tc = np.zeros(Ls)
    for l_ind in prange(Ls):
        T = np.linspace(0.1, 10, Ts)
        spec_ene, spec_mag = np.zeros([Ts, int(it * 0.9)]), np.zeros([Ts, int(it * 0.9)])
        Es, Ms, Cs, Chis = np.zeros(Ts), np.zeros(Ts), np.zeros(Ts), np.zeros(Ts)
        for t_ind in prange(Ts):
            print(T[t_ind])
            model_wolff = IsingModel(T[t_ind], J, L[l_ind], it)
            spec_ene[t_ind], spec_mag[t_ind], Es[t_ind], Ms[t_ind], Cs[t_ind], Chis[t_ind] = model_wolff.wolff()
            print(f'E = {Es[t_ind]}, M = {Ms[t_ind]}')
        Tc[l_ind] = T[Chis == np.max(Chis)]

    return spec_ene, spec_mag, Es, Ms, Cs, Chis, Tc

def find_beta(m, tc, t):
    popt, pcov = curve_fit(linear, np.log(tc - t[t<tc]), np.log(m[t<tc]))
    return popt[0]
    # num_l = len(tc)
    # beta = np.zeros(num_l)
    # for n in prange(num_l):
    #     popt, pcov = curve_fit(linear, np.log(tc[n] - t[t<tc[n]]), np.log(m[t<tc[n]]))
    #     beta[n] = popt[0]
    # return beta
def linear(x, a, b):
    return a * x + b

def xy_model():
    return


if __name__ == '__main__':
    T = np.linspace(0.1, 10, 11)
    L_checked = np.array([20, 50, 75, 100, 150])
    rec_L = 1 / L_checked
    J = 1
    it_ising = int(5e5)
    it_wolff = int(5e4)
    ene_list_ising, mag_list_ising, U_ising, M_ising, C_v_ising, chi_ising, Tc_ising = simulate_Ising(J, it_ising,11)
    ene_list_wolff, mag_list_wolff, U_wolff, M_wolff, C_v_wolff, chi_wolff, Tc_wolff = simulate_wolff(J, it_wolff, 11)

    print(f'The crit temps found from the metropolis algo are:\nL - {L_checked}\nT - {Tc_ising}')
    print(f'The crit temps found from the wolff algo are:\nL - {L_checked}\nT - {Tc_wolff}')

    popt_ising, pcov_ising = curve_fit(linear, rec_L, Tc_ising)
    popt_wolff, pcov_wolff = curve_fit(linear, rec_L, Tc_wolff)

    Tc_inf_ising = linear(0, *popt_ising)
    Tc_inf_wolff = linear(0, *popt_wolff)

    print(f'For the Metropolis algo, Tc for an infinite lattice is: {Tc_inf_ising}')
    print(f'For the Wolff algo, Tc for an infinite lattice is: {Tc_inf_wolff}')

    beta_ising = find_beta(np.mean(M_ising, axis=0), Tc_inf_ising, T)
    beta_wolff = find_beta(M_wolff, Tc_inf_wolff, T)

    print(f'For the Metropolis algo, beta is: {beta_ising}')
    print(f'For the Wolff algo, beta is: {beta_wolff}')

    plt.figure()
    plt.plot(rec_L, Tc_ising, 'o', label='Ising')
    plt.plot(rec_L, linear(rec_L, *popt_ising), label='Ising  - fit')
    plt.plot(rec_L, Tc_wolff, 'o', label='Wolff')
    plt.plot(rec_L, linear(rec_L, *popt_wolff), label='Wolff  - fit')
    plt.xlabel(r'$1/L$')
    plt.ylabel(r'$T_c$')
    plt.legend()
    plt.savefig(r'./Tc_L.png')
    plt.show()

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    for i in range(len(L_checked)):
        ax1.plot(T, M_ising[i], '*', label='Ising - L = ' + str(L_checked[i]))
    ax2.plot(T, np.sign(M_ising)[0, 0]*M_wolff, '*', label='Wolff')
    ax1.set_ylabel('M')
    ax1.legend()
    ax2.set_xlabel('T')
    ax2.set_ylabel('M')
    ax2.legend()
    plt.savefig(r'./magnetization.png')
    plt.show()

    U_wolff[U_wolff < 0] *= -1
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    for i in range(len(L_checked)):
        ax1.plot(T, U_ising[i], '*', label='Ising - L = ' + str(L_checked[i]))
    ax1.set_ylabel('U')
    ax1.legend()
    ax2.plot(T, U_wolff, '*', label='Wolff')
    ax2.set_xlabel('T')
    ax2.set_ylabel('U')
    ax2.legend()
    plt.savefig(r'./energy.png')
    plt.show()

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    for i in range(len(L_checked)):
        ax1.plot(T, C_v_ising[i], '*', label='Ising - L = ' + str(L_checked[i]))
    ax1.set_ylabel('$C_v$')
    ax1.legend()
    ax2.plot(T, C_v_wolff, '*', label='Wolff')
    ax2.set_xlabel('T')
    ax2.set_ylabel('$C_v$')
    ax2.legend()
    plt.savefig(r'./specific_heat.png')
    plt.show()

    chi_wolff[chi_wolff < 0] *= -1
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    for i in range(len(L_checked)):
        ax1.plot(T, chi_ising[i, :], '*', label='Ising - L = ' + str(L_checked[i]))
    ax1.set_ylabel('$\chi$')
    ax1.legend()
    ax2.plot(T, chi_wolff, '*', label='Wolff')
    ax2.set_xlabel('T')
    ax2.set_ylabel('$\chi$')
    ax2.legend()
    plt.savefig(r'./susceptibility.png')
    plt.show()

    #
    # timer_ising = np.linspace(1, it, it)
    # timer_wolff = np.linspace(it*0.1+1, it, int(it*.9))
    #
    # ## T_C is somewhere around 2.269
    # # # Tc = 2J / (k * np.log(1 + np.sqrt(2)))
    # T = np.linspace(2.4, 3, 10)
    # # # t = (T - Tc) / Tc
    # # E_start = total_energy(starting_model, L, J)
    # # eq_model, E_tot, M = metropolis(starting_model, L, J, T, it, M_start)
    # for t in T:
    #     print(t)
    #     model_ising = IsingModel(t, J, L, it)
    #     mag_ising, ene_ising = model_ising.metropolis()
    #
    #     plt.figure()
    #     plt.plot(timer_ising, mag_ising, label='magnet')
    #     plt.title(f'T = {t}, Ising')
    #     plt.legend()
    #     plt.show()
    #
    #     # plt.figure()
    #     # plt.plot(timer_ising, ene_ising, label='energy')
    #     # plt.title(f'T = {t}, Ising')
    #     # plt.legend()
    #     # plt.show()
    #     #
    #     # plt.figure()
    #     # plt.pcolormesh(model_ising.model)
    #     # plt.title(f'T = {t}, Ising')
    #     # plt.show()
    #
    #     model_wolff = IsingModel(t, J, L, it)
    #     Es, Ms, E, M = model_wolff.wolff()
    #     print(f'E = {E}, M = {M}')
    #
    #     plt.figure()
    #     plt.plot(timer_wolff, Ms, label='magnet')
    #     plt.title(f'T = {t}, Wolff')
    #     plt.legend()
    #     plt.show()
    #
    #     # plt.figure()
    #     # plt.plot(timer_wolff, Es, label='energy')
    #     # plt.title(f'T = {t}, Wolff')
    #     # plt.legend()
    #     # plt.show()
    #
    #     # fig, (ax1, ax2) = plt.subplots(1, 2)
    #     # ax1.pcolormesh(model_wolff.model)
    #     # ax2.pcolormesh(model_wolff.starting_model)
    #     # plt.suptitle(f'T = {t}, Wolff')
    #     # plt.show()
