import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigh

# basic constants
a = 1
G1 = np.pi / a
hbar = 1

# potential strength
A = 3
V1 = A/2
V2 = A/4
V3 = A/6

# plane wave basis
basis = np.arange(-3,4)

# k values in Brillouin zone
k_vals = np.linspace(-np.pi/a, np.pi/a, 600)


# ---------------- Band structure calculation ----------------
bands = []

for k in k_vals:

    H = np.zeros((len(basis), len(basis)))

    for i, g in enumerate(basis):

        G = g * G1

        # kinetic energy
        H[i,i] = (hbar**2/2)*(k + G)**2

        # coupling from periodic potential
        if i > 0:
            H[i,i-1] = V1
            H[i-1,i] = V1

        if i > 1:
            H[i,i-2] = V2
            H[i-2,i] = V2

        if i > 2:
            H[i,i-3] = V3
            H[i-3,i] = V3

    eigvals,_ = eigh(H)
    bands.append(eigvals)

bands = np.array(bands)


# ---------------- Plot band structure ----------------
plt.figure(figsize=(10,10))

colors = ['navy','crimson','darkgreen','purple','orange','teal','brown']

for n in range(bands.shape[1]):
    plt.plot(k_vals, bands[:,n], color=colors[n % len(colors)], label=f'Band {n+1}')

plt.axvline(np.pi/(2*a), linestyle='--', color='black')
plt.axvline(-np.pi/(2*a), linestyle='--', color='black')

plt.axvline(2*np.pi/(2*a), linestyle='--', color='red')
plt.axvline(-2*np.pi/(2*a), linestyle='--', color='red')

plt.title('Electronic Band Dispersion for Nearly Free Electrons')
plt.ylim(-2,35)

plt.xlabel('Wave vector k (1/a)')
plt.ylabel('Energy')

plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)

plt.legend()
plt.grid(alpha=0.4)

plt.show()


# ---------------- Density of states ----------------
energies = bands.flatten()

E = np.linspace(min(energies), max(energies), 1000)

sigma = 0.1

dos = np.zeros_like(E)

for En in energies:
    dos += np.exp(-(E-En)**2/(2*sigma**2))

dos = dos/(sigma*np.sqrt(2*np.pi)*len(energies))


plt.figure(figsize=(6,8))

plt.plot(dos, E, color='darkblue')

plt.xlabel("Density of States")
plt.ylabel("Energy")

plt.title("Density of States obtained using Gaussian Broadening")

plt.grid(alpha=0.4)

plt.show()


# ---------------- Band structure + DOS ----------------
fig, ax = plt.subplots(1,2, figsize=(12,8))

for n in range(bands.shape[1]):
    ax[0].plot(k_vals, bands[:,n], color=colors[n % len(colors)])

ax[0].set_xlabel("k")
ax[0].set_ylabel("Energy")
ax[0].set_title("Energy Band Diagram")

ax[1].plot(dos, E, color='darkblue')
ax[1].set_xlabel("DOS")
ax[1].set_title("Electronic Density of States")

plt.show()


# ---------------- Function to compute bands ----------------
def band_structure(M, A_val, k_vals):

    G = np.pi/a

    V1 = A_val/2
    V2 = A_val/4
    V3 = A_val/6

    basis = np.arange(-M, M+1)

    bands = []

    for k in k_vals:

        H = np.zeros((len(basis), len(basis)))

        for i,g in enumerate(basis):

            Gk = g*G

            H[i,i] = (hbar**2/2)*(k + Gk)**2

            if i>0:
                H[i,i-1] = V1
                H[i-1,i] = V1

            if i>1:
                H[i,i-2] = V2
                H[i-2,i] = V2

            if i>2:
                H[i,i-3] = V3
                H[i-3,i] = V3

        eigvals,_ = eigh(H)
        bands.append(eigvals)

    return np.array(bands)


# ---------------- Convergence with basis size ----------------
M_vals = [0,1,2,3,4,5]

k_vals = np.linspace(-np.pi/a, np.pi/a, 600)

plt.figure(figsize=(12,10))

color_list = ['red','blue','green','purple','orange','brown']

for i, M in enumerate(M_vals):
    band_data = band_structure(M, A, k_vals)
    plt.plot(k_vals, band_data[:,0], color=color_list[i], label=f'M={M}')

plt.axvline(np.pi/(2*a), linestyle='--', color='black')
plt.axvline(-np.pi/(2*a), linestyle='--', color='black')

plt.axvline(2*np.pi/(2*a), linestyle='--', color='red')
plt.axvline(-2*np.pi/(2*a), linestyle='--', color='red')

plt.ylim(-0.75,-0.25)

plt.title('Convergence of First Energy Band with Basis Size M')

plt.xlabel('k (1/a)')
plt.ylabel('Energy')

plt.legend()
plt.grid(alpha=0.4)

plt.show()


# ---------------- Fermi energy convergence ----------------
k_list = [100,200,400,800,1000,2000,4000,8000,10000]

def fermi_energy(k_list, shift):

    Ef_vals = []

    for Nk in k_list:

        k_vals = np.linspace(-np.pi/a, np.pi/a, Nk)

        bands = band_structure(7, A, k_vals)

        energies = bands.flatten()
        energies.sort()

        Ef = energies[Nk//2 - shift]

        Ef_vals.append(Ef)

    return Ef_vals


Ef_HOMO = fermi_energy(k_list,1)
Ef_LUMO = fermi_energy(k_list,0)


plt.plot(k_list, Ef_HOMO, 'o-', color='darkgreen')

plt.xlabel("Number of k points")
plt.ylabel("Fermi Energy")

plt.title("HOMO Energy Convergence with k-point Sampling")

plt.grid(alpha=0.4)

plt.show()


plt.plot(k_list, Ef_LUMO, 'o-', color='darkred')

plt.xlabel("Number of k points")
plt.ylabel("Fermi Energy")

plt.title("LUMO Energy Convergence with k-point Sampling")

plt.grid(alpha=0.4)

plt.show()


plt.figure(figsize=(8,6))

plt.plot(k_list, Ef_HOMO, 'o-', color='darkgreen', label="HOMO")
plt.plot(k_list, Ef_LUMO, 's-', color='darkred', label="LUMO")

plt.xlabel("Number of k points")
plt.ylabel("Energy")

plt.title("Fermi Energy Convergence Study")

plt.legend()
plt.grid(alpha=0.4)

plt.show()


# ---------------- Modified potential case ----------------
V0 = A

k_vals = np.linspace(-np.pi/2*a, np.pi/2*a, 600)

bands2 = []

for k in k_vals:

    H = np.zeros((len(basis), len(basis)))

    for i,g in enumerate(basis):

        G = g * G1/2

        H[i,i] = (hbar**2/2)*(k + G)**2

        if i>0:
            H[i,i-1] = V0
            H[i-1,i] = V0

        if i>1:
            H[i,i-2] = V1
            H[i-2,i] = V1

        if i>2:
            H[i,i-3] = V2
            H[i-3,i] = V2

        if i>3:
            H[i,i-4] = V3
            H[i-4,i] = V3

    eigvals,_ = eigh(H)
    bands2.append(eigvals)

bands2 = np.array(bands2)


plt.figure(figsize=(10,10))

for n in range(bands2.shape[1]):
    plt.plot(k_vals, bands2[:,n], color=colors[n % len(colors)], label=f'Band {n+1}')

plt.axvline(np.pi/(4*a), linestyle='--', color='black')
plt.axvline(-np.pi/(4*a), linestyle='--', color='black')

plt.axvline(2*np.pi/(4*a), linestyle='--', color='red')
plt.axvline(-2*np.pi/(4*a), linestyle='--', color='red')

plt.title('Band Structure with Modified Periodic Potential (G/2)')

plt.xlim(-2,2)
plt.ylim(-2,15)

plt.xlabel('k (1/a)')
plt.ylabel('Energy')

plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)

plt.legend()
plt.grid(alpha=0.4)

plt.show()


# ---------------- DOS for modified potential ----------------
energies = bands2.flatten()

E_grid = np.linspace(min(energies), max(energies), 1000)

dos2 = np.zeros_like(E_grid)

for En in energies:
    dos2 += np.exp(-(E_grid-En)**2/(2*sigma**2))

dos2 = dos2/(sigma*np.sqrt(2*np.pi)*len(energies))


plt.figure(figsize=(6,8))

plt.plot(dos2, E_grid, color='purple')

plt.xlabel("Density of States")
plt.ylabel("Energy")

plt.title("DOS for Modified Potential Case")

plt.grid(alpha=0.4)

plt.show()


# ---------------- Total energy vs potential ----------------
M = 3
Nk = 1000

k_vals = np.linspace(-np.pi/a, np.pi/a, Nk)

A_vals = np.linspace(0,6,20)

E_total = []

for A_val in A_vals:

    bands = band_structure(M, A_val, k_vals)

    energies = bands.flatten()
    energies.sort()

    occ = energies[:Nk//2]

    Et = np.sum(occ)/Nk

    E_total.append(Et)


plt.plot(A_vals, E_total, 'o-', color='darkorange')

plt.xlabel("Potential Strength A")
plt.ylabel("Total Energy per Electron")

plt.title("Variation of Total Energy with Potential Strength")

plt.grid(alpha=0.4)

plt.show()