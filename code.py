import numpy as np
import matplotlib.pyplot as plt


# PARAMETERS

E0 = -5.0      # on-site energy (eV)
beta = -1.0   # hopping parameter (eV)


# TIGHT-BINDING HAMILTONIAN

def tight_binding_chain(N, E0, beta):
    H = np.zeros((N, N))
    np.fill_diagonal(H, E0)
    for i in range(N - 1):
        H[i, i + 1] = beta
        H[i + 1, i] = beta
    eigvals, eigvecs = np.linalg.eigh(H)
    return eigvals, eigvecs

# FERMI LEVEL (spin degeneracy)

def fermi_energy(eigvals):
    N = len(eigvals)
    return 0.5 * (eigvals[N//2 - 1] + eigvals[N//2])

# DOS PLOTS

Ns_dos = [5, 10, 20, 50, 200]
bin_widths = [0.2, 0.1, 0.05]

for bw in bin_widths:
    plt.figure()
    for N in Ns_dos:
        eigvals, _ = tight_binding_chain(N, E0, beta)
        bins = np.arange(eigvals.min() - bw, eigvals.max() + bw, bw)
        hist, edges = np.histogram(eigvals, bins=bins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        plt.plot(centers, hist, label=f"N = {N}")
        EF = fermi_energy(eigvals)
        plt.axvline(EF, linestyle="--")

    plt.xlabel("Energy (eV)")
    plt.ylabel("Density of States (arb. units)")
    plt.title(f"DOS for 1D chain (bin width = {bw} eV)")
    plt.legend()
    plt.show()

# EIGENSTATE PLOTS

Ns_states = [5, 10, 20, 50]

for N in Ns_states:
    eigvals, eigvecs = tight_binding_chain(N, E0, beta)

    idx_min = 0
    idx_max = N - 1
    idx_homo = N//2 - 1
    idx_lumo = N//2

    states = {
        "Minimum energy state": idx_min,
        "HOMO": idx_homo,
        "LUMO": idx_lumo,
        "Maximum energy state": idx_max
    }

    for label, idx in states.items():
        plt.figure()
        plt.plot(np.arange(1, N + 1), eigvecs[:, idx], marker='o')
        plt.xlabel("Site index")
        plt.ylabel("Eigenstate amplitude")
        plt.title(f"N = {N} | {label} | E = {eigvals[idx]:.3f} eV")
        plt.show()




