import numpy as np
import matplotlib.pyplot as plt

# Basic setup
a = 1.0
k = np.linspace(-6*np.pi/a, 6*np.pi/a, 600)

#  PLOT 1: For periodicity a
fig1 = plt.figure(figsize=(11, 7))

# Background parabola
plt.plot(k*a/np.pi, k**2, 'slategray', linestyle=':', alpha=0.4, label='Free electron')

# Different color scheme
band_colors = ['#E63946', '#F4A261', '#2A9D8F', '#264653', '#E9C46A', '#4361EE', '#7209B7']
m_list = [-3, -2, -1, 0, 1, 2, 3]

for idx, m_val in enumerate(m_list):
    shift = m_val * 2 * np.pi / a
    energy = (k + shift)**2
    plt.plot(k*a/np.pi, energy, color=band_colors[idx],
             linewidth=2.2, label=f'Band m={m_val}')

# Zone edges
plt.axvline(x=-1, color='maroon', linestyle='--', alpha=0.6, linewidth=1.8)
plt.axvline(x=1, color='maroon', linestyle='--', alpha=0.6, linewidth=1.8)

plt.xlabel('Wavevector k (π/a)', fontsize=13)
plt.ylabel('Energy E (ħ²/2m)', fontsize=13)
plt.title('Energy Bands for Lattice Period a', fontsize=15)
plt.xlim(-6, 6)
plt.ylim(0, 140)
plt.grid(alpha=0.25, linestyle='-')
plt.legend(loc='upper left', fontsize=10)
plt.tight_layout()
plt.savefig('bands_period_a.png', dpi=150)
plt.show()





# PLOT 2: Comparison
fig2, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6))

# Left side - period a
ax_left.plot(k*a/np.pi, k**2, 'slategray', linestyle=':', alpha=0.4)
for mb in [-2, -1, 0, 1, 2]:
    shift = mb * 2 * np.pi / a
    energy = (k + shift)**2
    ax_left.plot(k*a/np.pi, energy, color='#1E88E5', linewidth=2.1, alpha=0.85)
ax_left.set_title('Periodicity = a', fontsize=14)
ax_left.set_xlabel('k (π/a)', fontsize=12)
ax_left.set_ylabel('E (ħ²/2m)', fontsize=12)
ax_left.set_xlim(-2.5, 2.5)
ax_left.set_ylim(0, 35)
ax_left.axvline(x=-1, color='#D81B60', linestyle='--', linewidth=2)
ax_left.axvline(x=1, color='#D81B60', linestyle='--', linewidth=2)
ax_left.text(0.5, 30, 'BZ edges at ±π/a', fontsize=10, color='#D81B60')
ax_left.grid(alpha=0.25)

# Right side - period 2a
ax_right.plot(k*a/np.pi, k**2, 'slategray', linestyle=':', alpha=0.4)

m_vals_2a = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
rainbow_colors = ['#FF5252', '#FF4081', '#7C4DFF', '#536DFE', '#00B8D4',
                  '#00BFA5', '#64DD17', '#FFD600', '#FF9100']

for j, m_val in enumerate(m_vals_2a):
    shift = m_val * np.pi / a
    energy = (k + shift)**2
    if m_val in [-4, -2, 0, 2, 4]:
        ax_right.plot(k*a/np.pi, energy, color=rainbow_colors[j],
                     linewidth=2.3, label=f'm={m_val}')
    else:
        ax_right.plot(k*a/np.pi, energy, color=rainbow_colors[j],
                     linewidth=1.8, alpha=0.75)

ax_right.set_title('Periodicity = 2a', fontsize=14)
ax_right.set_xlabel('k (π/a)', fontsize=12)
ax_right.set_xlim(-2.5, 2.5)
ax_right.set_ylim(0, 35)
ax_right.axvline(x=-0.5, color='#388E3C', linestyle='-', linewidth=2.2)
ax_right.axvline(x=0.5, color='#388E3C', linestyle='-', linewidth=2.2)
ax_right.text(0.55, 30, 'BZ edges at ±π/2a', fontsize=10, color='#388E3C')
ax_right.grid(alpha=0.25)
ax_right.legend(loc='upper left', fontsize=9)

fig2.suptitle('Effect of Changing Periodicity', fontsize=16)
plt.tight_layout()
plt.savefig('compare_periods.png', dpi=150)
plt.show()




# PLOT 3: Band folding details
fig3 = plt.figure(figsize=(13, 8))

k_small = np.linspace(-2.5*np.pi/a, 2.5*np.pi/a, 500)

# Zone backgrounds
plt.axvspan(-0.5, 0.5, alpha=0.12, color='honeydew')
plt.axvspan(-1, 1, alpha=0.08, color='lavender')

# Free electron reference
plt.plot(k_small*a/np.pi, k_small**2, 'gray', linestyle=(0, (5, 5)),
         alpha=0.5, linewidth=1.2, label='Free electron')

# Solid lines for period a
a_colors = ['#1565C0', '#C2185B', '#FF8F00']
for p, m_val in enumerate([-1, 0, 1]):
    shift = m_val * 2 * np.pi / a
    energy = (k_small + shift)**2
    plt.plot(k_small*a/np.pi, energy, color=a_colors[p],
             linewidth=3.5, label=f'a-period, m={m_val}')

# Dashed lines for period 2a
b_colors = ['#81D4FA', '#F48FB1', '#FFE082']
for q, m_val in enumerate([-2, 0, 2]):
    shift = m_val * np.pi / a
    energy = (k_small + shift)**2
    plt.plot(k_small*a/np.pi, energy, color=b_colors[q],
             linestyle='--', linewidth=3.5, label=f'2a-period, m={m_val}')

# Zone boundaries
plt.axvline(x=-1, color='indigo', linestyle=':', linewidth=2.5, alpha=0.6)
plt.axvline(x=1, color='indigo', linestyle=':', linewidth=2.5, alpha=0.6)
plt.axvline(x=-0.5, color='forestgreen', linestyle='-', linewidth=2.5, alpha=0.7)
plt.axvline(x=0.5, color='forestgreen', linestyle='-', linewidth=2.5, alpha=0.7)

plt.xlabel('Wavevector k (π/a)', fontsize=14)
plt.ylabel('Energy E (ħ²/2m)', fontsize=14)
plt.title('Band Folding Demonstration', fontsize=16)
plt.xlim(-2.2, 2.2)
plt.ylim(0, 28)
plt.grid(alpha=0.2)
plt.legend(loc='upper right', fontsize=11, framealpha=0.9)

# Info box
info_text = (
    'Physics Summary:\n'
    '• Period 2a → BZ halves\n'
    '• Bands fold into smaller zone\n'
    '• Same states, different labels\n'
    '• Gap positions change'
)
plt.text(1.3, 22, info_text, fontsize=12,
         bbox=dict(boxstyle='round', facecolor='linen', alpha=0.9))

plt.tight_layout()
plt.savefig('band_folding_demo.png', dpi=150)
plt.show()
