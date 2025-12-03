import numpy as np
import matplotlib.pyplot as plt
import time

import autograd.numpy as npa
from autograd import grad, value_and_grad

import legume
from legume.minimize import Minimize
from legume.utils import grad_num
from matplotlib import gridspec as gs
GMAX = 3
K_INTERP = 5
# Define a PhC with a diatom shape in a square lattice.
lattice = legume.Lattice('square')
def nzi_phc(d, r1, r2):
    """
    d: slab thikcness
    r1, r2: radii of the shapes
    """
    phc = legume.PhotCryst(lattice)
    phc.add_layer(d=d, eps_b=9.46) # InP slab

    # We implement the NZI structure with two circles per unit cell
    circle1 = legume.Circle(eps=1, x_cent=0.5, y_cent=0.5, r=r1)
    circle2 = legume.Circle(eps=1, x_cent=0, y_cent=0, r=r2)
    phc.add_shape(circle1, layer=0)
    phc.add_shape(circle2, layer=0)

    return phc

phc = nzi_phc(0.25, 0.24, 0.21)    # parameters of the paper
gme = legume.GuidedModeExp(phc, gmax=GMAX)

# We can have a look at the structure as defined and obtained from an inverse FT

legume.viz.structure(phc, xz=True, yz=True, cladding=False, figsize=(4,10), cbar=False)
legume.viz.eps_ft(gme, cbar=False)
plt.show()

print("Running guided-mode expansion for NZI structure...")
path = lattice.bz_path(["X", "G", "M", "X"], ns=[K_INTERP]*3)
#options = {'gmode_inds': [0,3,4,7], 'numeig': 10, 'verbose': False}
options = {'gmode_inds': [0, 3, 4, 7, 8, 11], 'numeig': 10, 'verbose': False}

gme.run(kpoints=path['kpoints'], **options)
def plot_bands(gme):
    fig, ax = plt.subplots(1, figsize = (7, 5))
    legume.viz.bands(gme, Q=True, ax=ax)
    ax.set_xticks(path['indexes'])
    ax.set_xticklabels(path['labels'])
    ax.xaxis.grid('True')

    

plot_bands(gme)
plt.savefig('nzi_bands_no_opt.png', dpi=300)  
plt.show()
for i in range(10):
    fig = legume.viz.field(gme, "H",K_INTERP, mind=i, z=0.25/2, val="re")
    fig = plt.gcf()
    fig.set_size_inches(15,5)
    plt.savefig(f'nzi_mode_H_{i}.png', dpi=300)
    plt.show() 


## Now we fix r1 and sweep r2; we plot the bands at the Gamma point
r1_fixed = 0.24
r2_vals = np.linspace(0.20, 0.30, 25)
freqs_map = np.zeros((len(r2_vals), options['numeig']))
for i, r2 in enumerate(r2_vals):
    print(f"Computing bands for r2={r2:.3f}, progress: {i+1}/{len(r2_vals)}", end='\r')
    phc = nzi_phc(0.25, r1_fixed, r2)
    gme = legume.GuidedModeExp(phc, gmax=GMAX)
    gme.run(kpoints=np.array([[0], [0]]), **options)
    freqs_map[i, :] = gme.freqs[0, :]
plt.figure(figsize=(6,5))
for n in range(options['numeig']):
    plt.plot(r2_vals, freqs_map[:, n], label=f'Mode {n}')
plt.xlabel('r2 (a.u.)')
plt.ylabel('Frequency (a.u.)')
plt.title(f'Band Structure at Gamma Point (r1={r1_fixed})')
plt.legend()
plt.savefig('nzi_bands_r2_sweep.png', dpi=300)
plt.show()


## Test gradient 
# To compute gradients, we need to set the `legume` backend to 'autograd'
legume.set_backend('autograd')

# Objective function is the difference in frequency between modes 6 and 4
# Mode 2 is by symmetry degenerate with either 1 or 3
def of_nzi(params):
    r1 = params[0]
    r2 = params[1]

    phc = nzi_phc(0.35, r1, r2)
    gme = legume.GuidedModeExp(phc, gmax=5)
    gme.run(kpoints=np.array([[0], [0]]), **options)

    return gme.freqs[0, 5] - gme.freqs[0, 3]


pstart = np.array([0.239, 0.24])
obj_grad = value_and_grad(of_nzi)

# Compute the autograd gradients (NB: all at once!)
grad_a = obj_grad(pstart)[1]
print("Autograd gradient w.r.t. d, r0, rd:   ", grad_a)

# Compute a numerical gradient
grad_n = grad_num(of_nzi, pstart)

# # Now we optimize the parameters using L-BFGS
# print("Numerical gradient w.r.t. d, r0, rd:  ", grad_n)
# bounds=[(0.1, 0.3), (0.1, 0.3)]
# opt = Minimize(of_nzi)
# # Run an 'lbfgs' optimization
# (p_opt, ofs) = opt.lbfgs(pstart, Nepochs=10, bounds=bounds)
# # Print optimal parameters and visualize PhC bands
# print("Optimal parameters found are d = %1.2f, r0 = %1.2f" %(p_opt[0], p_opt[1]))
# phc = nzi_phc(0.25, p_opt[0], p_opt[1])
# gme = legume.GuidedModeExp(phc, gmax=5)
# gme.run(kpoints=path['kpoints'], **options)
# plot_bands(gme)
# plt.savefig('nzi_bands_opt.png', dpi=300)   
# plt.show()


# Now map the space in r1, r2 to find the detuning gap
r1_vals = np.linspace(0.21, 0.25, 10)
r2_vals = np.linspace(0.21, 0.25, 10)
gap_map = np.zeros((len(r1_vals), len(r2_vals)))

for i, r1 in enumerate(r1_vals):
    for j, r2 in enumerate(r2_vals):
        print(f"Computing gap for r1={r1:.3f}, r2={r2:.3f}, progress: {i*len(r2_vals)+j+1}/{len(r1_vals)*len(r2_vals)}", end='\r')
        phc = nzi_phc(0.25, r1, r2)
        gme = legume.GuidedModeExp(phc, gmax=GMAX)
        gme.run(kpoints=np.array([[0], [0]]), **options)
        gap_map[i, j] = gme.freqs[0, 5] - gme.freqs[0, 3]
plt.figure(figsize=(6,5))
plt.imshow(gap_map.T, extent=(r1_vals[0], r1_vals[-1], r2_vals[0], r2_vals[-1]), 
           origin='lower', aspect='auto', interpolation="spline16", cmap='viridis')
plt.colorbar(label='Detuning Gap (a.u.)')
plt.xlabel('r1 (a.u.)')
plt.ylabel('r2 (a.u.)')
plt.title('Detuning Gap Map for NZI Structure')
plt.savefig('nzi_gap_map.png', dpi=300)
plt.show()