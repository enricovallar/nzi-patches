import numpy as np
import matplotlib.pyplot as plt
import time

import autograd.numpy as npa
from autograd import grad, value_and_grad

import legume
from legume.minimize import Minimize
from legume.utils import grad_num

# Define a PhC with a diatom shape in a square lattice.
lattice = legume.Lattice('square')
def nzi_phc(d, r1, r2):
    """
    d: slab thikcness
    r1, r2: radii of the shapes
    """
    phc = legume.PhotCryst(lattice)
    phc.add_layer(d=d, eps_b=9.46) # InP slab

    # We implement the daisy as a FourierShape class
    circle1 = legume.Circle(eps=1, x_cent=0.5, y_cent=0, r=r1)
    circle2 = legume.Circle(eps=1, x_cent=0, y_cent=0.5, r=r2)
    phc.add_shape(circle1, layer=0)
    phc.add_shape(circle2, layer=0)

    return phc

phc = nzi_phc(0.25, 0.239, 0.24)    # parameters of the paper
gme = legume.GuidedModeExp(phc, gmax=10)

# We can have a look at the structure as defined and obtained from an inverse FT
legume.viz.structure(phc, figsize=2., cbar=False, Nx=200, Ny=300)
legume.viz.eps_ft(gme, figsize=2., cbar=False, Nx=200, Ny=300)
path = lattice.bz_path(['M', 'G', 'X'], ns=[40, 40])
options = {'gmode_inds': [0,3], 'numeig': 10, 'verbose': False}

gme.run(kpoints=path['kpoints'], **options)

def plot_bands(gme):
    fig, ax = plt.subplots(1, figsize = (7, 5))
    legume.viz.bands(gme, Q=True, ax=ax)
    ax.set_xticks(path['indexes'])
    ax.set_xticklabels(path['labels'])
    ax.xaxis.grid('True')
    ax.set_ylim(0.6,0.8)
    

plot_bands(gme)
plt.savefig('nzi_bands_no_opt.png', dpi=300)   
plt.show()


## Test gradient 
# To compute gradients, we need to set the `legume` backend to 'autograd'
legume.set_backend('autograd')

# Objective function is the difference in frequency between modes 6 and 4
# Mode 2 is by symmetry degenerate with either 1 or 3
def of_nzi(params):
    r1 = params[0]
    r2 = params[1]

    phc = nzi_phc(0.25, r1, r2)
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

# Now we optimize the parameters using L-BFGS
print("Numerical gradient w.r.t. d, r0, rd:  ", grad_n)
bounds=[(0.1, 0.3), (0.1, 0.3)]
opt = Minimize(of_nzi)
# Run an 'lbfgs' optimization
(p_opt, ofs) = opt.lbfgs(pstart, Nepochs=10, bounds=bounds)
# Print optimal parameters and visualize PhC bands
print("Optimal parameters found are d = %1.2f, r0 = %1.2f" %(p_opt[0], p_opt[1]))
phc = nzi_phc(0.25, p_opt[0], p_opt[1])
gme = legume.GuidedModeExp(phc, gmax=5)
gme.run(kpoints=path['kpoints'], **options)
plot_bands(gme)
plt.savefig('nzi_bands_opt.png', dpi=300)   
plt.show()