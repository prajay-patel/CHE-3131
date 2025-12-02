#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider
from scipy.optimize import fsolve
import warnings

# Physical constants
eV2J=1.60e-19 #J
h=6.62606957e-34 #J*s
hbar= h/(2*np.pi)
me=9.11e-31 #kg

# Define fitting functions
def f(Ei,V0):
  '''Ei is in eV'''
  '''V0 is in eV'''
  E=Ei*eV2J
  V=V0*eV2J
  # Suppress RuntimeWarning for invalid value encountered in sqrt
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    a=2*np.sqrt(E*(V-E))
  b=2*E-V
  return a/b

def g(Ei,L):
  '''Ei is in eV'''
  '''L is in m'''
  E=Ei*eV2J
  a=2*me*E*(L**2)
  b= hbar**2
  # Suppress RuntimeWarning for invalid value encountered in sqrt
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    c=np.sqrt(a/b)
  return np.tan(c)

# Modified j to accept V0 and L as arguments
def j(x, V0_local, L_local):
  return f(x,V0_local)-g(x,L_local)

def E_pib(n,L=10):
  return (n**2)*(h**2)/(8*me*(L**2)*eV2J)

def intersections(X2,Y2,Z2): #Keep as is, what does it do?
  roots_idx = np.argwhere(np.diff(np.sign(Y2-Z2)) != 0).flatten()
  Xint = X2[roots_idx]
  return Xint[1::2]
  

def finite_well(V0,LA):
  L = LA*1e-10 #m
  X = np.linspace(0.005,V0, 10000)

  with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    Y = f(X,V0)
    Z = g(X,L)

  # Filter the data to remove values approaching the asymptote
  valid_indices = ~np.isnan(Y) & ~np.isnan(Z) & np.isfinite(Y) & np.isfinite(Z) & (np.abs(Z) < V0)
  X2=X[valid_indices]
  Y2=Y[valid_indices]
  Z2=Z[valid_indices]

  # Solve for the roots
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    Xint = fsolve(lambda x_val: j(x_val, V0, L), intersections(X2,Y2,Z2))

  # Return the roots as the final answer
  return Xint

def plot_finite_well(V0, LA):
    '''Plot the finite well with energy levels'''
    L = LA * 1e-10

    # Calculate energy levels
    Xint = finite_well(V0, LA)

    # Create figure
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    ax.axis('off')

    # Plot the potential well
    ax.plot([0, 0], [0, V0], 'k-', lw=4)
    ax.plot([L, L], [0, V0], 'k-', lw=4)
    ax.plot([0, L], [0, 0], 'k-', lw=4)
    ax.plot([-L/2, 0], [V0, V0], 'k-', lw=4)
    ax.plot([L, 3/2*L], [V0, V0], 'k-', lw=4)

    # Label positions
    ax.text(0, -V0*0.1, '0', horizontalalignment='center', fontsize=20)
    ax.text(L, -V0*0.1, 'L', horizontalalignment='center', fontsize=20)

    # Label regions
    ax.text(-L/4, V0*0.85, 'I', horizontalalignment='center', fontsize=20)
    ax.text(L/2, V0*0.85, 'II', horizontalalignment='center', fontsize=20)
    ax.text(5/4*L, V0*0.85, 'III', horizontalalignment='center', fontsize=20)
    ax.text(-0.6*L, V0, '$V_0$', horizontalalignment='center',
            verticalalignment='center', fontsize=20)

    # Create color map
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i) for i in np.linspace(0, 0.85, max(len(Xint), 1))]

    # Plot energy levels only
    for i, E in enumerate(Xint):
        # Plot energy level line
        ax.plot([0, L], [E, E], lw=2.5, color=colors[i], alpha=0.7)
        ax.text(-L/8, E, f'$E_{{{i+1}}}$', horizontalalignment='right',
                fontsize=16, verticalalignment='center', color=colors[i],
                weight='bold')

    # Set plot limits
    ax.set_ylim(-V0*0.15, V0*1.2)
    ax.set_xlim(-0.7*15e-10, 1.6*15e-10)

    # Display energy values
    energy_text = "Energy Levels (eV):\n" + "-"*20 + "\n"
    for i, E in enumerate(Xint):
        energy_text += f"E_{i+1} = {E:.3f} eV\n"

    if len(Xint) == 0:
        energy_text = "No bound states\nfor these parameters"

    ax.text(1.35*L, V0*0.6, energy_text, fontsize=13,
            verticalalignment='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    plt.show()
    fig.savefig('FiniteWell_Interactive.png',dpi=300,bbox_inches='tight')

# Create interactive widget
interact(plot_finite_well,
         V0=FloatSlider(min=8.0, max=30.0, step=0.01, value=10.0, description='V₀ (eV)', readout_format='.2f'),
         LA=FloatSlider(min=1.0, max=20.0, step=0.00001, value=5.0, description='L (Å)', readout_format='.5f'))

# Display the interactive plot
#display(interactive_plot)