#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider, Checkbox, FloatSlider, VBox, HBox, Label
from IPython.display import display

# Original data
wavelengths = np.array([519.3, 352.3, 310.4, 300.1, 293.4, 274.5, 258.2, 251, 
                        245.2, 229.5, 222.1, 218.3, 215.6, 208.7, 202.3, 200.1,
                        198, 196.6, 195.4, 194.1])

f_values = np.array([2.250663123, 0.087828481, 0.131133197, 0.178865429,
                     0.136105546, 0.001247733, 0.128665748, 0.386021518,
                     0.038209115, 0.13633804, 0.065530745, 0.020615738,
                     0.345155453, 0.386199338, 0.043982073, 0.000438533,
                     0.113700968, 0.004359182, 0.081426429, 0.258937165])

def broaden_peak_gaussian(X, h, f, broadening_factor=2.5):
    """
    Broaden a spectral peak using Gaussian function
    X: wavelength array
    h: peak center wavelength
    f: oscillator strength
    broadening_factor: divisor for sigma (higher = narrower peaks)
                      σ = 0.2 eV = 1/6199.2 nm⁻¹ / broadening_factor
    """
    # Base sigma: 1/6199.2 nm^-1 (corresponds to 0.2 eV)
    sigma = (1/6199.2) / broadening_factor
    
    f2 = 1e7 * sigma
    A = 1.3062974e8
    e1 = 1/h - 1/X
    return A * (f / f2) * np.exp(-(e1 / sigma)**2)

def plot_spectral_peaks(num_peaks=20, broadening_factor=2.5, show_lines=True, 
                       show_individual=True, show_combined=True):
    """
    Plot spectral peaks with interactive controls
    """
    # Create wavelength array for plotting
    X = np.linspace(180, 700, 1000)

    # Set up the plot
    fig, ax = plt.subplots(figsize=(7, 6))

    # Colors for individual peaks
    colors = plt.cm.tab20(np.linspace(0, 1, 20))

    # Calculate and plot individual peaks
    combined_spectrum = np.zeros_like(X)

    for i in range(num_peaks):
        h = wavelengths[i]
        f = f_values[i]

        # Calculate broadened peak
        peak = broaden_peak_gaussian(X, h, f, broadening_factor) / 2e5
        combined_spectrum += peak

        # Plot vertical line at peak position
        if show_lines:
            # Plot a vertical line from 0 to half the peak's maximum height
            ax.plot([h, h], [0, peak.max() / 2], color=colors[i], linestyle='--', 
                   alpha=0.75, linewidth=1)

        # Plot individual Gaussian peak
        if show_individual:
            ax.plot(X, peak, color=colors[i], linewidth=1.5, alpha=0.75,
                   label=f'{h} nm' if i < 10 else '')

    # Plot combined spectrum
    if show_combined:
        ax.plot(X, combined_spectrum, 'k-', linewidth=3, label='Combined Spectrum')

    # Calculate actual sigma for display
    sigma_nm = (1/6199.2) / broadening_factor
    sigma_cm = 1e7 * sigma_nm
    
    # Formatting
    ax.set_xlabel('Wavelength (nm)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Intensity (ε)', fontsize=14, fontweight='bold')
    ax.set_title(f'Spectral Peaks Visualization (Showing {num_peaks} of 20 peaks)\nBroadening Factor: {broadening_factor:.2f} (σ = {sigma_cm:.1f} cm⁻¹)',
                fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(180, 700)
    ax.set_yticklabels([])

    plt.tight_layout()
    plt.show()

    # Print peak information
    print(f"\n{'='*60}")
    print(f"Showing {num_peaks} of 20 peaks")
    print(f"Broadening Factor: {broadening_factor:.2f}")
    print(f"σ = (1/6199.2 nm⁻¹) / {broadening_factor:.2f} = {sigma_nm:.6e} nm⁻¹")
    print(f"σ = {sigma_cm:.1f} cm⁻¹ (= {sigma_cm/1e7:.2e} × 10⁷ cm⁻¹)")
    print(f"{'='*60}")
    #print(f"{'Peak':<6} {'Wavelength (nm)':<18} {'f value':<15}")
    #print(f"{'-'*60}")
    #for i in range(num_peaks):
    #    print(f"{i+1:<6} {wavelengths[i]:<18.1f} {f_values[i]:<15.6f}")
    #print(f"{'='*60}\n")

# Create interactive widget
interact(plot_spectral_peaks,
         num_peaks=IntSlider(min=1, max=20, step=1, value=1,
                            description='# Peaks:',
                            style={'description_width': 'initial'},
                            continuous_update=False),
         broadening_factor=FloatSlider(min=0.5, max=10.0, step=0.5, value=2.5,
                                       description='Broadening:',
                                       style={'description_width': 'initial'},
                                       continuous_update=False,
                                       readout_format='.1f'),
         show_lines=Checkbox(value=True, description='Show Vertical Lines'),
         show_individual=Checkbox(value=True, description='Show Individual Peaks'),
         show_combined=Checkbox(value=True, description='Show Combined Spectrum'));