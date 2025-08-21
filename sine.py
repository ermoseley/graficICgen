#!/usr/bin/env python3
"""
Simple script to plot a sine function from 0 to 2π.
This script displays the plot in a GUI window when run.

Usage:
    python3 sine.py
"""

import numpy as np
import matplotlib.pyplot as plt


def main():
    """Plot a sine function from 0 to 2π and display in GUI."""
    # Create x values from 0 to 2π
    x = np.linspace(0, 2 * np.pi, 100)
    
    # Calculate y = sin(x)
    y = np.sin(x)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', linewidth=2, label='sin(x)')
    
    # Add labels and title
    plt.xlabel('x (radians)')
    plt.ylabel('sin(x)')
    plt.title('Sine Function from 0 to 2π')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Set x-axis ticks to show π values
    plt.xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], 
               ['0', 'π/2', 'π', '3π/2', '2π'])
    
    # Set y-axis limits
    plt.ylim(-1.1, 1.1)
    
    # Display the plot in GUI
    plt.show()


if __name__ == "__main__":
    main()