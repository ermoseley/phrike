#!/usr/bin/env python3
"""Plot initial and final conditions for 1D RTI test with no gravity."""

import numpy as np
import matplotlib.pyplot as plt
from phrike.io import load_checkpoint
from phrike.equations import EulerEquations1D

def plot_rti_1d_no_gravity_comparison():
    """Plot initial vs final conditions for 1D RTI test with no gravity."""
    
    # Load initial and final snapshots
    initial_path = "/Users/moseley/phrike/outputs/rti_1d_no_gravity/snapshot_t0.000000.npz"
    final_path = "/Users/moseley/phrike/outputs/rti_1d_no_gravity/snapshot_t2.000000.npz"
    
    try:
        # Load snapshots
        initial_data = load_checkpoint(initial_path)
        final_data = load_checkpoint(final_path)
        
        # Extract data
        x_initial = initial_data['x']
        U_initial = initial_data['U']
        x_final = final_data['x']
        U_final = final_data['U']
        
        # Create equations object to get primitive variables
        equations = EulerEquations1D(gamma=1.4)
        
        # Get primitive variables
        rho_initial, u_initial, p_initial, _ = equations.primitive(U_initial)
        rho_final, u_final, p_final, _ = equations.primitive(U_final)
        
        # Convert to numpy if needed (in case they're torch tensors)
        if hasattr(x_initial, 'cpu'):
            x_initial = x_initial.cpu().numpy()
            rho_initial = rho_initial.cpu().numpy()
            p_initial = p_initial.cpu().numpy()
            u_initial = u_initial.cpu().numpy()
            
        if hasattr(x_final, 'cpu'):
            x_final = x_final.cpu().numpy()
            rho_final = rho_final.cpu().numpy()
            p_final = p_final.cpu().numpy()
            u_final = u_final.cpu().numpy()
        
        # Check for NaN values
        if np.any(np.isnan(rho_initial)) or np.any(np.isnan(rho_final)):
            print("Warning: NaN values detected!")
            print(f"Initial density NaN count: {np.sum(np.isnan(rho_initial))}")
            print(f"Final density NaN count: {np.sum(np.isnan(rho_final))}")
        else:
            print("No NaN values detected - simulation completed successfully!")
        
        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), constrained_layout=True)
        
        # Plot density
        ax1.plot(x_initial, rho_initial, 'b--', linewidth=2, label='Initial (t=0.000)', alpha=0.8)
        ax1.plot(x_final, rho_final, 'b-', linewidth=2, label='Final (t=2.000)', alpha=0.8)
        ax1.set_xlabel('x')
        ax1.set_ylabel('Density')
        ax1.set_title('Density Profile: Initial vs Final (No Gravity)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot velocity
        ax2.plot(x_initial, u_initial, 'r--', linewidth=2, label='Initial (t=0.000)', alpha=0.8)
        ax2.plot(x_final, u_final, 'r-', linewidth=2, label='Final (t=2.000)', alpha=0.8)
        ax2.set_xlabel('x')
        ax2.set_ylabel('Velocity')
        ax2.set_title('Velocity Profile: Initial vs Final (No Gravity)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Plot pressure
        ax3.plot(x_initial, p_initial, 'g--', linewidth=2, label='Initial (t=0.000)', alpha=0.8)
        ax3.plot(x_final, p_final, 'g-', linewidth=2, label='Final (t=2.000)', alpha=0.8)
        ax3.set_xlabel('x')
        ax3.set_ylabel('Pressure')
        ax3.set_title('Pressure Profile: Initial vs Final (No Gravity)')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Add some statistics
        if not np.any(np.isnan(rho_initial)) and not np.any(np.isnan(rho_final)):
            rho_change = np.max(np.abs(rho_final - rho_initial))
            p_change = np.max(np.abs(p_final - p_initial))
            u_change = np.max(np.abs(u_final - u_initial))
            
            ax1.text(0.02, 0.98, f'Max density change: {rho_change:.2e}', 
                     transform=ax1.transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax2.text(0.02, 0.98, f'Max velocity change: {u_change:.2e}', 
                     transform=ax2.transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax3.text(0.02, 0.98, f'Max pressure change: {p_change:.2e}', 
                     transform=ax3.transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.suptitle('1D RTI Test (No Gravity): Initial vs Final Conditions', fontsize=14)
        
        # Save plot
        output_path = "/Users/moseley/phrike/outputs/rti_1d_no_gravity/rti_1d_no_gravity_comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
        
        # Show plot
        plt.show()
        
    except Exception as e:
        print(f"Error creating comparison plot: {e}")
        print("This might be due to NaN values in the simulation data.")

if __name__ == "__main__":
    plot_rti_1d_no_gravity_comparison()
