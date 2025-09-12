#!/usr/bin/env python3
"""
Patch script to add artificial viscosity initialization to solver.py
"""

def patch_solver():
    """Add artificial viscosity initialization to the 1D solver."""
    
    # Read the current solver file
    with open('phrike/solver.py', 'r') as f:
        lines = f.readlines()
    
    # Find the line number where we need to insert the artificial viscosity initialization
    # Look for the end of the adaptive time-stepping setup in the 1D solver
    insert_line = None
    for i, line in enumerate(lines):
        if 'self.scheme = adaptive_config.get("fallback_scheme", "rk4")' in line:
            # This should be in the 1D solver (around line 133)
            if i < 200:  # Make sure it's in the 1D solver section
                insert_line = i + 1
                break
    
    if insert_line is None:
        print("Could not find insertion point for artificial viscosity initialization")
        return False
    
    # Insert the artificial viscosity initialization code
    av_init_code = [
        '        \n',
        '        # Setup artificial viscosity if configured\n',
        '        self.artificial_viscosity = None\n',
        '        if artificial_viscosity_config and artificial_viscosity_config.get("enabled", False):\n',
        '            from .artificial_viscosity import create_artificial_viscosity\n',
        '            self.artificial_viscosity = create_artificial_viscosity(artificial_viscosity_config)\n',
        '\n'
    ]
    
    # Insert the code
    for j, code_line in enumerate(av_init_code):
        lines.insert(insert_line + j, code_line)
    
    # Write the modified file
    with open('phrike/solver.py', 'w') as f:
        f.writelines(lines)
    
    print(f"Added artificial viscosity initialization at line {insert_line}")
    return True

if __name__ == "__main__":
    patch_solver()
