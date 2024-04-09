# Import necessary PyMOL modules
from pymol import cmd

# Define the directory path and file pattern
directory_path = "3hak_it150_v3_lr0.5+10.0_batch1_subr1.0_solvTrue_scaleTrue_rbrlbfgs_150.0_aliB_L21e-12+15.0__A/"
file_pattern = "_postRBR.pdb"

# Function to load and process PDB structures
def process_pdb_structures(start, end, step):
    for i in range(start, end, step):
        pdb_file = f"{i}{file_pattern}"

        # Load PDB structure
        cmd.load(directory_path + pdb_file, f"structure_{i}")

        # Set the desired view
        cmd.set_view (
                      (\
              0.725313783,    0.465831846,   -0.506674290,\
              0.107098803,   -0.803474903,   -0.585425079,\
              -0.679851174,    0.370396137,   -0.632723212,\
              0.002600513,    0.002271399, -155.451629639,\
              8.955833435,   17.182674408,   18.048002243,\
            -783.587280273, 1094.575317383,  -20.000000000 ))

        # Display as cartoon
        cmd.show("cartoon")
        cmd.cartoon("loop", f"structure_{i} and chain A")

        # Color each residue by Bfactor using the specified spectrum
        cmd.spectrum("b", "blue_red", f"structure_{i} and polymer", minimum=5, maximum=100)

    # Load additional PDB file "3hak_noalts.pdb" and color it orange
    cmd.load("3hak_noalts.pdb", "exp_structure")
    cmd.color("iron", "exp_structure")
    cmd.cartoon("loop", "exp_structure and chain A")

    # Save the view as a PDF
    cmd.ray()
    cmd.png(directory_path+"structure_ensembles.png", width=800, height=620, dpi=2000)  # Change the file format if needed

# Set the range of structures to process
start_structure = 0
end_structure = 150
structure_step = 20

cmd.set("ray_shadows", 0)

# Call the function to process PDB structures
process_pdb_structures(start_structure, end_structure, structure_step)



