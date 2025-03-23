import argparse
import subprocess
import os
import shutil
import glob

### Phenix variables
phenix_directory = "/dev/shm/alisia/phenix-2.0rc1-5641/"
phenix_source = os.path.join(phenix_directory, "phenix_env.sh")
em_nodockedmodel_script = os.path.join(phenix_directory, "lib/python3.9/site-packages/New_Voyager/scripts/emplace_simple.py")
em_dockedmodel_script = os.path.join(phenix_directory, "lib/python3.9/site-packages/cctbx/maptbx/prepare_map_for_refinement.py")
xtal_edata_script = os.path.join(phenix_directory, "lib/python3.9/site-packages/phasertng/scripts/mtz_generator.py")

def run_command(command, env_source=None):
    """Runs a shell command with optional Phenix environment sourcing."""
    cmd_str = f"bash -c 'source {env_source} && {' '.join(command)}'" if env_source else " ".join(command)
    
    print(f"Executing: {cmd_str}")
    
    subprocess.run(cmd_str, shell=True, check=True, executable="/bin/bash")

def run_openfold(file_id, output_dir, precomputed_alignment_dir, mmcif_dir, jax_param_path):
    """Runs OpenFold inference using the specified parameters."""
    fasta_dir = f"{file_id}_fasta"
    predicted_model = os.path.join(output_dir, "predictions", f"{file_id}_model_1_ptm_unrelaxed.pdb")

    if os.path.exists(predicted_model):
        print(f"Skipping OpenFold: output {predicted_model} already exists.")
        return predicted_model

    openfold_cmd = [
        "python3", "run_pretrained_openfold.py",
        fasta_dir, mmcif_dir,
        "--output_dir", output_dir,
        "--config_preset", "model_1_ptm",
        "--model_device", "cuda:0",
        "--save_output",
        "--data_random_seed", "42",
        "--skip_relaxation",
        "--jax_param_path", jax_param_path,
        "--use_precomputed_alignments", precomputed_alignment_dir
    ]

    run_command(openfold_cmd)

    if not os.path.exists(predicted_model):
        raise FileNotFoundError(f"Expected output model {predicted_model} not found.")

    return predicted_model

def run_process_predicted_model(file_id, input_dir, predicted_model):
    """Processes the predicted model using Phenix."""
    print("Looking for", predicted_model)


    process_cmd = [
        "phenix.process_predicted_model",
        "output_files.mark_atoms_to_keep_with_occ_one=True",
        f"{predicted_model}",
        "minimum_domain_length=20",
        "b_value_field_is=plddt",
        "minimum_sequential_residues=10",
        f"pae_file={os.path.join(input_dir, f'{file_id}_pae.json')}",
        "pae_power=2",
        "pae_cutoff=4",
        "pae_graph_resolution=0.5"
    ]
    
    run_command(process_cmd, env_source=phenix_source)

def move_processed_predicted_files(output_dir):
    """Moves processed files into a 'processed_predicted_files' directory."""
    processed_dir = os.path.join(output_dir, "processed_predicted_files")
    os.makedirs(processed_dir, exist_ok=True)

    processed_files = glob.glob("*processed*") + glob.glob("*.seq")

    if not processed_files:
        print("No processed files found to move.")
        return

    for file_path in processed_files:
        if os.path.isfile(file_path): 
            shutil.move(file_path, os.path.join(processed_dir, os.path.basename(file_path)))

def dock_into_data(file_id, method, resolution, output_dir, predicted_model, predocked_model, map1, map2, fixed_model=None, fasta_composition=None):
    """Handles molecular docking for X-ray or Cryo-EM data."""
    docking_output_dir = os.path.join(output_dir, "docking_outputs")
    os.makedirs(docking_output_dir, exist_ok=True)

    if method == "x-ray":
        mtz_files = glob.glob(os.path.join(f"{file_id}_data", "*.mtz"))

        for mtz_file in mtz_files:
            if os.path.isfile(mtz_file):
                shutil.copy2(mtz_file, os.path.join(output_dir, "processed_predicted_files", os.path.basename(mtz_file)))
                edata_cmd = ["phenix.python",
                    xtal_edata_script,
                    f"-i",
                    f"{mtz_file}"
                ]
                run_command(edata_cmd, env_source=phenix_source)

        mr_cmd = [
            "phasertng.picard",
            f"directory={os.path.join(output_dir, 'processed_predicted_files')}",
            f"database={os.path.join(output_dir, 'phaser_files')}"
        ]
        run_command(mr_cmd, env_source=phenix_source)

    elif method == "cryo-em":
        docking_script = em_dockedmodel_script if predocked_model else em_nodockedmodel_script
        docking_cmd = ["phenix.python", docking_script]

        if predocked_model:
            docking_cmd += [map1, map2, predocked_model, resolution]
            if fixed_model:
                docking_cmd.append(f"--fixed_model={fixed_model}")
        else:
            docking_cmd += [
                f"--d_min={resolution}",
                f"--output_folder={docking_output_dir}",
                f"--model_file={predicted_model}",
                f"--map1={map1}",
                f"--map2={map2}",
                f"--sequence_composition={fasta_composition}",
                "--level=logfile"
            ]
            if fixed_model:
                docking_cmd.append(f"--fixed_model={fixed_model}")

        run_command(docking_cmd, env_source=phenix_source)

        if predocked_model:
            for file in ["weighted_map_data.mtz", "likelihood_weighted.map"]:
                src_path = os.path.join(".", file)
                dest_path = os.path.join(docking_output_dir, file)
                if os.path.exists(src_path):
                    shutil.move(src_path, dest_path)

            # Move the predocked model
            model_filename = os.path.basename(predocked_model)
            model_dest_path = os.path.join(docking_output_dir, model_filename)
            if os.path.exists(predocked_model):
                shutil.copy(predocked_model, model_dest_path)

def prepare_rk_inputs(file_id, output_dir, method):
    """Creates ROCKET_inputs directory and moves necessary files."""
    rocket_dir = os.path.join(output_dir, "ROCKET_inputs")
    os.makedirs(rocket_dir, exist_ok=True)

    if method == "x-ray":
        best_pdb_src = os.path.join(output_dir, "phaser_files", "best.1.coordinates.pdb")
        mtz_files = glob.glob(f"./*feff/*.data.mtz")
    elif method == "cryo-em":
        best_pdb_src = next(iter(glob.glob(os.path.join(output_dir, "docking_outputs", "*.pdb"))), None)
        mtz_files = glob.glob(f"{output_dir}/docking_outputs/weighted_map_data.mtz")
    else:
        raise ValueError("Invalid method. Choose either 'x-ray' or 'cryo-em'.")

    best_pdb_dst = os.path.join(rocket_dir, f"{file_id}-pred-aligned.pdb")
    if best_pdb_src and os.path.exists(best_pdb_src):
        shutil.copy2(best_pdb_src, best_pdb_dst)

    for mtz_src in mtz_files:
        mtz_dst = os.path.join(rocket_dir, f"{file_id}-Edata.mtz")
        shutil.copy2(mtz_src, mtz_dst)

def parse_args():
    """Parses and validates command-line arguments."""
    parser = argparse.ArgumentParser(description="Run OpenFold inference and dock into data")
    
    parser.add_argument("--file_id", required=True)
    parser.add_argument("--resolution", required=True)
    parser.add_argument("--method", choices=["x-ray", "cryo-em"], required=True)
    parser.add_argument("--output_dir", default="preprocessing_output")
    parser.add_argument("--precomputed_alignment_dir", default="alignments/")
    parser.add_argument("--mmcif_dir", default="/net/cci-gpu-00/raid1/scratch1/alisia/programs/openfold/openfold_v2.0.0/data/pdb_mmcif/mmcif_files/")
    parser.add_argument("--jax_params_path", default="/net/cci-gpu-00/raid1/scratch1/alisia/programs/openfold/openfold_v2.0.0/openfold/resources/params/params_model_1_ptm.npz")
    parser.add_argument("--predocked_model", default=None)
    parser.add_argument("--fixed_model", default=None)
    parser.add_argument("--xray_data_label", default=None)
    parser.add_argument("--map1", default=None)
    parser.add_argument("--map2", default=None)
    parser.add_argument("--full_composition", default=None)

    args = parser.parse_args()

    if args.method == "x-ray" and not args.xray_data_label:
        parser.error("--xray_data_label is required when --method is x-ray.")

    if args.method == "cryo-em":
        missing = [arg for arg in ["map1", "map2"] if getattr(args, arg) is None]
        if missing:
            parser.error(f"The following arguments are required for 'cryo-em' method: {', '.join(missing)}")

        # Require full_composition only if predocked_model is not provided
        if not args.predocked_model and args.full_composition is None:
            parser.error("--full_composition is required for cryo-em when --predocked_model is not provided.")

    return args



def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    predicted_model = run_openfold(args.file_id, args.output_dir, args.precomputed_alignment_dir, args.mmcif_dir, args.jax_params_path)
    run_process_predicted_model(args.file_id, args.output_dir, predicted_model)
    move_processed_predicted_files(args.output_dir)

    dock_into_data(args.file_id, args.method, args.resolution, args.output_dir, predicted_model, args.predocked_model, args.map1, args.map2, args.fixed_model, args.full_composition)
    prepare_rk_inputs(args.file_id, args.output_dir, args.method)

if __name__ == "__main__":
    main()
