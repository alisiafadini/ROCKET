import argparse
import subprocess
import os
import shutil
import glob
import reciprocalspaceship as rs
from loguru import logger

from ..refinement_config import gen_config_phase1, gen_config_phase2
from ..utils import plddt2pseudoB
from SFC_Torch import PDBParser

### Phenix variables
# phenix_directory = "/dev/shm/alisia/phenix-2.0rc1-5641/"
phenix_directory = os.environ["PHENIX_ROOT"]
phenix_source = os.path.join(phenix_directory, "phenix_env.sh")
em_nodockedmodel_script = os.path.join(
    phenix_directory,
    "lib/python3.9/site-packages/New_Voyager/scripts/emplace_simple.py",
)
em_dockedmodel_script = os.path.join(
    phenix_directory,
    "lib/python3.9/site-packages/cctbx/maptbx/prepare_map_for_refinement.py",
)
xtal_edata_script = os.path.join(
    phenix_directory, "lib/python3.9/site-packages/phasertng/scripts/mtz_generator.py"
)


def internal_mtz_labels(mtz_path, label_string):
    """Extracts and renames columns from MTZ file based on input label string.

    Args:
        mtz_path (str): Path to input MTZ file.
        label_string (str): Comma-separated string of two labels (e.g., "F,SIGF").

    Returns:
        rs.DataSet: Modified dataset with "FP" and "SIGFP" columns added.
    """
    labels = [label.strip() for label in label_string.split(",")]
    if len(labels) != 2:
        raise ValueError(
            "xray_data_label must contain exactly two labels separated by a comma (e.g., 'F,SIGF')."
        )

    ds = rs.read_mtz(mtz_path)
    if not all(label in ds.columns for label in labels):
        raise ValueError(f"Labels {labels} not found in MTZ file {mtz_path}.")

    ds["FP"] = ds[labels[0]]
    ds["SIGFP"] = ds[labels[1]]

    return ds


def run_command(command, env_source=None):
    """Runs a shell command with optional Phenix environment sourcing."""
    cmd_str = (
        f"bash -c 'source {env_source} && {' '.join(command)}'"
        if env_source
        else " ".join(command)
    )

    logger.info(f"Executing: {cmd_str}")

    subprocess.run(cmd_str, shell=True, check=True, executable="/bin/bash")


def run_openfold(
    file_id,
    output_dir,
    precomputed_alignment_dir,
    jax_param_path,
    max_recycling_iters,
    use_deepspeed_evoformer_attention,
):
    """Runs OpenFold inference using the specified parameters."""
    fasta_dir = f"{file_id}_fasta"
    predicted_model = os.path.join(
        output_dir, "predictions", f"{file_id}_model_1_ptm_unrelaxed.pdb"
    )

    if os.path.exists(predicted_model):
        logger.info(f"Skipping OpenFold: output {predicted_model} already exists.")
        return predicted_model

    openfold_cmd = [
        "rk.predict",
        fasta_dir,
        "--output_dir",
        f"{output_dir}",
        "--config_preset",
        "model_1_ptm",
        "--model_device",
        "cuda:0",
        "--save_output",
        "--data_random_seed",
        "42",
        "--skip_relaxation",
        "--max_recycling_iters",
        f"{max_recycling_iters}",
        "--use_precomputed_alignments",
        f"{precomputed_alignment_dir}",
    ]
    if use_deepspeed_evoformer_attention:
        openfold_cmd.extend(["--use_deepspeed_evoformer_attention"])

    if jax_param_path:
        openfold_cmd.extend(["--jax_params_path", jax_param_path])

    run_command(openfold_cmd)

    if not os.path.exists(predicted_model):
        raise FileNotFoundError(f"Expected output model {predicted_model} not found.")

    return predicted_model


def generate_seg_id_file(file_id, output_dir):
    """Generates seg_id.txt using chain changes and >20-residue continuous stretches. Skips first seg_id, outputs None if only one domain."""
    seg_id_path = os.path.join(output_dir, "ROCKET_inputs", "seg_id.txt")
    aligned_pdb_path = os.path.join(output_dir, "ROCKET_inputs", f"{file_id}-MRed.pdb")

    if not os.path.exists(aligned_pdb_path):
        raise FileNotFoundError(f"Aligned PDB file not found at {aligned_pdb_path}")

    # Collect residues per chain in order of appearance
    chain_residues = {}
    chain_order = []
    with open(aligned_pdb_path, "r") as f:
        for line in f:
            if line.startswith("ATOM"):
                try:
                    chain_id = line[21].strip()
                    res_num = int(line[22:26].strip())
                    if chain_id not in chain_residues:
                        chain_residues[chain_id] = set()
                        chain_order.append(chain_id)
                    chain_residues[chain_id].add(res_num)
                except ValueError:
                    continue

    domain_ranges = []
    seg_start_residues = []
    previous_chain = None

    for chain_id in chain_order:
        if chain_id == previous_chain:
            continue  # Only one domain per unique chain

        residues = sorted(chain_residues[chain_id])
        if not residues:
            continue

        # Find first continuous stretch >20
        current_stretch = [residues[0]]
        for i in range(1, len(residues)):
            if residues[i] == residues[i - 1] + 1:
                current_stretch.append(residues[i])
            else:
                if len(current_stretch) > 20:
                    domain_ranges.append((current_stretch[0], current_stretch[-1]))
                    seg_start_residues.append(current_stretch[0])
                    break
                current_stretch = [residues[i]]

        # Handle final stretch
        if len(current_stretch) > 20 and (
            not domain_ranges
            or domain_ranges[-1] != (current_stretch[0], current_stretch[-1])
        ):
            domain_ranges.append((current_stretch[0], current_stretch[-1]))
            seg_start_residues.append(current_stretch[0])

        previous_chain = chain_id

    # Write seg_id.txt
    with open(seg_id_path, "w") as out_f:
        for i, (start, end) in enumerate(domain_ranges, 1):
            out_f.write(f"domain{i}: {start}-{end}\n")

        if len(seg_start_residues) > 1:
            seg_ids = ",".join(str(r) for r in seg_start_residues[1:])  # Skip first
            out_f.write(f'seg_id: "{seg_ids}"\n')
            logger.info(f"Segment ID file written to {seg_id_path}")
            return seg_start_residues[1:]
        else:
            out_f.write("seg_id: None\n")
            logger.info("No segment, only one domain found.")
            return None


def run_process_predicted_model(file_id, input_dir, predicted_model):
    """Processes the predicted model using Phenix."""
    logger.info("Looking for", predicted_model)

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
        "pae_graph_resolution=0.5",
    ]

    run_command(process_cmd, env_source=phenix_source)


def move_processed_predicted_files(output_dir):
    """Moves processed files into a 'processed_predicted_files' directory."""
    processed_dir = os.path.join(output_dir, "processed_predicted_files")
    os.makedirs(processed_dir, exist_ok=True)

    processed_files = glob.glob("*processed*") + glob.glob("*.seq")

    if not processed_files:
        logger.info("No processed files found to move.")
        return

    for file_path in processed_files:
        if os.path.isfile(file_path):
            shutil.move(
                file_path, os.path.join(processed_dir, os.path.basename(file_path))
            )


def dock_into_data(
    file_id,
    method,
    resolution,
    output_dir,
    predicted_model,
    predocked_model,
    map1,
    map2,
    fixed_model=None,
    fasta_composition=None,
):
    """Handles molecular docking for Xray or CryoEM data."""
    docking_output_dir = os.path.join(output_dir, "docking_outputs")
    os.makedirs(docking_output_dir, exist_ok=True)

    if method == "xray":
        mtz_files = glob.glob(os.path.join(f"{file_id}_data", "*.mtz"))

        for mtz_file in mtz_files:
            if os.path.isfile(mtz_file):
                shutil.copy2(
                    mtz_file,
                    os.path.join(
                        output_dir,
                        "processed_predicted_files",
                        os.path.basename(mtz_file),
                    ),
                )

                # Always run Edata generation
                edata_cmd = ["phenix.python", xtal_edata_script, "-i", mtz_file]
                run_command(edata_cmd, env_source=phenix_source)

        # If predocked_model is provided, skip MR and copy the model directly
        if predocked_model:
            print("Predocked model provided for Xray: skipping MR step.")
            rocket_dir = os.path.join(output_dir, "ROCKET_inputs")
            os.makedirs(rocket_dir, exist_ok=True)

            aligned_pdb_path = os.path.join(rocket_dir, f"{file_id}-MRed.pdb")
            shutil.copy2(predocked_model, aligned_pdb_path)
        else:
            # Proceed with MR step
            mr_cmd = [
                "phasertng.picard",
                f"directory={os.path.join(output_dir, 'processed_predicted_files')}",
                f"database={os.path.join(output_dir, 'phaser_files')}",
            ]
            run_command(mr_cmd, env_source=phenix_source)

    elif method == "cryoem":
        docking_script = (
            em_dockedmodel_script if predocked_model else em_nodockedmodel_script
        )
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
                "--level=logfile",
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

    if method == "xray":
        feff_dirs = glob.glob("./*feff")
        if len(feff_dirs) == 0:
            raise FileNotFoundError(
                "No '*feff' directory found in the current directory."
            )
        if len(feff_dirs) > 1:
            raise RuntimeError(
                f"More than one '*feff' directory found: {feff_dirs}. You may have run preprocessing twice. Please clean up ambiguous directories."
            )

        feff_dir = feff_dirs[0]
        best_pdb_src = os.path.join(
            output_dir, "phaser_files", "best.1.coordinates.pdb"
        )
        mtz_files = glob.glob(os.path.join(feff_dir, "*.data.mtz"))
    elif method == "cryoem":
        best_pdb_src = next(
            iter(glob.glob(os.path.join(output_dir, "docking_outputs", "*.pdb"))), None
        )
        mtz_files = glob.glob(f"{output_dir}/docking_outputs/weighted_map_data.mtz")
    else:
        raise ValueError("Invalid method. Choose either 'xray' or 'cryoem'.")

    best_pdb_dst = os.path.join(rocket_dir, f"{file_id}-MRed.pdb")
    if best_pdb_src and os.path.exists(best_pdb_src):
        shutil.copy2(best_pdb_src, best_pdb_dst)

    for mtz_src in mtz_files:
        mtz_dst = os.path.join(rocket_dir, f"{file_id}-Edata.mtz")
        shutil.copy2(mtz_src, mtz_dst)


def prepare_pred_aligned(output_dir, file_id):
    mr_model_path = os.path.join(output_dir, "ROCKET_inputs", f"{file_id}-MRed.pdb")
    assert os.path.exists(mr_model_path), f"MR model not found: {mr_model_path}"
    pred_model_path = os.path.join(
        output_dir, "predictions", f"{file_id}_model_1_ptm_unrelaxed.pdb"
    )
    assert os.path.exists(
        pred_model_path
    ), f"Predicted model not found: {pred_model_path}"
    superpose_command = [
        "phenix.superpose_pdbs",
        f"{mr_model_path}",
        f"{pred_model_path}",
        f"output.file_name={os.path.join(output_dir, 'ROCKET_inputs', f'{file_id}-pred-aligned_unprocessed.pdb')}",
    ]
    run_command(superpose_command, env_source=phenix_source)
    aligned_model_path = os.path.join(
        output_dir, "ROCKET_inputs", f"{file_id}-pred-aligned_unprocessed.pdb"
    )
    assert os.path.exists(
        aligned_model_path
    ), f"Failed to superpose models: {aligned_model_path}"

    mr_model = PDBParser(mr_model_path)
    align_model = PDBParser(aligned_model_path)
    align_model.set_spacegroup(mr_model.spacegroup)
    align_model.set_unitcell(mr_model.cell)
    align_model.set_biso(plddt2pseudoB(align_model.atom_b_iso))
    align_model.savePDB(
        os.path.join(output_dir, "ROCKET_inputs", f"{file_id}-pred-aligned.pdb")
    )


def symlink_input_files(file_id, output_dir, precomputed_alignment_dir):
    """Symlinks the sequence FASTA and alignment directory to the output folder."""
    fasta_src = os.path.join(f"{file_id}_fasta", f"{file_id}.fasta")
    fasta_dst = os.path.join(output_dir, f"{file_id}.fasta")
    if os.path.exists(fasta_src):
        if not os.path.exists(fasta_dst):
            os.symlink(os.path.abspath(fasta_src), fasta_dst)
    else:
        raise FileNotFoundError(f"FASTA file not found: {fasta_src}")

    alignments_dst = os.path.join(output_dir, "alignments")
    if not os.path.exists(alignments_dst):
        os.symlink(
            os.path.join(os.path.abspath(precomputed_alignment_dir), file_id),
            alignments_dst,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run OpenFold inference and dock into data"
    )

    parser.add_argument("--file_id", required=True)
    parser.add_argument("--method", choices=["xray", "cryoem"], required=True)
    parser.add_argument("--resolution", default=None)
    parser.add_argument("--output_dir", default="preprocessing_output")
    parser.add_argument("--precomputed_alignment_dir", default="alignments/")
    parser.add_argument("--max_recycling_iters", type=int, default=4)
    parser.add_argument(
        "--use_deepspeed_evoformer_attention",
        action="store_true",
        default=False,
        help="Whether to use the DeepSpeed evoformer attention layer. Must have deepspeed installed in the environment.",
    )
    parser.add_argument("--jax_params_path", default=None)
    parser.add_argument("--predocked_model", default=None)
    parser.add_argument("--fixed_model", default=None)
    parser.add_argument("--xray_data_label", default=None)
    parser.add_argument("--map1", default=None)
    parser.add_argument("--map2", default=None)
    parser.add_argument("--full_composition", default=None)

    args = parser.parse_args()

    if args.method == "xray" and not args.xray_data_label:
        parser.error("--xray_data_label is required when --method is xray.")

    if args.method == "cryoem":
        missing = [
            arg for arg in ["map1", "map2", "resolution"] if getattr(args, arg) is None
        ]
        if missing:
            parser.error(
                f"The following arguments are required for 'cryoem' method: {', '.join(missing)}"
            )

        # Require full_composition only if predocked_model is not provided
        if not args.predocked_model and args.full_composition is None:
            parser.error(
                "--full_composition is required for cryoem when --predocked_model is not provided."
            )

    return args


def cli_runpreprocess():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    symlink_input_files(args.file_id, args.output_dir, args.precomputed_alignment_dir)

    predicted_model = run_openfold(
        args.file_id,
        args.output_dir,
        args.precomputed_alignment_dir,
        args.jax_params_path,
        args.max_recycling_iters,
        args.use_deepspeed_evoformer_attention,
    )
    run_process_predicted_model(args.file_id, args.output_dir, predicted_model)
    move_processed_predicted_files(args.output_dir)

    dock_into_data(
        args.file_id,
        args.method,
        args.resolution,
        args.output_dir,
        predicted_model,
        args.predocked_model,
        args.map1,
        args.map2,
        args.fixed_model,
        args.full_composition,
    )
    prepare_rk_inputs(args.file_id, args.output_dir, args.method)
    prepare_pred_aligned(args.output_dir, args.file_id)
    seg_id = generate_seg_id_file(args.file_id, args.output_dir)

    # Internal ROCKET labels
    if args.method == "xray":
        edata_path = os.path.join(
            args.output_dir, "ROCKET_inputs", f"{args.file_id}-Edata.mtz"
        )
        logger.info(
            f"Adding internal MTZ labels from {args.xray_data_label} to {edata_path}"
        )
        ds = internal_mtz_labels(edata_path, args.xray_data_label)
        try:
            ds.write_mtz(edata_path)
        except Exception as e:
            logger.info("MTZ write failed:", e)

    # Generate ROCKET configuration yaml files
    phase1_config = gen_config_phase1(
        datamode=args.method,
        file_id=args.file_id,
        working_dir=os.path.abspath(args.output_dir),
    )
    phase1_config.algorithm.init_recycling = args.max_recycling_iters
    if seg_id:
        phase1_config.algorithm.domain_segs = seg_id
    phase1_config.to_yaml_file(
        os.path.join(args.output_dir, "ROCKET_config_phase1.yaml")
    )
    phase2_config = gen_config_phase2(phase1_config)
    phase2_config.to_yaml_file(
        os.path.join(args.output_dir, "ROCKET_config_phase2.yaml")
    )

    # Move *feff folder(s) to output_dir (if only one exists)
    feff_dirs = glob.glob("./*feff")
    if len(feff_dirs) == 1:
        feff_dir = feff_dirs[0]
        dest = os.path.join(args.output_dir, os.path.basename(feff_dir))
        # Remove target if it exists to avoid shutil.move error
        if os.path.exists(dest):
            shutil.rmtree(dest)
        shutil.move(feff_dir, dest)
        logger.info(f"Moved {feff_dir} to {dest}")

    elif len(feff_dirs) > 1:
        raise RuntimeError(
            f"More than one '*feff' directory found: {feff_dirs}. You may have run preprocessing twice. Please clean up ambiguous directories."
        )


if __name__ == "__main__":
    cli_runpreprocess()
