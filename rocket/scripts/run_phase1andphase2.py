import subprocess


def run_refinement(
    root, device, flag, testset_value, bias_start=None, weight_start=None
):
    # Phase 1
    phase1_command = [
        "rk.refine",
        "-root",
        root,
        "-v",
        "3",
        "-it",
        "50",
        "-c",
        str(device),
        "--solvent",
        "--scale",
        "--sigmaArefine",
        "--lr_add",
        "0.5",
        "--lr_mul",
        "1.0",
        "--free_flag",
        flag,
        "--testset_value",
        str(testset_value),
        "-L2_w",
        "1e-11",
        "-b_thresh",
        "10",
        "-n",
        "Phase1",
    ]

    try:
        subprocess.run(phase1_command, check=True)
    except subprocess.CalledProcessError as e:
        print("Error occurred in Phase 1:", e)

    # Phase 2
    phase2_command = [
        "rk.refine",
        "-root",
        root,
        "-v",
        "3",
        "-it",
        "300",
        "-c",
        str(device),
        "--solvent",
        "--scale",
        "--sigmaArefine",
        "--lr_add",
        "0.001",
        "--lr_mul",
        "0.001",
        "--weight_decay",
        "0.0001",
        "--free_flag",
        flag,
        "--testset_value",
        str(testset_value),
        "-n",
        "Phase2",
    ]
    if bias_start:
        phase2_command.extend(["--bias_start", bias_start])
    if weight_start:
        phase2_command.extend(["--weights_start", weight_start])

    try:
        subprocess.run(phase2_command, check=True)
    except subprocess.CalledProcessError as e:
        print("Error occurred in Phase 2:", e)


def read_root_flag_from_file(file_path):
    with open(file_path, "r") as file:
        for line in file:
            parts = line.strip().split()  # Assuming space-separated values
            if len(parts) >= 2:
                root_name = parts[0]
                free_flag = parts[1]
                yield root_name, free_flag


# Usage example
path = "/net/cci/alisia/openfold_tests/run_openfold/test_cases"
root_name = "3hak"
device_number = 0  # for CUDA
flag_name = "R-free-flags"
testset_value = 0
res_min = 1.80
res_max = 16.30

# Iterate over the file and run the refinement for each pair
# for root_name, free_flag in read_root_flag_from_file(root_flag_file_path):
#    run_refinement(root_name, device_number, free_flag, testset_value, output_name, path)

output_name = "{r}_it50_v3_lr0.5+1.0_wdNone_batch1_subr1.0_solvTrue_scaleTrue_sigmaATrue_resol_{max}_{min}_rbrlbfgs_150.0_aliB_L21e-11+10.0_Phase1".format(
    r=root_name, max=res_max, min=res_min
)

bias_start_path = "{path}/{r}/outputs/{out}/best_feat_weights.pt".format(
    path=path, r=root_name, out=output_name
)
weight_start_path = "{path}/{r}/outputs/{out}/best_msa_bias.pt".format(
    path=path, r=root_name, out=output_name
)


run_refinement(
    root_name,
    device_number,
    flag_name,
    testset_value,
    bias_start_path,
    weight_start_path,
)
