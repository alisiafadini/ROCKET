import subprocess
import itertools
import git


def get_latest_commit_hash(folder):
    repo = git.Repo(folder)
    return repo.head.commit.hexsha[:7]


# Arguments to screen
date = "Feb182024"
free_column = "FreeR_flags"

# rocket_root = "/n/hekstra_lab/people/minhuan/projects/AF2_refine/ROCKET/"
# latest_commit = get_latest_commit_hash(rocket_root)

path = "/net/cci/alisia/openfold_tests/run_openfold/test_cases"
template_pdb = "3hak-pred-aligned.pdb"
root = ["3hak"]
version = ["4"]
lr_add = [0.1, 0.5, 1.0]
lr_mul = [0.0]
weight_decay = [0.1]
rbr_opt = [("lbfgs", 150.0)]
l2_weight = [1e-11]
b_thresh = [10.0]
# resol_min = [1.7, 2.2, 2.5, 3.0, 4.0]
note = ["Feb28"]
# Generate all possible combinations
combinations = list(
    itertools.product(
        root, lr_add, lr_mul, rbr_opt, weight_decay, l2_weight, b_thresh, version, note
    )
)
# With Solvent
for combo in combinations:
    command = [
        "rk.refine",
        f"--path={path}",
        f"--file_root={combo[0]}",
        f"--template_pdb={template_pdb}",
        f"--version={combo[7]}",
        f"--iterations=300",
        f"--cuda=1",
        f"--solvent",
        f"--scale",
        f"--lr_add={combo[1]}",
        f"--lr_mul={combo[2]}",
        f"--weight_decay={combo[4]}",
        f"--sub_ratio=1.0",
        f"--batches=1",
        f"--rbr_opt={combo[3][0]}",
        f"--rbr_lbfgs_lr={combo[3][1]}",
        f"--L2_weight={combo[5]}",
        f"--b_thresh={combo[6]}",
        f"--resol_min=0.0",
        f"--free_flag={free_column}",
        # f"--testset_value=1",
        f"--note={combo[8]}",
        # f"--added_chain",
    ]
    subprocess.run(command)
