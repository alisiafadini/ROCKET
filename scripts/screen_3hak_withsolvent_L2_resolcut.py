import subprocess
import itertools
import git

def get_latest_commit_hash(folder):
    repo = git.Repo(folder)
    return repo.head.commit.hexsha[:7]

# Arguments to screen
date='Feb162024'
free_column="FreeR_flag"

rocket_root="/n/hekstra_lab/people/minhuan/projects/AF2_refine/ROCKET/"
latest_commit=get_latest_commit_hash(rocket_root)

path = "/n/hekstra_lab/people/minhuan/projects/AF2_refine/ROCKET/dev/test_cases"
template_pdb="3hak-pred-aligned.pdb"
root = ["3hak"]
lr_add = [0.05]
lr_mul = [1.0]
weight_decay = [None]
rbr_opt = [("lbfgs", 150.0)]
l2_weight = [1e-11]
b_thresh = [10.0]
resol_min = [1.7, 2.2, 2.5, 3.0, 4.0]
note = ["A", "B", "C"]
# Generate all possible combinations
combinations = list(itertools.product(root, lr_add, lr_mul, rbr_opt, weight_decay, l2_weight, b_thresh, resol_min, note))
# With Solvent
for combo in combinations:
    command = [
        "rk.refine",
        f"--path={path}",
        f"--file_root={combo[0]}",
        f"--template_pdb={template_pdb}",
        f"--version=3",
        f"--iterations=100",
        f"--cuda=0",
        f"--solvent",
        f"--scale",
        f"--lr_add={combo[1]}",
        f"--lr_mul={combo[2]}",
        # f"--weight_decay={combo[4]}",
        f"--sub_ratio=0.7",
        f"--batches=1",
        f"--rbr_opt={combo[3][0]}",
        f"--rbr_lbfgs_lr={combo[3][1]}",
        f"--L2_weight={combo[5]}",
        f"--b_thresh={combo[6]}",
        f"--resol_min={combo[7]}",
        f"--free_flag={free_column}",
        f"--note={date}_{latest_commit}_{combo[8]}"
    ]
    subprocess.run(command)
    # exit()