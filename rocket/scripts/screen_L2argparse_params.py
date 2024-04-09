import subprocess

import itertools
from skopt import BayesSearchCV


# Arguments to screen
root = "3hak"
version = 3
iterations = 300
lr_add = [0.05]
lr_mul = [1.0]
# solvent = [True]
batches = [(1, 1.0)]
# schedule = [False]
L2_weight = [1e-11]
note = ["A", "B", "C"]
b_thresh =  [8, 10, 15, 25]
# ["3hak", "6lzm", "7dt2_short", "3pyfy_short", "4wa9_short"]

# Generate all possible combinations
#combinations = list(itertools.product(lr_add, lr_mul, L2_weight, note, b_thresh))

combinations = [ 
                ["3hak", "2", "0.01", "0.001", "0.0", "10", "0.1", ""],
                 ["3hak", "3", "0.01", "0.001", "0.0", "10", "0.1", ""],
                 ["3hak", "4", "0.01", "0.001", "0.0", "10", "0.1", "3hak-pred-aligned.pdb"],

                 ["3hak", "2", "0.05", "0.5", "1e-11", "10", "0.1", ""],
                 ["3hak", "3", "0.05", "0.5", "1e-11", "10", "0.1", ""],
                 ["3hak", "4", "0.05", "0.5", "1e-11", "10", "0.1", "3hak-pred-aligned.pdb"],

                 #["3pyy_short", "2", "0.05", "0.5", "0.0", "10", "0.01", ""],
                 #["3pyy_short", "3", "0.05", "0.5", "0.0", "10", "0.01", ""],
                 #["3pyy_short", "4", "0.05", "0.5", "0.0", "10", "0.01", "3pyy_short-pred-aligned.pdb"],

                 #["3pyy_short", "2", "0.05", "0.5", "1e-11", "10", "0.01", ""],
                 #["3pyy_short", "3", "0.05", "0.5", "1e-11", "10", "0.01", ""],
                 #["3pyy_short", "4", "0.05", "0.5", "1e-11", "10", "0.01", "3pyy_short-pred-aligned.pdb"],

]

for combo in combinations:
    command = [
        "python",
        "/net/cci/alisia/rocket/ROCKET/rocket/scripts/iterate_rbr_argparse_L2loss.py",
        f"--file_root={combo[0]}",
        f"--version={combo[1]}",
        f"--iterations={iterations}",
        f"--cuda=0",
        f"--solvent",
        f"--scale",
        f"-sub_r=0.7",
        f"--L2_weight={combo[4]}",
        f"--lr_add={combo[2]}",
        f"--lr_mul={combo[3]}",
        #f"--note={combo[3]}",
        f"-flag=FreeR_flag",
        f"--weight_decay={combo[6]}",
        f"-b_thresh={combo[5]}",
        f"-temp={combo[7]}",
        #f"--added_chain",
    ]
    subprocess.run(command)

"""
param_space = {
    "version": [2, 3],
    "lr_add": [1e-3, 1e-4, 1e-5],
    "lr_mul": [1e-1, 1e-2, 1e-3],
    "solvent": [True, False],
    "batches": [1],
    "subr": [1.0],
    "schedule": [True, False],
    "root": ["3hak"],
    "iterations": [1],
}


def objective_function(params):
    command = [
        "python",
        "/net/cci/alisia/rocket/ROCKET/rocket/scripts/iterate_rbr_argparse.py",
        f"--file_root={params['root']}",
        f"--version={params['version']}",
        f"--iterations={params['iterations']}",
        f"--cuda=2",
        f"--solvent" if params.get("solvent", False) else "",
        f"--scale" if params.get("scale", False) else "",
        f"--lr_add={params['lr_add']}",
        f"--lr_mul={params['lr_mul']}",
        f"--sub_ratio={params['subr']}",
        f"--batches={params['batches']}",
        # f"--added_chain",  # TODO fix based on root
    ]

    path = "/net/cci/alisia/openfold_tests/run_openfold/test_cases"
    root = params["root"]
    output_name = "{root}_it{it}_v{v}_lr{a}+{m}_batch{b}_subr{subr}_solv{solv}_scale{scale}_{align}{add}".format(
        root=params["root"],
        it=params["iterations"],
        v=params["version"],
        a=params["lr_add"],
        m=params["lr_mul"],
        b=params["batches"][0],
        subr=params["batches"][1],
        solv=params["solvent"],
        scale=params["scale"],
        align="B",
    )

    subprocess.run(command)

    LLGs = np.load(
        "{path}/{r}/outputs/{out}/LLG_it.npy".format(
            path=path,
            r=root,
            out=output_name,
        )
    )
    #print(LLGs)
    # Run the script and return the value to optimize
    return float(LLGs[-1])

    # Perform Bayesian optimization


opt = BayesSearchCV(
    objective_function,
    search_spaces=param_space,
    n_iter=1,  # Number of samples tested
    random_state=42,  # Set a random seed for reproducibility
)

opt.fit(None)
"""
