import subprocess

import itertools
from skopt import BayesSearchCV


# Arguments to screen
root = ["3pyy_short", "4wa9_short", "7dt2_short"]
version = [2, 3]
iterations = 300
lr_add = [1e-3, 1e-4]
lr_mul = [1e-1, 1e-2, 1e-3]
# solvent = [True]
batches = [(1, 1.0)]
# schedule = [False]
# ["3hak", "6lzm", "7dt2_short", "3pyy_short", "4wa9_short"]

# Generate all possible combinations
combinations = list(itertools.product(root, version, lr_add, lr_mul, batches))

for combo in combinations:
    command = [
        "python",
        "/net/cci/alisia/rocket/ROCKET/rocket/scripts/iterate_rbr_argparse.py",
        f"--file_root={combo[0]}",
        f"--version={combo[1]}",
        f"--iterations={iterations}",
        f"--cuda=0",
        f"--solvent",
        f"--scale",
        f"--lr_add={combo[2]}",
        f"--lr_mul={combo[3]}",
        f"--sub_ratio={combo[4][1]}",
        f"--batches={combo[4][0]}",
        f"--added_chain",  # TODO fix based on root
    ]

    subprocess.run(command)
    exit()

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
