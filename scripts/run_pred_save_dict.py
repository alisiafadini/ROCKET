import os, glob, subprocess
import pandas as pd
from tqdm import tqdm

device="cuda:0"
script_path = "/n/hekstra_lab/people/minhuan/projects/AF2_refine/openfold/run_pretrained_openfold.py"
root = "/n/hekstra_lab/people/minhuan/projects/AF2_refine/screen_data/"
valid_seq_df = pd.read_csv(os.path.join(root, "valid_seq.csv"))
config_preset = "model_1"
jax_para_path = f"/n/hekstra_lab/people/minhuan/projects/AF2_refine/openfold_xtal/openfold/resources/params/params_{config_preset}.npz"
output_dir = os.path.join(root, "AF2_outputs")
command = [
    "python",
    f"{script_path}",
    f"{os.path.join(root, 'valid_seq_dir')}",
    f"--use_precomputed_alignments={os.path.join(root, 'msa_dir_new_a3m')}",
    f"--output_dir={output_dir}",
    f"--config_preset={config_preset}",
    f"--jax_param_path={jax_para_path}",
    f"--model_device={device}",
    f"--skip_relaxation",
]
subprocess.run(command)