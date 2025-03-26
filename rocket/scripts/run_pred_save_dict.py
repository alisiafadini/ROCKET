import os, glob, subprocess
import pandas as pd
from tqdm import tqdm

device="cuda:0"
script_path = "/n/hekstra_lab/people/minhuan/projects/AF2_refine/screen_data/run_pretrained_openfold.py"
root = "/n/hekstra_lab/people/minhuan/projects/AF2_refine/screen_data/"
config_preset = "model_1_ptm"
max_recycling_iters = 20
jax_para_path = f"/n/hekstra_lab/people/minhuan/projects/AF2_refine/openfold_xtal/openfold/resources/params/params_{config_preset}.npz"
output_dir = os.path.join(root, f"AF2_trimmed_all_outputs_{max_recycling_iters}recycle_ptm")
command = [
    "python",
    f"{script_path}",
    f"{os.path.join(root, 'valid_seq_dir_trimmed_all')}",
    f"--use_precomputed_alignments={os.path.join(root, 'msa_dir_new_a3m_trimmed_all')}",
    f"--output_dir={output_dir}",
    f"--config_preset={config_preset}",
    f"--jax_param_path={jax_para_path}",
    f"--model_device={device}",
    f"--max_recycling_iters={max_recycling_iters}",
    f"--use_deepspeed_evoformer_attention",
    f"--skip_relaxation",
]
subprocess.run(command)