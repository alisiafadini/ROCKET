# ROCKET Preprocessing Script

## Overview
This script performs the preprocessing of predicted protein structures for **ROCKET**. It runs **OpenFold inference**, processes structures using **Phenix**, and performs **Molecular Replacement or Cryo-EM Docking**.

## Input Parameters
| Argument | Description |
|----------|------------|
| `--file_id` | Identifier for input files. |
| `--resolution` | The best resolution for cryoEM map. Not used for x-ray case. |
| `--method` | Choose `"xray"` (calls Phaser) or `"cryoem"` (calls EMPlacement). |
| `--output_dir` | Directory to store results (default: `"preprocessing_output"`). |
| `--precomputed_alignment_dir` | Path to OpenFold precomputed alignments (default: `"alignments/"`). |
| `--jax_params_path` | Path to JAX parameter file (`"params_model_1_ptm.npz"`). Default `None`, will use system env var `$OPENFOLD_RESOURCES` |


The scripts expects input files organized as follows:

```
<working_directory>/
├── {file_id}_fasta/
│   └── {file_id}.fasta       # FASTA file containing the chain to refine
│                             # Header should be "> {file_id}"
│
├── {file_id}_data/
│   ├── *.mtz                 # For X-ray data
│   ├── *_half_map*.mrc       # For Cryo-EM data
│   └── <optional files>/     # e.g., predicted or docked models
│
├── alignments/               # (default: --precomputed_alignment_dir)
│   └── {file_id}
|       └──*.a3m / *.hhr          
```

### Additional Parameters for X-ray (`--method x-ray`)
| Argument | Description |
|----------|------------|
| `--xray_data_label` | Reflection data labels (e.g., `"FP,SIGFP"`). |

### Additional Parameters for Cryo-EM (`--method cryo-em`)
| Argument | Description |
|----------|------------|
| `--map1` | Path to **Half-map 1**. |
| `--map2` | Path to **Half-map 2**. |
| `--full_composition` | Full sequence composition file. |

### Optional Arguments
| Argument | Description |
|----------|------------|
| `--predocked_model` | Path to an already docked model (default: `None`). |
| `--fixed_model` | Optional fixed model contribution (default: `None`). |

## Expected Data Input

Expects to find `file_id_data` and `file_id_fasta` directories containing the experimental data (reflection file or half maps) and sequence file for chain to model respectively.

## Expected Outputs
After execution, results will be structured in the `--output_dir` directory:

```
output_dir/
|── {file_id}.fasta                 # FASTA file containing the chain to refine, copied from input
├── alignments/                     # MSA files for the input sequence, copied from input
|   └──*.a3m / *.hhr               
│── predictions/                    # OpenFold structure predictions and pkl files
│   └── xxx_processed_feats.pickle  # Processed feature dict with cluster profiles           
│── processed_predicted_files/      # Processed predictions from Phenix (including trimmed confidence loops)
│── docking_outputs/                # Cryo-EM docking results
│── phaser_files/                   # X-ray molecular replacement results
│── ROCKET_inputs/                  # Final outputs for ROCKET main trunk
│   ├── {file_id}-pred-aligned.pdb  # Aligned prediction with pseudo-Bs
│   ├── {file_id}-Edata.mtz         # Experimental data in LLG convention
```

## Usage Examples
### Running for X-ray Crystallography
```bash
python3 rocket_preprocessing.py \
    --file_id sample1 \
    --resolution 2.5 \
    --method xray \
    --xray_data_label "FP,SIGFP" \
    --output_dir results_xray
```

### Running for Cryo-EM
```bash
python3 rocket_preprocessing.py \
    --file_id sample2 \
    --resolution 4.0 \
    --method cryoem \
    --map1 path/to/halfmap1.mrc \
    --map2 path/to/halfmap2.mrc \
    --full_composition path/to/composition.fasta \
    --predocked_model path/to/thechain/torefine \
    --fixed_model /path/to/otherchains/indata \
    --output_dir results_cryoem
```


