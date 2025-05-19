# *R*efining *O*penfold predictions with *C*rystallographic/*C*ryo-EM Li*KE*lihood *T*argets (ROCKET)

[![Build](https://github.com/alisiafadini/ROCKET/actions/workflows/test.yml/badge.svg)](https://github.com/alisiafadini/ROCKET/actions/workflows/test.yml)
[![Ruff](https://github.com/alisiafadini/ROCKET/actions/workflows/lint.yml/badge.svg)](https://github.com/alisiafadini/ROCKET/actions/workflows/lint.yml)
[![GitHub License](https://img.shields.io/github/license/alisiafadini/ROCKET)](https://github.com/alisiafadini/ROCKET/blob/main/LICENSE)
[![BioRXiv](https://img.shields.io/badge/DOI-10.1101%2F2025.02.18.638828v2-purple?link=https%3A%2F%2Fwww.biorxiv.org%2Fcontent%2F10.1101%2F2025.02.18.638828v2)](https://www.biorxiv.org/content/10.1101/2025.02.18.638828v2)
[![Doc](https://img.shields.io/badge/Doc-GitBook-violet)](https://rocket-9.gitbook.io/rocket-docs)




This is the code repo for [AlphaFold as a Prior: Experimental Structure Determination Conditioned on a Pretrained Neural Network](https://www.biorxiv.org/content/10.1101/2025.02.18.638828v2)

You can find detailed documentation and walk-through tutorials at: https://rocket-9.gitbook.io/rocket-docs

## Installation

### 1. Install OpenFold

To ensure usability, we forked the OpenFold repo, and sorted a couple details in the installation guides. Here is what we advise ROCKET users to do:

**Note**: The ⁠openfold installation requires approximately 6 GB of free space to download weights. Please ensure you start in a directory with sufficient available space.

**Note**: To ensure a smooth installation and execution of ROCKET, install on a GPU machine that matches the hardware you’ll use in production. In other words, for HPC users, if you plan to run your code on a node with a particular GPU model, request the same GPU model when you install OpenFold. This is important because the installation process performs a hardware-specific compilation. We also recommend using GPUs with CUDA Compute Capability 8.0 or higher.

1. Clone our fork of the OpenFold repo, switch to the `pl_upgrades` branch to work with CUDA 12:

    ```
    git clone https://github.com/minhuanli/rocket_openfold.git
    cd rocket_openfold
    git checkout pl_upgrades
    ```

2. Create a conda/mamba env with the `environment.yml`
   
   
    Note: If you work with an HPC cluster with package management like `module`, purge all your modules before this step to avoid conflicts. 
    
    ```
    mamba env create -n <env_name_you_like> -f environment.yml
    mamba activate <env_name_you_like>
    ```
 
    The main change we made is moving the `flash-attn` package outside of the yml file, so you can install it manually afterwards. This is necessary because this OpenFold version relies on pytorch 2.1, which is incompatible with the latest flash-attn, so a simple `pip install flash-attn` would fail. Also using a `--no-build-isolation` flag allows using `ninja` for compilation, which is much faster.
 
   


3. Install compatible `flash-attn` (latest flash-attn with noted support for pytroch-2.1 + cuda-12.1)

    ```
    pip install flash-attn==2.2.2 --no-build-isolation
    ```

4. Run the setup script to install OpenFold, and configure kernels and folding resources
   
    ```
    ./scripts/install_third_party_dependencies.sh
    ```
 
    Add the following lines to `<path_to_your_conda_env>/etc/conda/activate.d/env_vars.sh`, create it if it doesn't exist
    
    ```
    #!/bin/sh
    
    export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
    ```
 
    This is so everytime you activate this env, the prepend will happen automatically.

5. Download AlphaFold2 weights, add **the resources path to system environment** (we need this for ROCKET)
   
    ```
    ./scripts/download_alphafold_params.sh ./openfold/resources
    ```
 
    Note: You can download OpenFold weights if you want to try

    Append the following line to `<path_to_your_conda_env>/etc/conda/activate.d/env_vars.sh`, you should have created it from the previous step

    ```
    export OPENFOLD_RESOURCES="<ABSOLUTE_PATH_TO_OPENFOLD_FOLDER>/openfold/resources"
    ```

    `<ABSOLUTE_PATH_TO_OPENFOLD_FOLDER>` should be the output of `pwd -P` you get from the OpenFold repo path.

    Deactivate and reactivate your python environment, you should be able to run and see the path:
    
    ```
    echo $OPENFOLD_RESOURCES 
    ```

6. Check your OpenFold build with unit tests:

    ```
    ./scripts/run_unit_tests.sh
    ```
 
    Ensure you see no errors:
    
    ```
    ...
    Time to load evoformer_attn op: 243.8257336616516 seconds
    ............s...s.sss.ss.....sssssssss.sss....ssssss..s.s.s.ss.s......s.s..ss...ss.s.s....s........
    ----------------------------------------------------------------------
    Ran 117 tests in 275.889s
 
    OK (skipped=41)
    ```   

### 2. Install Phenix (required from automatic preprocessing and post-refinement)

[Phenix](https://phenix-online.org/) is required for automatic data preprocessing and for post-refinement when polishing final model geometry. Follow the steps below to install it and **add the path to the system environment variables**:

1. Download the latest `nightly-build` Phenix python3 installer according to [https://phenix-online.org/download](https://phenix-online.org/download), note you have download the installer from the [show-all link](https://www.phenix-online.org/download/nightly_builds.cgi?show_all=1), with version newer than `2.0rc1-5647`

2. Run the installer

    ```
    bash phenix-installer-2.0rc1-5617-<platform>.sh
    ```

    You will be prompted to type your preferred path of installation, after specifying it, you will see:

    ```
    Phenix will now be installed into this location:
    <phenix_directory>/phenix-2.0rc1-5617
    ```

    Note: `<phenix_directory>` must be a absolute path. The installer will will make `<phenix_directory>/phenix-2.0rc1-5617` and install there.

3. Append the following line to `<path_to_your_conda_env>/etc/conda/activate.d/env_vars.sh`, you should have created it from the previous section

    ```
    export PHENIX_ROOT="<phenix_directory>/phenix-2.0rc1-5617"
    ```

    `<phenix_directory>` is where you install Phenix in the last step

    Deactivate and reactivate your python environment, you should be able to run and see the path:
    
    ```
    echo $PHENIX_ROOT 
    ``` 

### 3. Install ROCKET

Install ROCKET. First move to the parent folder, clone the ROCKET repo (so you don't mix the ROCKET repo with the OpenFold one), then install it with `pip`

```
git clone https://github.com/alisiafadini/ROCKET.git
cd ROCKET
pip install .
```

It will automatically install dependencies like `SFcalculator` and `reciprocalspaceship`.

Note: If you get errors about incompatibility of `prompt_toolkit`, ignore them.

For develop mode, run

```
pip install -e .
```

Run `rk.score --help` after installation, if you see a normal doc strings without errors, you are good to go!


### Citing

```
@article{fadini2025alphafold,
  title={AlphaFold as a Prior: Experimental Structure Determination Conditioned on a Pretrained Neural Network},
  author={Fadini, Alisia and Li, Minhuan and McCoy, Airlie J and Terwilliger, Thomas C and Read, Randy J and Hekstra, Doeke and AlQuraishi, Mohammed},
  journal={bioRxiv},
  year={2025}
}
```

   
