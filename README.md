# *R*efining *O*penfold predictions with *C*rystallographic Li*KE*lihood *T*argets (ROCKET)

This is the code repo for [AlphaFold as a Prior: Experimental Structure Determination Conditioned on a Pretrained Neural Network](https://www.biorxiv.org/content/10.1101/2025.02.18.638828v2)


## Installation

### 1. Install OpenFold

To ensure the usability, we forked and fix the openfold repo, and polish the installation guides. Here is what ROCKET users are advised to do:

1. Clone our fork of openfold repo, switch to the `pl_upgrades` branch to work with CUDA 12:

    ```
    git clone git@github.com:minhuanli/rocket_openfold.git
    cd rocket_openfold
    git checkout pl_upgrades
    ```

2. Create a conda/mamba env with the `environment.yml`
   
   
    Note: If you work with HPC cluster with package management like `module`, purge all your modules before this step to avoid conflicts. 
    
    ```
    mamba env create -n <env_name_you_like> -f environment.yml
    mamba activate <env_name_you_like>
    ```
 
    The main change we made is moving those `flash-attn` package outside of the yml file, so you can install manually afterwards. This is necessary because this openfold version relies on pytorch 2.1, which is incompatible with the latest flash-attn, so a simple `pip install flash-attn` would fail. Also using a `--no-build-isolation` flag allows to use `ninja` for compilation, which is way much faster.
 
   


3. Install compatible `flash-attn` (latest flash-attn with noted support for pytroch-2.1 + cuda-12.1)

    ```
    pip install flash-attn==2.2.2 --no-build-isolation
    ```

4. Run the setup script to install openfold, configure kernels and folding resources
   
    ```
    ./scripts/install_third_party_dependencies.sh
    ```
 
    Add the following lines to `<path_to_your_conda_env>/etc/conda/activate.d/env_vars.sh`, create it if it doesn't exist
    
    ```
    #!/bin/sh
    
    export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
    ```
 
    So everytime you activate this env, the prepend will happen automatically.

5. Download AlphaFold2 weights, add **the resources path to system environment** (we need this for ROCKET)
   
    ```
    ./scripts/download_alphafold_params.sh ./openfold/resources
    ```
 
    Note: You can download openfold weights if you want to try

    Append the following line to `<path_to_your_conda_env>/etc/conda/activate.d/env_vars.sh`, you should have created it from the previous step

    ```
    export OPENFOLD_RESOURCES="<ABSOLUTE_PATH_TO_OPENFOLD_FOLDER>/openfold/resources"
    ```

    `<ABSOLUTE_PATH_TO_OPENFOLD_FOLDER>` should be the output of `pwd -P` you get from the openfold repo path.

    Deactivate and reactivate your python environment, you should be able to run and see the path:
    
    ```
    echo $OPENFOLD_RESOURCES 
    ```

6. Check your openfold build with unit tests:

    ```
    ./scripts/run_unit_tests.sh
    ```
 
    Should see no errors:
    
    ```
    ...
    Time to load evoformer_attn op: 243.8257336616516 seconds
    ............s...s.sss.ss.....sssssssss.sss....ssssss..s.s.s.ss.s......s.s..ss...ss.s.s....s........
    ----------------------------------------------------------------------
    Ran 117 tests in 275.889s
 
    OK (skipped=41)
    ```   

### 2. Install ROCKET

Install ROCKET. First move to the parent folder, clone the ROCKET repo (so you don't mix the ROCKET repo with the openfold one), then install it with `pip`

```
git clone git@github.com:alisiafadini/ROCKET.git
cd ROCKET
pip install .
```

It will automatically install dependencies like `SFcalculator` and `reciprocalspaceship`.

Note: If you get errors about incompatibility of `prompt_toolkit`, ignore it.

For develop mode, run

```
pip install -e .
```

Run `rk.score --help` after installation, if you saw normal doc strings without error, you are good to go.


### Citing

```
@article{fadini2025alphafold,
  title={AlphaFold as a Prior: Experimental Structure Determination Conditioned on a Pretrained Neural Network},
  author={Fadini, Alisia and Li, Minhuan and McCoy, Airlie J and Terwilliger, Thomas C and Read, Randy J and Hekstra, Doeke and AlQuraishi, Mohammed},
  journal={bioRxiv},
  year={2025}
}
```

   



