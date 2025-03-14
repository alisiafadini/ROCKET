# *R*efining *O*penfold predictions with *C*rystallographic Li*KE*lihood *T*argets (ROCKET)


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

5. Download AlphaFold2 weights
   
    ```
    ./scripts/download_alphafold_params.sh ./openfold/resources
    ```
 
    Note: You can download openfold weights if you want to try

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
   




   



