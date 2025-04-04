from setuptools import setup, find_packages


# Get version number
def getVersionNumber():
    with open("rocket/VERSION", "r") as vfile:
        version = vfile.read().strip()
    return version


__version__ = getVersionNumber()

setup(
    name="rocket",
    version=__version__,
    description="Refining Openfold predictions with Crystallographic Likelihood Targets",
    url="https://github.com/alisiafadini/ROCKET",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "scikit-bio",
        "loguru",
        "SFcalculator-torch>=0.2.2",
        "matplotlib",
        "polyleven",
        "scikit-learn",
        "seaborn",
    ],
    entry_points={
        "console_scripts": [
            "rk.predict=rocket.scripts.run_pretrained_openfold:cli_runopenfold",
            "rk.preprocess=rocket.scripts.run_preprocess:cli_runpreprocess",
            "rk.refine=rocket.scripts.run_refine:cli_runrefine",
            "rk.config=rocket.scripts.run_config:cli_runconfig",
            "rk.mse=rocket.scripts.run_mse:run_mse_all_datasets",
            "rk.msacluster=rocket.scripts.run_msacluster:main",
            "rk.score=rocket.scripts.run_msascore:main",
            "rk.plddt=rocket.scripts.run_plddtoptimize:main"
        ]
    },
)
