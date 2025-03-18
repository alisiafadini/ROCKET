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
        "SFcalculator-torch==0.2.1",
    ],
    entry_points={
        "console_scripts": [
            "rk.multiphaserefine=rocket.scripts.run_phase1andphase2:run_both_phases_all_datasets",
            "rk.phase1=rocket.scripts.run_phase1:run_phase1_all_datasets",
            "rk.mse=rocket.scripts.run_mse:run_mse_all_datasets",
            "rk.msacluster=rocket.scripts.run_msacluster:main",
            "rk.score=rocket.scripts.run_msascore:main",
            "rk.plddt=rocket.scripts.run_plddtoptimize:main"
        ]
    },
)
