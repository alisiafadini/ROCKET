from setuptools import setup, find_packages

# Get version number
def getVersionNumber():
    with open("rocket/VERSION", "r") as vfile:
        version = vfile.read().strip()
    return version

__version__ = getVersionNumber()

setup(name="rocket",
    version=__version__,
    description="Refining Openfold predictions with Crystallographic Likelihood Targets", 
    url="https://github.com/alisiafadini/ROCKET",
    packages=find_packages(),
    include_package_data=True,
)