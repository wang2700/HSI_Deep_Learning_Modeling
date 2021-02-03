from setuptools import setup
from setuptools import find_packages

exclude_dirs = ("config",".vscode")

# for install, do: pip install -ve .

setup(
    name='hsidl',
    version="0.0.1",
    url="https://github.com/wang2700/HSI_Deep_Learning_Modeling",
    description="Hyperspectral Image Deep Learning Model for Nutrient Prediction",
    packages=find_packages(exclude=exclude_dirs),
)