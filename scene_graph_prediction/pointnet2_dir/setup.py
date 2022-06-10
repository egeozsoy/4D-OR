import os.path as osp

from setuptools import find_packages, setup

requirements = ["pytorch-lightning==1.6.0"]

exec(open(osp.join("pointnet2", "_version.py")).read())

setup(
    name="pointnet2",
    version=__version__,
    author="Erik Wijmans",
    packages=find_packages(),
    install_requires=requirements,
)
