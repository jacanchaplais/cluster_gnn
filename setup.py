from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='A prototype project for getting into ML and particle physics.',
    author='Jacan Chaplais',
    license='MIT',
    install_requires= [
        "vaex >= 3.0.0",
        "h5py >= 2.9",
        "hdf5"
    ]
)
