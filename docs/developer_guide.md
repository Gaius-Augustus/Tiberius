### Guide for developers

#### Setup

To install Tiberius as a local package in editable mode you need to install it in your python environment with the following recommended commands (skip steps 1-3 if you already have a working environment with all dependencies installed):
```bash

# step 1
conda create -n tiberius python=3.10
conda activate tiberius

# step 2
# install all dependencies
# see REAMDME.md for more details

# step 3 get the source code
git clone https://github.com/Gaius-Augustus/Tiberius

# step 4 install as an editable package
cd Tiberius
pip install -e .
```

You can now import tiberius in any python script or jupyter notebook:
```python
import tiberius
```


#### Running tests

You can run unit tests from the tiberius root folder like this:
```bash
python -m unittest test.unit_tests
```

Troubleshooting:
The error
```
importlib.metadata.PackageNotFoundError: No package metadata was found for tiberius
```
can be resolved by installing the package in editable mode from the root folder of the repository:
```bash
pip install -e .
```