# Evaluating the performance of kcat prediction tools and their impact on enzyme-constrained genome-scale metabolic models

This is a Master Thesis project conducted as part of the Master's in [Systems Biology & Bioinformatics](https://www.maastrichtuniversity.nl/education/master/programmes/systems-biology-and-bioinformatics) at Maastricht University. The project is carried out at the Reasearch Group for [Biochemical Network Analysis](https://chemnet.univie.ac.at/) in the Department of Analytical Chemistry at the University of Vienna.

## Setup

This project uses **Git submodules** to include external model repositories under the `models/` directory.  
After cloning the repository, **you must initialize and update all submodules** before running any code.

### Submodule Setup
Clone with submodules:
```bash
git clone --recurse-submodules git@github.com:JohannUM/kcat-prediction-benchmarking.git
```
Or set up submodules after cloning:
```bash
git submodule update --init --recursive
```

### Notebook Setup
To run the `notebooks` the kcatbench module has to be installed in the environment for all imports to be resolved. In the root directory of this project run:

```bash
pip install -e .
```

## Project Information
### Current Models
| Model  | Location        | Source Repository                                                                    |
| ------ | --------------- | ------------------------------------------------------------------------------------ |
| DLKcat | `models/DLKcat` | [https://github.com/SysBioChalmers/DLKcat](https://github.com/SysBioChalmers/DLKcat) |
| MMKcat | `models/MMKcat` | [https://github.com/ProEcho1/MMKcat](https://github.com/ProEcho1/MMKcat) |
