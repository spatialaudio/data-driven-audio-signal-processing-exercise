# data-driven-audio-signal-processing-exercise

**Data Driven Audio Signal Processing - A Tutorial with Computational Examples**

This tutorial accompanies the lecture [Data Driven Audio Signal Processing](https://github.com/spatialaudio/data-driven-audio-signal-processing-lecture). The lecture and the tutorial are designed for International Standard Classification of Education (ISCED) level 7 (Master).

Jupyter notebooks can be accessed via the services

- **dynamic** version using **mybinder**: TBA
- **static** version using **nbviewer**: TBA
- **sources** (tex, ipynb) at: https://github.com/spatialaudio/data-driven-audio-signal-processing-exercise

## Anaconda Environment for Local Usage

The [Anaconda distribution](https://www.anaconda.com/distribution/) is a convenient solution to install a required environment, i.e. to have access to the Jupyter Notebook renderer with a Python interpreter on a personal computer. It is very likely that a very recent installation of Anaconda already delivers all required packages just using the `base` environment. It is however good practice to create a dedicated environment for each projects So, for this tutorial we might use a `myddasp` (or whatever name works for us) environment.

- `conda create -n myddasp python=3.9 pip numpy scipy tensorflow scikit-learn pandas matplotlib notebook jupyterlab ipykernel nb_conda jupyter_nbextensions_configurator jupyter_contrib_nbextensions autopep8`
`pip install soundfile`

- under `conda 4.10.3` and `conda-build 3.21.4` the current environment to develop and test the notebooks locally by github user *fs446* is
`conda create -n myddasp python=3.9.7 pip=21.2.4 numpy=1.21.2 scipy=1.7.1 tensorflow=2.4.3 scikit-learn=1.0 pandas=1.3.3 matplotlib=3.4.3 notebook=6.4.4 jupyterlab=3.1.18 ipykernel=6.4.1 nb_conda=2.2.1 jupyter_nbextensions_configurator=0.4.1 jupyter_contrib_nbextensions=0.5.1 autopep8=1.5.7`
and using soundfile version 0.10.3

- activate this environment with `conda activate myddasp`

- Jupyter notebook renderer needs to know our dedicated environment:
`python -m ipykernel install --user --name myddasp --display-name "myddasp"`

- get into the folder where the exercises are located, e.g. `cd my_ddasp_folder`

- we might want to archive the actually installed package versions by: `python -m pip list > detailed_packages_list_pip.txt` and `conda env export --no-builds > detailed_packages_list_conda.txt`

- start either a Jupyter notebook or Jupyter lab working environment via a local server instance by either `jupyter notebook` or `jupyter lab`

- start the notebook `index.ipynb`as the landing page for the tutorial

- make sure that the notebook we want to work with uses our kernel `myddasp`

## License

- Creative Commons Attribution 4.0 International License (CC BY 4.0) for text/graphics
- MIT License for software

## Versions / Tags / Branches

- TBA for winter term 2021/2022

## Referencing

Please cite this open educational resource (OER) project as
*Frank Schultz, Data Driven Audio Signal Processing - A Tutorial Featuring Computational Examples, University of Rostock* ideally with relevant ``file(s), github URL, commit number and/or version tag, year``.

## Authorship

University of Rostock:

- Frank Schultz
- Sascha Spors
