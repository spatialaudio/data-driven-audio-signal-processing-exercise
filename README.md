# data-driven-audio-signal-processing-exercise

**Data Driven Audio Signal Processing - A Tutorial with Computational Examples**

This tutorial accompanies the lecture [Data Driven Audio Signal Processing](https://github.com/spatialaudio/data-driven-audio-signal-processing-lecture). The lecture and the tutorial are designed for International Standard Classification of Education (ISCED) level 7 (Master).

Jupyter notebooks can be accessed via the services

- **dynamic** version using **mybinder**: https://mybinder.org/v2/gh/spatialaudio/data-driven-audio-signal-processing-exercise/main?labpath=index.ipynb
- **static** version using **nbviewer**: https://nbviewer.org/github/spatialaudio/data-driven-audio-signal-processing-exercise/blob/main/index.ipynb
- **sources** (tex, ipynb) at: https://github.com/spatialaudio/data-driven-audio-signal-processing-exercise

## Branch Conventions

- we use the `main` branch as presentation branch, i.e. with plots / results rendered (for student's convenience)
- we use the `dev` branch as the developing main branch, i.e. all notebook outputs are cleared for convenient diff handling
- note that the `main` branch could be **hard reset** from time to time in order to represent an actual desired state of the learning material
- so please don't rely on `main` related commits, but rather act on the `dev` commits, where git history is not changed!!

## Versions / Tags

- TBA for winter term 2021/2022

## Anaconda Environment for Local Usage

The [Anaconda distribution](https://www.anaconda.com/distribution/) is a convenient solution to install a required environment, i.e. to have access to the Jupyter Notebook renderer with a Python interpreter on a personal computer. It is very likely that a very recent installation of Anaconda already delivers all required packages just using the `base` environment. It is however good practice to create a dedicated environment for each project. So, for this tutorial we might use a `myddasp` (or whatever name works for us) environment.

- `conda create -n myddasp python=3.9 pip numpy scipy librosa tensorflow scikit-learn pandas matplotlib notebook jupyterlab ipykernel nb_conda jupyter_nbextensions_configurator jupyter_contrib_nbextensions autopep8`
- `pip install pyloudnorm`

- under `conda 4.10.3` and `conda-build 3.21.4` the current environment to develop and test the notebooks locally by github user *fs446* is
`conda create -n myddasp python=3.9.7 pip=21.3.1 numpy=1.21.3 scipy=1.7.1 librosa=0.8.1 tensorflow=2.4.3 scikit-learn=1.0.1 pandas=1.3.4 matplotlib=3.4.3  notebook=6.4.5 jupyterlab=3.2.1 ipykernel=6.4.2 nb_conda=2.2.1 jupyter_nbextensions_configurator=0.4.1 jupyter_contrib_nbextensions=0.5.1 autopep8=1.6.0`
- `pip install pyloudnorm==0.1.0`

- activate this environment with `conda activate myddasp`

- Jupyter notebook renderer needs to know our dedicated environment:
`python -m ipykernel install --user --name myddasp --display-name "myddasp"`

- get into the folder where the exercises are located, e.g. `cd my_ddasp_folder`

- we might want to archive the actually installed package versions by: `python -m pip list > detailed_packages_list_pip.txt` and `conda env export --no-builds > detailed_packages_list_conda.txt`

- start either a Jupyter notebook or Jupyter lab working environment via a local server instance by either `jupyter notebook` or `jupyter lab`

- start the notebook `index.ipynb`as the landing page for the tutorial

- make sure that the notebook we want to work with is using our kernel `myddasp`

## Referencing

Please cite this open educational resource (OER) project as
*Frank Schultz, Data Driven Audio Signal Processing - A Tutorial Featuring Computational Examples, University of Rostock* ideally with relevant ``file(s), github URL, commit number and/or version tag, year``.

## License

- Creative Commons Attribution 4.0 International License (CC BY 4.0) for text/graphics
- MIT License for software

## Authorship

- University of Rostock:
    - Frank Schultz
    - Sascha Spors
