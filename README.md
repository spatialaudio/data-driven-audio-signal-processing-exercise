# data-driven-audio-signal-processing-exercise

**Data Driven Audio Signal Processing - A Tutorial with Computational Examples**

This tutorial accompanies the lecture [Data Driven Audio Signal Processing](https://github.com/spatialaudio/data-driven-audio-signal-processing-lecture). The lecture and the tutorial are designed for International Standard Classification of Education (ISCED) level 7 (Master).

Jupyter notebooks can be accessed via the services

- **dynamic** version using **mybinder**: https://mybinder.org/v2/gh/spatialaudio/data-driven-audio-signal-processing-exercise/dev?labpath=index.ipynb
- **static** version using **nbviewer**: https://nbviewer.org/github/spatialaudio/data-driven-audio-signal-processing-exercise/blob/dev/index.ipynb
- **sources** (tex, ipynb) at: https://github.com/spatialaudio/data-driven-audio-signal-processing-exercise

## Versions / Tags

- [v0.1](https://github.com/spatialaudio/data-driven-audio-signal-processing-exercise/releases/tag/v0.1) for winter term 2021/22, initial version
- TBD for winter term 2022/23

## Branch Conventions

- we use the `dev` branch as the developing branch, i.e. all notebook outputs are cleared for convenient diff handling
- we use the `main` branch as presentation branch, i.e. notebook outputs (such as plots, results) are included for students' convenience
- note that we **hard reset** `main` branch from time to time in order to represent an actual desired state of the material
- so please do not rely on `main` related commits, but rather act on the `dev` commits, where git history is not changed

## Anaconda Environment for Local Usage

The [Anaconda distribution](https://www.anaconda.com/distribution/) is a convenient solution to install a required environment, i.e. to have access to the Jupyter Notebook renderer with a Python interpreter on a personal computer. It is very likely that a very recent installation of Anaconda already delivers all required packages just using the `base` environment. It is however good practice to create a dedicated environment for each project. So, for this tutorial we might use a `myddasp` (or whatever name works for us) environment.

- get into the folder where the exercises are located, e.g. `cd my_ddasp_folder`
- in the subfolder `.binder` the `environment.yml` can be used to create a dedicated conda `myddasp` environment as
    - `conda env create -f environment.yml --force`
    - we can remove this environment with `conda env remove --name myddasp`
- this should also have installed sound/audio related libraries using pip
    - `pip install sounddevice==0.4.4`
    - `pip install soundfile==0.10.3.post1`
    - `pip install pyloudnorm==0.1.0`
    - we might check this with `pip list`
- activate this environment with `conda activate myddasp`
- Jupyter notebook renderer needs to know our dedicated environment:
`python -m ipykernel install --user --name myddasp --display-name "myddasp"`
- we might want to archive the actually installed package versions by
    - `python -m pip list > detailed_packages_list_pip.txt` and
    - `conda env export --no-builds > detailed_packages_list_conda.txt`
- start either a Jupyter notebook or Jupyter lab working environment via a local server instance by either `jupyter notebook` or `jupyter lab`
- start the landing page `index.ipynb` of the tutorial
- make sure that the notebooks we want to work with are using our dedicated kernel `myddasp`

## Authorship

- University of Rostock:
    - [Frank Schultz](https://orcid.org/0000-0002-3010-0294)
    - [Sascha Spors](https://orcid.org/0000-0001-7225-9992)

## Referencing

Please cite this open educational resource (OER) project as
*Frank Schultz, Data Driven Audio Signal Processing - A Tutorial Featuring Computational Examples, University of Rostock* ideally with relevant ``file(s), github URL, commit number and/or version tag, year``.

## License

- Creative Commons Attribution 4.0 International License (CC BY 4.0) for text/graphics
- MIT License for software

