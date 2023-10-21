# data-driven-audio-signal-processing-exercise

**Data Driven Audio Signal Processing - A Tutorial with Computational Examples**

This tutorial accompanies the lecture [Data Driven Audio Signal Processing](https://github.com/spatialaudio/data-driven-audio-signal-processing-lecture). The lecture and the tutorial are designed for International Standard Classification of Education (ISCED) level 7 (Master, in total 6 ECTS credits).

Jupyter notebooks can be accessed via the services

- **dynamic** version using **mybinder**: https://mybinder.org/v2/gh/spatialaudio/data-driven-audio-signal-processing-exercise/dev?labpath=index.ipynb
- **static** version using **nbviewer**: https://nbviewer.org/github/spatialaudio/data-driven-audio-signal-processing-exercise/blob/dev/index.ipynb
- **sources** (tex, ipynb) at: https://github.com/spatialaudio/data-driven-audio-signal-processing-exercise

Jupyter notebooks with rendered **outputs** can be viewed at https://nbviewer.org/github/spatialaudio/data-driven-audio-signal-processing-exercise/blob/main/index.ipynb

## Versions / Tags

- [v0.1](https://github.com/spatialaudio/data-driven-audio-signal-processing-exercise/releases/tag/v0.1) for winter term 2021/22, initial version
- [v0.2](https://github.com/spatialaudio/data-driven-audio-signal-processing-exercise/releases/tag/v0.2) for winter term 2022/23
- TBD for winter term 2023/24

## Branch Conventions

- the **default branch** of the repository is `dev` used for development 
- all notebook outputs in `dev` branch are cleared for convenient diff handling
- `main` branch contains notebooks with rendered outputs, which is maintained from time to time
- do **not** rely on `main` branch as this is hard reset from time to time
- probably in future we rename `main` to somewhat less confusing

## Anaconda Environment for Local Usage

The [Anaconda distribution](https://www.anaconda.com/distribution/) is a convenient solution to install a required environment, i.e. to have access to a Jupyter Notebook renderer with a Python interpreter on a personal computer. It is very likely that a very recent installation of Anaconda already delivers most of the required standard packages just using the `base` environment. It is however good practice to create a dedicated environment for each project. So, for this tutorial we might use a `myddasp` (or whatever name works for us) environment. We might consider the following install routine:

- clone the repo to local machine (if not already available)
    - `git clone git@github.com:spatialaudio/data-driven-audio-signal-processing-exercise.git` (via SSH) or
    - `git clone https://github.com/spatialaudio/data-driven-audio-signal-processing-exercise.git` (via https) or
    - get a zip file from current `dev` commit via https://github.com/spatialaudio/data-driven-audio-signal-processing-exercise/archive/refs/heads/dev.zip
- get into the folder where the exercises are located, e.g. `cd my_ddasp_folder`
- in the subfolder `.binder` the `environment.yml` can be used to create a dedicated conda `myddasp` environment as
    - `conda env create -f environment.yml --force`
    - (we can remove this environment with `conda env remove --name myddasp`)
- activate this environment with `conda activate myddasp`
- this should also have installed sound / audio related libraries using pip
    - `pip install pyloudnorm==0.1.0`
    - we might check this with `pip list`
- Jupyter notebook renderer needs to know our dedicated environment:
`python -m ipykernel install --user --name myddasp --display-name "myddasp"`
- we might want to archive the actually installed package versions by
    - `python -m pip list > detailed_packages_list_pip.txt` and
    - `conda env export --no-builds > detailed_packages_list_conda.txt`
- start a Jupyter lab environment via a local server instance by `jupyter lab`
- start the landing page `index.ipynb` of the tutorial
- make sure that the notebooks we want to work with are using our dedicated kernel `myddasp`

## Authorship

- University of Rostock:
    - [Frank Schultz](https://orcid.org/0000-0002-3010-0294), concept, coding
    - [Sascha Spors](https://orcid.org/0000-0001-7225-9992), concept

## Referencing

Please cite this open educational resource (OER) project as
*Frank Schultz, Data Driven Audio Signal Processing - A Tutorial Featuring Computational Examples, University of Rostock* ideally with relevant ``file(s), github URL, commit number and/or version tag, year``.

## License

- Creative Commons Attribution 4.0 International License (CC BY 4.0) for text/graphics
- MIT License for software
