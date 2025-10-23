# data-driven-audio-signal-processing-exercise

**Data Driven Audio Signal Processing - A Tutorial with Computational Examples**

This tutorial accompanies the lecture [Data Driven Audio Signal Processing](https://github.com/spatialaudio/data-driven-audio-signal-processing-lecture). The lecture and the tutorial are designed for International Standard Classification of Education (ISCED) level 7 (Master, in total 6 ECTS credits).

Jupyter notebooks can be accessed via the services

- **dynamic** version using **mybinder**: https://mybinder.org/v2/gh/spatialaudio/data-driven-audio-signal-processing-exercise/dev?labpath=index.ipynb
- **static** version using **nbviewer**: https://nbviewer.org/github/spatialaudio/data-driven-audio-signal-processing-exercise/blob/dev/index.ipynb
- **sources** (tex, ipynb) at: https://github.com/spatialaudio/data-driven-audio-signal-processing-exercise

## Versions / Tags

- [v0.1](https://github.com/spatialaudio/data-driven-audio-signal-processing-exercise/releases/tag/v0.1) for winter term 2021/22, initial version
- [v0.2](https://github.com/spatialaudio/data-driven-audio-signal-processing-exercise/releases/tag/v0.2) for winter term 2022/23
- [v0.3](https://github.com/spatialaudio/data-driven-audio-signal-processing-exercise/releases/tag/v0.3) for winter term 2023/24, many beamer tex slides added, CI
- [v0.4](https://github.com/spatialaudio/data-driven-audio-signal-processing-exercise/releases/tag/v0.4) winter term 2024/25, smaller mods due to API changes, PCA example on exam grades, slides
- [v0.5](https://github.com/spatialaudio/data-driven-audio-signal-processing-exercise/releases/tag/v0.5) winter term 2025/26, TBD

## Branch Conventions

- the **default branch** of the repository is `dev` and this is used for development
- the `dev` branch contains notebooks with cleared outputs for convenient diff handling
- the `main` branch contains notebooks with rendered outputs, which is maintained from time to time
- do **not** rely on `main` branch as this is hard reset from time to time
- probably in future we rename `main` to somewhat less confusing

## Python Environment
- the `pyproject.toml` contains the project info
- assuming we use uv for Python, packaging and environment handling a dedicated environment can be cerated with `uv sync`
 

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
