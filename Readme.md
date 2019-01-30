# Assignments for Numerical Linear Algebra

## Introduction

This is the repository of assignments of the course *Numerical Algebra* Lectured by *Jun Hu* in Fall 2018. The author of this repository is [Zhihan Li](mailto:lzh2016p@pku.edu.cn).

This repository relies on the personal TeX templates package [ptmpls](https://github.com/pppppass/ptmpls). Remember to specify `--recursive` option when cloning the repository.

## Organization

There are several sub folders in this repository:
1. `ptmpls`: Personal TeX / LaTeX templates;
2. `WxxExercise`: Written exercises for week `xx`.
3. `ExxExercise`: Written exercises for chapter `xx`.
4. `PxxYyyy`: Coding exercises with index `xx` realted to `Yyyy`.

## Environment

The numerical results are all produced in a specific environment. The hardware configuration can be found in `hardware.txt`. An Anaconda environment is set up according to `environment.yml`.

## Usage

There are Makefiles distributed in folders.

To compile reports:
1. Install TeX Live or other TeX utilities with LuaLaTeX and ensure that they can be found by `PATH`;
2. Execute `make` in the root folder to recursively compile all reports;
3. Or proceed down to a folder and execute `make report` to generate one single report;
4. The reports are `Answer.pdf` or `Report.pdf` in each folder.

To reproduce the numerical results:
1. Activate Anaconda and execute `make environment` in the root folder to create a new environment `numalg`;
2. Activate the environment `numalg`;
3. Proceed down to a folder and execute `make run` to reproduce numerical results;
4. Execute `make report` or `make` to regenerate the report;
5. The numerical results are updated in the report, i.e., `Answer.pdf` or `Report.pdf`.
