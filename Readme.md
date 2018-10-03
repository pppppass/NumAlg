# Assignments for Numerical Linear Algebra

## Introduction

This is the repository of assignments of the course *Numerical Algebra* Lectured by *Jun Hu* in Fall 2018. The author of this repository is [Zhihan Li](mailto:lzh2016p@pku.edu.cn).

This repository relies on the personal TeX templates package [ptmpls](https://github.com/pppppass/ptmpls). Remember to specify `--recursive` option when cloning the repository.

## Organization

There are several sub folders in this repository:
1. `WxxExercise`: Written exercise for week `xx`.
2. `ptmpls`: Personal TeX / LaTeX templates.

## Environment

All the numerical results are carried out on a specific environment. The hardware configuration can be found in `hardware.txt`. An Anaconda environment is also set up according to `environment.yml`.

## Usage

There are Makefiles distributed in folders and one may execute `make` on the root folder to recursively compile all reports. One may also proceed down to each folder and execute `make run` to run the codes or `make report` to generate one single report. One may also execute `make environment`, when Anaconda is activated, to create an Anaconda environment according to `environment.yml`.
