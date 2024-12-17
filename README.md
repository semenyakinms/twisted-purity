# Twisted purities

This repository contais code computing twisted purities, accompanying paper "Classifying fermionic states via many-body correlation measures" by Mykola Semenyakin, Yevheniia Cheipesh and Yaroslav Herasymenko ([arXiv:2309.07956](https://arxiv.org/abs/2309.07956)).

**Installation**

The code is essentially using QuSpin library, for installation see
```
https://quspin.github.io/QuSpin/
```

**Usage**

The main file in repository, which is producing Figure 2 from the paper is [plots.ipynb](/plots.ipynb). The notebooks [Haar.ipynb](/Haar.ipynb), [Hubbard.ipynb](/Hubbard.ipynb) and [SYK4.ipynb](/SYK4.ipynb) can be used to compute necessary numerical results. Folders [results_Haar](/results_Haar), [results_Hubbard](/results_Hubbard) and [results_SYK](/results_SYK) contain some precomputed simulations. Folder [modules](/modules) contains the core code of simulations.

For historical reasons in this repository we nickname "twisted purities" by "higher Pluckers".
