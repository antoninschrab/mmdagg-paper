# Code for MMDAgg: an MMD aggregated two-sample test

An improved implementation of MMDAgg which is more memory-efficient can be found [here](https://github.com/antoninschrab/FL-MMDAgg/tree/master/mmdaggupdate).

This GitHub repository contains the code for the reproducible experiments presented in our paper 
[MMD Aggregated Two-Sample Test](https://arxiv.org/abs/2110.15073).
We provide the code to run the experiments to generate Figures 1-9 and Table 1 from our paper, 
those can be found in [media](media). 

The function `mmdagg` in [tests.py](tests.py) corresponds to the aggregated test we propose
in [Algorithm 1](https://arxiv.org/pdf/2110.15073.pdf#page=17), it uses the efficient implementation
discussed in [Appendix C](https://arxiv.org/pdf/2110.15073.pdf#page=44).
This function can be used with Gaussian and Laplace kernels, considering 
a collection of bandwidths consisting of the median bandwidth scaled by powers of 2, and
using one of the four types of weights we propose in [Section 5.1](https://arxiv.org/pdf/2110.15073.pdf#page=22).

We also include a more general implementation `mmdagg_custom` of our test in [tests.py](tests.py) which can be used 
by providing any kernel matrices and using any weighting strategy.
Those kernel matrices can be obtained by fixing a kernel and varying the bandwidths, 
but they can also be obtained by considering fundamentally different kernels.

## Requirements
- `python 3.9`

## Installation

In a chosen directory, clone the repository and change to its directory by executing 
```
git clone git@github.com:antoninschrab/mmdagg-paper.git
cd mmdagg-paper
```
We then recommend creating and activating a virtual environment by either 
- using `venv`:
  ```
  python3 -m venv mmdagg-env
  source mmdagg-env/bin/activate
  # can be deactivated by running:
  # deactivate
  ```
- or using `conda`:
  ```
  conda create --name mmdagg-env python=3.9
  conda activate mmdagg-env
  # can be deactivated by running:
  # conda deactivate
  ```
The required packages can then be installed in the virtual environment by running
```
python -m pip install -r requirements.txt
```

## Reproducing the experiments of the paper

To run the experiments, the following command can be executed
```
python experiments.py 
```
This command creates all the data necessary to plot the figures of the paper and saves it in dedicated `.csv` and `.pkl` files 
in a new directory `user/raw`.
As the experiments take several days to run, the output of this command is already provided in [paper/raw](paper/raw).
The actual figures of the paper can be obtained from such files using the following command 
```
python figures.py  # create figures in paper/figures using reference files in paper/raw
# python figures.py -user  # alternatively, create figures in user/figures using output files in user/raw
```
The experiments in [experiments.py](experiments.py) are comprised of 'embarrassingly parallel for loops', significant speed up can be obtained by using 
parallel computing libraries such as `joblib` or `dask`.

## Data

Half of the experiments uses a down-sampled version of the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset which is created as a `.data` file in a new directory `mnist_dataset` when running the script [experiments.py](experiments.py). 
This dataset can also be generated on its own by executing 
```
python mnist.py
```
The other half of the experiments uses samples drawn from a perturbed uniform density ([Eq. (17)](https://arxiv.org/pdf/2110.15073.pdf#page=27)).
A rejection sampler `f_theta_sampler` for this density is implemented in [sampling.py](sampling.py).

## Author

[Antonin Schrab](https://antoninschrab.github.io)

Centre for Artificial Intelligence, Department of Computer Science, University College London

Gatsby Computational Neuroscience Unit, University College London

Inria, Lille - Nord Europe research centre and Inria London Programme

## Bibtex

```
@unpublished{schrab2021mmd,
  title={{MMD} Aggregated Two-Sample Test},
  author={Antonin Schrab and Ilmun Kim and M{\'e}lisande Albert and B{\'e}atrice Laurent and Benjamin Guedj and Arthur Gretton},
  year={2021},
  note = "Submitted.",
  abstract = {We propose a novel nonparametric two-sample test based on the Maximum Mean Discrepancy (MMD), which is constructed by aggregating tests with different kernel bandwidths. This aggregation procedure, called MMDAgg, ensures that test power is maximised over the collection of kernels used, without requiring held-out data for kernel selection (which results in a loss of test power), or arbitrary kernel choices such as the median heuristic. We work in the non-asymptotic framework, and prove that our aggregated test is minimax adaptive over Sobolev balls. Our guarantees are not restricted to a specific kernel, but hold for any product of one-dimensional translation invariant characteristic kernels which are absolutely and square integrable. Moreover, our results apply for popular numerical procedures to determine the test threshold, namely permutations and the wild bootstrap. Through numerical experiments on both synthetic and real-world datasets, we demonstrate that MMDAgg outperforms alternative state-of-the-art approaches to MMD kernel adaptation for two-sample testing.},
  url = {https://arxiv.org/abs/2110.15073},
  url_PDF = {https://arxiv.org/pdf/2110.15073.pdf},
  url_Code = {https://github.com/antoninschrab/mmdagg-paper},
  eprint={2110.15073},
  archivePrefix={arXiv},
  primaryClass={stat.ML}
}
```

## License

MIT License (see [LICENSE.md](LICENSE.md))
