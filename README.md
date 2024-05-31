# Reproducibility code for MMDAgg: MMD Aggregated Two-Sample Test

This GitHub repository contains the code for the reproducible experiments presented in our paper 
[MMD Aggregated Two-Sample Test](https://arxiv.org/abs/2110.15073).

We provide the code to run the experiments to generate Figures 1-10 and Table 2 from our paper, 
those can be found in [media](media).
The code for the [Failing Loudly](https://github.com/steverab/failing-loudly) experiment (with results reported in Table 1) can be found on the [FL-MMDAgg](https://github.com/antoninschrab/FL-MMDAgg) repository.

To use our MMDAgg test in practice, we recommend using our `mmdagg` package, more details available on the [mmdagg](https://github.com/antoninschrab/mmdagg) repository.

Our implementation uses two quantile estimation methods (wild bootstrap and permutations).
The MMDAgg test aggregates over different types of kernels (e.g. Gaussian, Laplace, Inverse Multi-Quadric (IMQ), Matérn (with various parameters) kernels), each with several bandwidths.
In practice, we recommend aggregating over both Gaussian and Laplace kernels, each with 10 bandwidths.

## Requirements
- `python 3.9`

The packages in [requirements.txt](requirements.txt) are required to run our tests and the ones we compare against. 

Additionally, the `jax` and `jaxlib` packages are required to run the Jax implementation of MMDAgg in [mmdagg/jax.py](mmdagg/jax.py).

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
The packages required for reproducibility of the experiments can then be installed in the virtual environment by running
```
python -m pip install -r requirements.txt
```

For using the Jax implementation of MMDAgg, Jax needs to be installed ([instructions](https://github.com/google/jax#installation)). For example, this can be done by running
- for GPU:
  ```bash
  pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
  # conda install -c conda-forge -c nvidia pip numpy scipy cuda-nvcc "jaxlib=0.4.1=*cuda*" jax
  ```
- or, for CPU:
  ```bash
  conda install -c conda-forge -c nvidia pip jaxlib=0.4.1 jax
  ```

## Reproducing the experiments of the paper

To run the experiments, the following command can be executed
```
python experiments.py
```
This command saves the results in dedicated `.csv` and `.pkl` files in a new directory `user/raw`.
The output of this command is already provided in [paper/raw](paper/raw).
The results of the rest of the experiments, saved in the [results](results) directory, can be obtained by running the [Computations_mmdagg.ipynb](Computations_mmdagg.ipynb) notebook and the [Computations_autotst.ipynb](Computations_autotst.ipynb) notebook which uses the [autotst](https://github.com/jmkuebler/auto-tst) package introduced in the [AutoML Two-Sample Test](https://arxiv.org/abs/2206.08843) paper.

The actual figures of the paper can be obtained from the saved results by running the code in the [figures.ipynb](figures.ipynb) notebook.

All the experiments are comprised of 'embarrassingly parallel for loops', significant speed up can be obtained by using parallel computing libraries such as `joblib` or `dask`.

## Data

Half of the experiments uses a down-sampled version of the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset which is created as a `.data` file in a new directory `mnist_dataset` when running the script [experiments.py](experiments.py).
This dataset can also be generated on its own by executing
```
python mnist.py
```
The other half of the experiments uses samples drawn from a perturbed uniform density ([Eq. 17](https://arxiv.org/pdf/2110.15073.pdf)).
A rejection sampler `f_theta_sampler` for this density is implemented in [sampling.py](sampling.py).

## How to use MMDAgg in practice?

The MMDAgg test is implemented as the function `mmdagg` in [mmdagg/np.py](mmdagg/np.py) for the Numpy version and in [mmdagg/jax.py](mmdagg/jax.py) for the Jax version.

For the Numpy implementation of our MMDAgg test, we only require the `numpy` and `scipy` packages.

For the Jax implementation of our MMDAgg test, we only require the `jax` and `jaxlib` packages.

To use our tests in practice, we recommend using our `mmdagg` package which is available on the [mmdagg](https://github.com/antoninschrab/mmdagg) repository. It can be installed by running
```bash
pip install git+https://github.com/antoninschrab/mmdagg.git
```
Installation instructions and example code are available on the [mmdagg](https://github.com/antoninschrab/mmdagg) repository. 

We also provide some code showing how to use our MMDAgg test in the [demo_speed.ipynb](demo_speed.ipynb) notebook which also contains speed comparisons between the Jax and Numpy implementations, as reported below.

| Speed in s | Numpy (CPU) | Jax (CPU) | Jax (GPU) | 
| -- | -- | -- | -- |
| MMDAgg | 43.1 | 14.9 | 0.495 | 

In practice, we recommend using the Jax implementation as it runs considerably faster (100 times faster in the above table, see notebook [demo_speed.ipynb](demo_speed.ipynb)).
 
## References

*Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift.*
Stephan Rabanser, Stephan Günnemann, Zachary C. Lipto.
([paper](https://arxiv.org/abs/1810.11953), [code](https://github.com/steverab/failing-loudly))

*Learning Kernel Tests Without Data Splitting.*
Jonas M. Kübler, Wittawat Jitkrittum, Bernhard Schölkopf, Krikamol Muandet.
([paper](https://arxiv.org/abs/2006.02286), [code](https://github.com/jmkuebler/tests-wo-splitting))

*AutoML Two-Sample Test.*
Jonas M. Kübler, Vincent Stimper, Simon Buchholz, Krikamol Muandet, Bernhard Schölkopf.
([paper](https://arxiv.org/abs/2206.08843), [code](https://github.com/jmkuebler/auto-tst))


## MMDAggInc

For a computationally efficient version of MMDAgg which can run in linear time, check out our paper [Efficient Aggregated Kernel Tests using Incomplete U-statistics](https://arxiv.org/pdf/2206.09194.pdf) with reproducible experiments in the [agginc-paper](https://github.com/antoninschrab/agginc-paper) repository and a package in the [agginc](https://github.com/antoninschrab/agginc) repository.

## Contact

If you have any issues running our code, please do not hesitate to contact [Antonin Schrab](https://antoninschrab.github.io).

## Affiliations

Centre for Artificial Intelligence, Department of Computer Science, University College London

Gatsby Computational Neuroscience Unit, University College London

Inria London

## Bibtex

```
@article{schrab2021mmd,
  author  = {Antonin Schrab and Ilmun Kim and M{\'e}lisande Albert and B{\'e}atrice Laurent and Benjamin Guedj and Arthur Gretton},
  title   = {{MMD} Aggregated Two-Sample Test},
  journal = {Journal of Machine Learning Research},
  year    = {2023},
  volume  = {24},
  number  = {194},
  pages   = {1--81},
  url     = {http://jmlr.org/papers/v24/21-1289.html}
}
```

## License

MIT License (see [LICENSE.md](LICENSE.md)).

## Related tests

- [ksdagg](https://github.com/antoninschrab/ksdagg/): KSD Aggregated KSDAgg test
- [agginc](https://github.com/antoninschrab/agginc/): Efficient MMDAggInc HSICAggInc KSDAggInc tests
- [mmdfuse](https://github.com/antoninschrab/mmdfuse/): MMD-Fuse test
- [dpkernel](https://github.com/antoninschrab/dpkernel/): Differentially private dpMMD dpHSIC tests
- [dckernel](https://github.com/antoninschrab/dckernel/): Robust to Data Corruption dcMMD dcHSIC tests
