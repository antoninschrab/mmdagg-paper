"""
Running this script creates two files results.csv and results.plk in 
the 'user/raw' directory containing the relevant parameters and the estimated 
power/level of the tests. We use the following numbering of experiments: 
uniform mnist 
   1      2      
   3      4
   5      6
   7      8
   9      10
   11     12
Experiments i and i+1 are the same but using the uniform and mnist data.
We first run the uniform experiments followed by the mnist experiments.
The settings of all those experiments can be understood by their 
relations to the figures presented in our paper:
Figure 3 and 4: experiments 1 and 3
Figure 5: experiments 2 and 4
Figure 6: experiments 9 and 10
Figure 7: experiments 1, 2, 3 and 4
Figure 8: experiments 7 and 8
Figure 9: experiments 1, 2, 11 and 12
Table 1: experiments 5 and 6
The figures and table can be created by running the figures.py script.
"""

import numpy as np
import itertools
import pandas as pd
from mnist import download_mnist, load_mnist
from pathlib import Path
from seed import generate_seed
from sample_test import sample_and_test_uniform, sample_and_test_mnist
import argparse

# create results directory if it does not exist
Path("user/raw").mkdir(exist_ok=True, parents=True)

# panda dataframe: lists of indices and entries
index_vals = []
results = []

# parameters shared for all experiments
alpha = 0.05
B1 = 500
B2 = 500
B3 = 100
kernel_types = ["gaussian", "laplace"]
k_num = len(kernel_types)
approx_types = ["wild bootstrap", "permutation"]
a_num = len(approx_types)

############# UNIFORM #############
# parameters for all uniform experiments
s = 1
perturbation_multipliers = [2.7, 7.3]
bandwidth_multipliers = np.linspace(0.1, 1, 10)
e_num = 2
p_num = 4

# Experiment 1
exp = "1: uniform alternative"
repetitions = 500
sample_sizes = [500, 2000]
L = [(-6, -2), (-4, 0), (-2, 2)]
l_num = len(L)
function_types = ["uniform", "increasing", "decreasing", "centred", "ost"]
f_num = len(function_types)

for a, k, e, l, f, p in itertools.product(
    range(a_num), range(k_num), range(2), range(l_num), range(f_num), range(p_num)
):
    if (a, l) not in [(1, 0), (1, 2)] and (e, p) != (1, 3):
        approx_type = approx_types[a]
        kernel_type = kernel_types[k]
        d = e + 1
        n = m = sample_sizes[e]
        perturbation_multiplier = perturbation_multipliers[e]
        l_minus, l_plus = L[l]
        l_minus_l_plus = L[l]
        function_type = function_types[f]
        perturbation = p + 1
        perturbation_or_Qi = perturbation
        test_output_list = []
        for i in range(repetitions):
            seed = generate_seed(k, e, l, f, p, i)
            test_output_list.append(
                sample_and_test_uniform(
                    function_type,
                    seed,
                    kernel_type,
                    approx_type,
                    m,
                    n,
                    d,
                    perturbation,
                    s,
                    perturbation_multiplier,
                    alpha,
                    l_minus,
                    l_plus,
                    B1,
                    B2,
                    B3,
                    bandwidth_multipliers,
                )
            )
        power = np.mean(test_output_list)
        index_val = (
            exp,
            d,
            repetitions,
            m,
            n,
            approx_type,
            kernel_type,
            l_minus_l_plus,
            function_type,
            perturbation_or_Qi,
        )
        index_vals.append(index_val)
        results.append(power)
print('Experiment 1 completed.')

# Experiment 3
exp = "3: uniform alternative"
repetitions = 500
sample_sizes = [500, 2000]
l_minus = l_plus = None
l_minus_l_plus = None
function_types = ["median", "split", "split (doubled sample sizes)"]
f_num = len(function_types)

for a, k, e, f, p in itertools.product(
    range(a_num), range(k_num), range(e_num), range(f_num), range(p_num)
):
    if (e, p) != (1, 3):
        approx_type = approx_types[a]
        kernel_type = kernel_types[k]
        d = e + 1
        n = m = sample_sizes[e]
        perturbation_multiplier = perturbation_multipliers[e]
        function_type = function_types[f]
        perturbation = p + 1
        perturbation_or_Qi = perturbation
        test_output_list = []
        for i in range(repetitions):
            seed = generate_seed(k, e, 3, f, p, i)
            test_output_list.append(
                sample_and_test_uniform(
                    function_type,
                    seed,
                    kernel_type,
                    approx_type,
                    m,
                    n,
                    d,
                    perturbation,
                    s,
                    perturbation_multiplier,
                    alpha,
                    l_minus,
                    l_plus,
                    B1,
                    B2,
                    B3,
                    bandwidth_multipliers,
                )
            )
        power = np.mean(test_output_list)
        index_val = (
            exp,
            d,
            repetitions,
            m,
            n,
            approx_type,
            kernel_type,
            l_minus_l_plus,
            function_type,
            perturbation_or_Qi,
        )
        index_vals.append(index_val)
        results.append(power)
print('Experiment 3 completed.')

# Experiment 5
exp = "5: uniform null"
repetitions = 5000
sample_sizes = [500, 2000]
function_types = [
    "uniform",
    "increasing",
    "decreasing",
    "centred",
    "ost",
    "median",
    "split",
]
f_num = len(function_types)
L = [(-4, 0)]
l_num = len(L)
l_minus, l_plus = L[0]
l_minus_l_plus = L[0]
perturbation = 0
perturbation_or_Qi = perturbation

for a, k, e, f in itertools.product(
    range(a_num), range(k_num), range(e_num), range(f_num)
):
    approx_type = approx_types[a]
    kernel_type = kernel_types[k]
    d = e + 1
    n = m = sample_sizes[e]
    function_type = function_types[f]
    test_output_list = []
    for i in range(repetitions):
        seed = generate_seed(k, e, 0, f, 5, i)
        test_output_list.append(
            sample_and_test_uniform(
                function_type,
                seed,
                kernel_type,
                approx_type,
                m,
                n,
                d,
                perturbation,
                s,
                perturbation_multiplier,
                alpha,
                l_minus,
                l_plus,
                B1,
                B2,
                B3,
                bandwidth_multipliers,
            )
        )
    power = np.mean(test_output_list)
    index_val = (
        exp,
        d,
        repetitions,
        m,
        n,
        approx_type,
        kernel_type,
        l_minus_l_plus,
        function_type,
        perturbation_or_Qi,
    )
    index_vals.append(index_val)
    results.append(power)
print('Experiment 5 completed.')

# Experiment 7
exp = "7: uniform alternative"
repetitions = 500
sample_sizes_m = [100, 250]
sample_sizes_n = [1000, 2000, 3000, 4000, 5000]
v_num = len(sample_sizes_n)
function_types = ["uniform", "increasing", "decreasing", "centred"]
f_num = len(function_types)
l_minus, l_plus = (-4, 0)
l_minus_l_plus = (-4, 0)
perturbations = [3, 2]
approx_type = "permutation"

for k, e, f, v in itertools.product(
    range(k_num), range(e_num), range(f_num), range(v_num)
):
    kernel_type = kernel_types[k]
    d = e + 1
    n = sample_sizes_n[v]
    m = sample_sizes_m[e]
    perturbation_multiplier = perturbation_multipliers[e]
    function_type = function_types[f]
    perturbation = perturbations[e]
    perturbation_or_Qi = perturbation
    test_output_list = []
    for i in range(repetitions):
        seed = generate_seed(k, e, 3, f, v, i)
        test_output_list.append(
            sample_and_test_uniform(
                function_type,
                seed,
                kernel_type,
                approx_type,
                m,
                n,
                d,
                perturbation,
                s,
                perturbation_multiplier,
                alpha,
                l_minus,
                l_plus,
                B1,
                B2,
                B3,
                bandwidth_multipliers,
            )
        )
    power = np.mean(test_output_list)
    index_val = (
        exp,
        d,
        repetitions,
        m,
        n,
        approx_type,
        kernel_type,
        l_minus_l_plus,
        function_type,
        perturbation_or_Qi,
    )
    index_vals.append(index_val)
    results.append(power)
print('Experiment 7 completed.')

# Experiment 9
exp = "9: uniform alternative"
repetitions = 500
sample_sizes = [500, 2000]
function_types = ["uniform", "centred"]
f_num = len(function_types)
L = [(-2, -2), (-3, -1), (-4, 0), (-5, 1), (-6, 2), (-7, 3), (-8, 4), (-9, 5)]
l_num = len(L)
perturbations = [3, 2]
approx_type = "wild bootstrap"

for k, e, l, f in itertools.product(
    range(k_num), range(e_num), range(l_num), range(f_num)
):
    kernel_type = kernel_types[k]
    d = e + 1
    n = m = sample_sizes[e]
    perturbation_multiplier = perturbation_multipliers[e]
    l_minus, l_plus = L[l]
    l_minus_l_plus = L[l]
    function_type = function_types[f]
    perturbation = perturbations[e]
    p = perturbation - 1
    perturbation_or_Qi = perturbation
    test_output_list = []
    for i in range(repetitions):
        seed = generate_seed(k, e, l, f, p, i)
        test_output_list.append(
            sample_and_test_uniform(
                function_type,
                seed,
                kernel_type,
                approx_type,
                m,
                n,
                d,
                perturbation,
                s,
                perturbation_multiplier,
                alpha,
                l_minus,
                l_plus,
                B1,
                B2,
                B3,
                bandwidth_multipliers,
            )
        )
    power = np.mean(test_output_list)
    index_val = (
        exp,
        d,
        repetitions,
        m,
        n,
        approx_type,
        kernel_type,
        l_minus_l_plus,
        function_type,
        perturbation_or_Qi,
    )
    index_vals.append(index_val)
    results.append(power)
print('Experiment 9 completed.')

# Experiment 11
exp = "11: uniform alternative"
repetitions = 500
function_types = ["ost"]
f = 0
function_type = function_types[f]
l_minus, l_plus = (-4, 0)
l_minus_l_plus = (-4, 0)
l = 3
approx_type = None
sample_sizes = [
    [int(i * 10 ** 5) for i in [0.25, 0.5, 0.75, 1]],
    [int(i * 10 ** 5) for i in [2.5, 5, 7.5, 10]],
]
v_num = len(sample_sizes[0])

for v, k, e, p in itertools.product(
    range(v_num), range(k_num), range(e_num), range(p_num)
):
    if (e, p) != (1, 3):
        kernel_type = kernel_types[k]
        d = e + 1
        n = m = sample_sizes[e][v]
        perturbation_multiplier = perturbation_multipliers[e]
        function_type = function_types[f]
        perturbation = p + 1
        perturbation_or_Qi = perturbation
        test_output_list = []
        for i in range(repetitions):
            seed = generate_seed(k, e, 3, 0, p, i)
            test_output_list.append(
                sample_and_test_uniform(
                    function_type,
                    seed,
                    kernel_type,
                    approx_type,
                    m,
                    n,
                    d,
                    perturbation,
                    s,
                    perturbation_multiplier,
                    alpha,
                    l_minus,
                    l_plus,
                    B1,
                    B2,
                    B3,
                    bandwidth_multipliers,
                )
            )
        power = np.mean(test_output_list)
        index_val = (
            exp,
            d,
            repetitions,
            m,
            n,
            approx_type,
            kernel_type,
            l_minus_l_plus,
            function_type,
            perturbation_or_Qi,
        )
        index_vals.append(index_val)
        results.append(power)
print('Experiment 11 completed.')


############# MNIST #############
# parameters for all mnist experiments
p_num = 5
d = None
Q_list = ["Q1", "Q2", "Q3", "Q4", "Q5"]
bandwidth_multipliers = np.array([2 ** i for i in range(10, 21)])
Path("mnist_dataset").mkdir(exist_ok=True)
if Path("mnist_dataset/mnist_7x7.data").is_file() == False:
    download_mnist()
P, Q_list = load_mnist()
Q_list_str = ["Q1", "Q2", "Q3", "Q4", "Q5"]

# Experiment 2
exp = "2: mnist alternative"
repetitions = 500
sample_size = 500
function_types = ["uniform", "increasing", "decreasing", "centred", "ost"]
f_num = len(function_types)
L = [[(8, 12), (10, 14), (12, 16)], [(10, 14), (12, 16), (14, 18)]]
l_num = len(L[0])

for a, k, l, f, p in itertools.product(
    range(a_num), range(k_num), range(l_num), range(f_num), range(p_num)
):
    if (a, l) not in [(1, 0), (1, 2)]:
        approx_type = approx_types[a]
        kernel_type = kernel_types[k]
        n = m = sample_size
        l_minus, l_plus = L[k][l]
        l_minus_l_plus = L[k][l]
        function_type = function_types[f]
        Qi = Q_list[p]
        perturbation_or_Qi = Q_list_str[p]
        test_output_list = []
        for i in range(repetitions):
            seed = generate_seed(k, 2, l, f, p, i)
            test_output_list.append(
                sample_and_test_mnist(
                    P,
                    Qi,
                    function_type,
                    seed,
                    kernel_type,
                    approx_type,
                    m,
                    n,
                    alpha,
                    l_minus,
                    l_plus,
                    B1,
                    B2,
                    B3,
                    bandwidth_multipliers,
                )
            )
        power = np.mean(test_output_list)
        index_val = (
            exp,
            d,
            repetitions,
            m,
            n,
            approx_type,
            kernel_type,
            l_minus_l_plus,
            function_type,
            perturbation_or_Qi,
        )
        index_vals.append(index_val)
        results.append(power)
print('Experiment 2 completed.')

# Experiment 4
exp = "4: mnist alternative"
repetitions = 500
sample_size = 500
function_types = ["median", "split", "split (doubled sample sizes)"]
f_num = len(function_types)
l_minus = l_plus = None
l_minus_l_plus = None

for a, k, f, p in itertools.product(
    range(a_num), range(k_num), range(f_num), range(p_num)
):
    approx_type = approx_types[a]
    kernel_type = kernel_types[k]
    n = m = sample_size
    function_type = function_types[f]
    Qi = Q_list[p]
    perturbation_or_Qi = Q_list_str[p]
    test_output_list = []
    for i in range(repetitions):
        seed = generate_seed(k, 2, 3, f, p, i)
        test_output_list.append(
            sample_and_test_mnist(
                P,
                Qi,
                function_type,
                seed,
                kernel_type,
                approx_type,
                m,
                n,
                alpha,
                l_minus,
                l_plus,
                B1,
                B2,
                B3,
                bandwidth_multipliers,
            )
        )
    power = np.mean(test_output_list)
    index_val = (
        exp,
        d,
        repetitions,
        m,
        n,
        approx_type,
        kernel_type,
        l_minus_l_plus,
        function_type,
        perturbation_or_Qi,
    )
    index_vals.append(index_val)
    results.append(power)
print('Experiment 4 completed.')

# Experiment 6
exp = "6: mnist null"
repetitions = 5000
sample_size = 500
function_types = [
    "uniform",
    "increasing",
    "decreasing",
    "centred",
    "ost",
    "median",
    "split",
]
f_num = len(function_types)
L = [(10, 14), (12, 16)]

for a, k, f in itertools.product(range(a_num), range(k_num), range(f_num)):
    approx_type = approx_types[a]
    kernel_type = kernel_types[k]
    n = m = sample_size
    function_type = function_types[f]
    l_minus, l_plus = L[k]
    l_minus_l_plus = L[k]
    perturbation_or_Qi = "P"
    test_output_list = []
    for i in range(repetitions):
        seed = generate_seed(k, 2, l, f, 5, i)
        test_output_list.append(
            sample_and_test_mnist(
                P,
                P,
                function_type,
                seed,
                kernel_type,
                approx_type,
                m,
                n,
                alpha,
                l_minus,
                l_plus,
                B1,
                B2,
                B3,
                bandwidth_multipliers,
            )
        )
    power = np.mean(test_output_list)
    index_val = (
        exp,
        d,
        repetitions,
        m,
        n,
        approx_type,
        kernel_type,
        l_minus_l_plus,
        function_type,
        perturbation_or_Qi,
    )
    index_vals.append(index_val)
    results.append(power)
print('Experiment 6 completed.')

# Experiment 8
exp = "8: mnist alternative"
repetitions = 500
sample_size_m = 100
sample_sizes_n = [200, 400, 600, 800, 1000]
v_num = len(sample_sizes_n)
function_types = ["uniform", "increasing", "decreasing", "centred"]
f_num = len(function_types)
L = [(10, 14), (12, 16)]
approx_type = "permutation"

for k, f, v in itertools.product(range(k_num), range(f_num), range(v_num)):
    kernel_type = kernel_types[k]
    n = sample_sizes_n[v]
    m = sample_size_m
    function_type = function_types[f]
    Qi = Q_list[2]
    perturbation_or_Qi = Q_list_str[2]
    test_output_list = []
    for i in range(repetitions):
        seed = generate_seed(k, 2, 3, f, v, i)
        test_output_list.append(
            sample_and_test_mnist(
                P,
                Qi,
                function_type,
                seed,
                kernel_type,
                approx_type,
                m,
                n,
                alpha,
                l_minus,
                l_plus,
                B1,
                B2,
                B3,
                bandwidth_multipliers,
            )
        )
    power = np.mean(test_output_list)
    index_val = (
        exp,
        d,
        repetitions,
        m,
        n,
        approx_type,
        kernel_type,
        l_minus_l_plus,
        function_type,
        perturbation_or_Qi,
    )
    index_vals.append(index_val)
    results.append(power)
print('Experiment 8 completed.')

# Experiment 10
exp = "10: mnist alternative"
repetitions = 500
sample_sizes = 500
function_types = ["uniform", "centred"]
f_num = len(function_types)
L = [
    [(12, 12), (11, 13), (10, 14), (9, 15), (8, 16), (7, 17), (6, 18), (5, 19)],
    [(14, 14), (13, 15), (12, 16), (11, 17), (10, 18), (9, 19), (8, 20), (7, 21)],
]
l_num = len(L[0])
approx_type = "wild bootstrap"

for k, l, f in itertools.product(range(k_num), range(l_num), range(f_num)):
    kernel_type = kernel_types[k]
    n = m = sample_sizes
    l_minus, l_plus = L[k][l]
    l_minus_l_plus = L[k][l]
    function_type = function_types[f]
    Qi = Q_list[2]
    perturbation_or_Qi = Q_list_str[2]
    test_output_list = []
    for i in range(repetitions):
        seed = generate_seed(k, 2, l, f, v, i)
        test_output_list.append(
            sample_and_test_mnist(
                P,
                Qi,
                function_type,
                seed,
                kernel_type,
                approx_type,
                m,
                n,
                alpha,
                l_minus,
                l_plus,
                B1,
                B2,
                B3,
                bandwidth_multipliers,
            )
        )
    power = np.mean(test_output_list)
    index_val = (
        exp,
        d,
        repetitions,
        m,
        n,
        approx_type,
        kernel_type,
        l_minus_l_plus,
        function_type,
        perturbation_or_Qi,
    )
    index_vals.append(index_val)
    results.append(power)
print('Experiment 10 completed.')

# Experiment 12
exp = "12: mnist alternative"
repetitions = 500
function_types = ["ost"]
f = 0
function_type = function_types[f]
L = [(10, 14), (12, 16)]
l = 3
approx_type = None
sample_sizes = [int(i * 10 ** 5) for i in [0.5, 1, 1.5, 2, 2.5]]
v_num = len(sample_sizes)

for v, k, p in itertools.product(range(v_num), range(k_num), range(p_num)):
    kernel_type = kernel_types[k]
    n = m = sample_sizes[v]
    function_type = function_types[f]
    l_minus, l_plus = L[k]
    l_minus_l_plus = L[k]
    Qi = Q_list[p]
    perturbation_or_Qi = Q_list_str[p]
    test_output_list = []
    for i in range(repetitions):
        seed = generate_seed(k, 2, 3, f, p, i)
        test_output_list.append(
            sample_and_test_mnist(
                P,
                Qi,
                function_type,
                seed,
                kernel_type,
                approx_type,
                m,
                n,
                alpha,
                l_minus,
                l_plus,
                B1,
                B2,
                B3,
                bandwidth_multipliers,
            )
        )
    power = np.mean(test_output_list)
    index_val = (
        exp,
        d,
        repetitions,
        m,
        n,
        approx_type,
        kernel_type,
        l_minus_l_plus,
        function_type,
        perturbation_or_Qi,
    )
    index_vals.append(index_val)
    results.append(power)
print('Experiment 12 completed.')

# save panda dataframe
index_names = (
    "experiment",
    "d",
    "repetitions",
    "m",
    "n",
    "approx_type",
    "kernel_type",
    "l_minus_l_plus",
    "function_type",
    "perturbation_or_Qi",
)
index = pd.MultiIndex.from_tuples(index_vals, names=index_names)
results_df = pd.Series(results, index=index).to_frame("power")
results_df.reset_index().to_csv("user/raw/results.csv")
results_df.to_pickle("user/raw/results.pkl")
