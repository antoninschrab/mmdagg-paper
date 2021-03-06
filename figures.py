"""
Running this script generates the figures (Figures 1-9) and table (Table 1) from our paper.
"""

import numpy as np
import itertools
from matplotlib import rc, rcParams
from matplotlib import pyplot as plt
import pandas as pd
import plotting_functions
from pathlib import Path
import dataframe_image as dfi
import argparse

parser = argparse.ArgumentParser(
    description="If the command is run with '-user' the raw data generated by the user "
    "(user/raw/results.pkl) is used to create the figures (saved in user/figures)."
    "If the command is run without '-user' the raw data we have already generated "
    "(paper/raw/results.pkl) is used to create the figures (saved in paper/figures)."
)
parser.add_argument('-user', action='store_true')
user = parser.parse_args().user

# running the terminal command 'python figures.py'
# or running this script from an IDE
# gives user = False

# running the terminal command 'python figures.py -user'
# gives user = True
# to do the same when running this script from an IDE simply uncomment the following line:
# user = True 

if user:
    user_or_paper = "user/"
else:
    user_or_paper = "paper/"

# create figures directory if it does not exist
Path(user_or_paper + "figures").mkdir(exist_ok=True, parents=True)

# load panda dataframe 
# either paper/raw/results.pkl or user/raw/results.pkl
# depending on '-user' or user = True as explained above
Path("user/raw").mkdir(exist_ok=True, parents=True)
results_df = pd.read_pickle(user_or_paper + "raw/results.pkl")

# Parameters for plots
fs = 32
rcParams.update({"font.size": fs})
rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
rc("text", usetex=True)

def mutate(List):
    """
    Mutate a list into a list of lists except for elements of type slice.
    input: List: list
    """
    for i in range(len(List)):
        if type(List[i]) not in [list, slice]:
            List[i] = [List[i]] 


def get_data(
    df,
    exp_number,
    d=slice(None),
    approx_type=slice(None),
    kernel_type=slice(None),
    l_minus_l_plus=slice(None),
):
    """
    Extract the relevant data points given the dataframe results.pkl for the different experiments.
    inputs: df: dataframe
            exp_number: integer between 1 and 12
            d: integer between 1 and 2
            approx_types: "permutation" or "wild bootstrap"
            kernel_type: "gaussian" or "laplace"
            l_minus_l_plus: tuple of the form (l_minus,l_plus)
    output: array consisting of relevant data points for the specific experiment
    """
    exps = [
        "1: uniform alternative",
        "2: mnist alternative",
        "3: uniform alternative",
        "4: mnist alternative",
        "5: uniform null",
        "6: mnist null",
        "7: uniform alternative",
        "8: mnist alternative",
        "9: uniform alternative",
        "10: mnist alternative",
        "11: uniform alternative",
        "12: mnist alternative",
    ]
    exp = exps[exp_number - 1]
    function_type = slice(None)
    if exp_number in [1, 2]:
        function_type = ["uniform", "increasing", "decreasing", "centred", "ost"]
        List = [
            exp,
            d,
            slice(None),
            slice(None),
            slice(None),
            approx_type,
            kernel_type,
            l_minus_l_plus,
            function_type,
        ]
        mutate(List)
        return df.loc[tuple(List)].unstack().to_numpy()
    elif exp_number in [3, 4]:
        function_type = ["median", "split", "split (doubled sample sizes)"]
        List = [
            exp,
            d,
            slice(None),
            slice(None),
            slice(None),
            approx_type,
            kernel_type,
            l_minus_l_plus,
            function_type,
        ]
        mutate(List)
        return df.loc[tuple(List)].unstack().to_numpy()
    elif exp_number in [7, 8]:
        List = [
            exp,
            d,
            slice(None),
            slice(None),
            slice(None),
            approx_type,
            kernel_type,
            l_minus_l_plus,
            function_type,
        ]
        mutate(List)
        return (
            df.loc[tuple(List)].swaplevel("n", "perturbation_or_Qi").unstack().to_numpy()
        )
    elif exp_number in [9, 10]:
        if l_minus_l_plus != slice(None):
            raise ValueError(
                "l_minus_l_plus should not be specified for exp_number = 9 or 10."
            )
        List = [
            exp,
            d,
            slice(None),
            slice(None),
            slice(None),
            approx_type,
            kernel_type,
            l_minus_l_plus,
            function_type,
        ]
        mutate(List)
        return (
            df.loc[tuple(List)]
            .swaplevel("perturbation_or_Qi", "l_minus_l_plus")
            .unstack()
            .to_numpy()
        )
    elif exp_number in [11, 12]:
        if approx_type != slice(None):
            raise ValueError(
                "approx_type should either not be specified for exp_number = 11 or 12."
            )
        List = [
            exp,
            d,
            slice(None),
            slice(None),
            slice(None),
            approx_type,
            kernel_type,
            l_minus_l_plus,
            function_type,
        ]
        mutate(List)
        return df.loc[tuple(List)].unstack().to_numpy()
    else:
        raise ValueError(
            "exp_number should be an integer between 1 and 12 (excluding 5 and 6)."
        )


def table_1(df):
    """
    Extract data relevant to constructing the table labelled Table 1 in our paper.
    input: df: dataframe
    output: dataframe entries corresponding to Table 1.
    """
    exps = [
        "1: uniform alternative",
        "2: mnist alternative",
        "3: uniform alternative",
        "4: mnist alternative",
        "5: uniform null",
        "6: mnist null",
        "7: uniform alternative",
        "8: uniform mnist",
        "9: uniform alternative",
        "10: uniform mnist",
        "11: uniform alternative",
        "12: uniform mnist",
    ]
    d = slice(None)
    approx_type = ["wild bootstrap", "permutation"]
    kernel_type = ["gaussian", "laplace"]
    l_minus_l_plus = slice(None)
    exp = [exps[5 - 1], exps[6 - 1]]
    function_type = [
        "uniform",
        "centred",
        "increasing",
        "decreasing",
        "median",
        "split",
        "ost",
    ]
    List = [
        exp,
        slice(None),
        d,
        slice(None),
        slice(None),
        slice(None),
        kernel_type,
        approx_type,
        slice(None),
        function_type,
    ]
    mutate(List)
    columns = [("power", function_type[i]) for i in range(len(function_type))]
    levels_order = [
        "experiment",
        "perturbation_or_Qi",
        "d",
        "repetitions",
        "m",
        "n",
        "kernel_type",
        "approx_type",
        "l_minus_l_plus",
        "function_type",
    ]
    return (
        df.reorder_levels(levels_order)
        .loc[tuple(List)]
        .reindex(index=["wild bootstrap", "permutation"], level=7)
        .unstack()[columns]
    )

approx_types = ["wild bootstrap", "permutation"]
kernel_types = ["gaussian", "laplace"]

# Table 1
dfi.export(table_1(results_df), user_or_paper + "figures/table_1.png")

# Figure 1
mult = 3
width = 433.62 / 72.27 * mult
height = width * (5 ** 0.5 - 1) / 2 * (2 / 3)
f, axs = plt.subplots(1, 2, figsize=(width, height - 1), sharey=True)
f.tight_layout()
f.subplots_adjust(wspace=0.1, hspace=0.45)

plotting_functions.plot_fig_1(f, axs)
axs[0].legend(
    fontsize=fs,
    ncol=2,
    handleheight=0.5,
    labelspacing=0.05,
    columnspacing=0.6,
    loc="lower center",
    bbox_to_anchor=(1, -0.47),
)
f.savefig(user_or_paper + "figures/figure_1.png", dpi=300, bbox_inches="tight")

# Figure 2
mult = 3
width = 433.62 / 72.27 * mult
height = width * (5 ** 0.5 - 1) / 2 * (2 / 3)
f, axs = plt.subplots(1, 3, figsize=(width, height - 1.2))
f.tight_layout()
f.subplots_adjust(wspace=0.15, hspace=0.25)

plotting_functions.plot_fig_2(f, axs)
f.savefig(user_or_paper + "figures/figure_2.png", dpi=300, bbox_inches="tight")

# Figure 3
mult = 3
width = 433.62 / 72.27 * mult
height = width * (5 ** 0.5 - 1) / 2 * (2 / 3)
f, axs = plt.subplots(2, 3, figsize=(width, height + 2), sharey=True, sharex=True)
f.tight_layout()
f.subplots_adjust(wspace=0.03, hspace=0.03)

L = [(-6, -2), (-4, 0), (-2, 2)]
d = 1
a = 0
for k, l in itertools.product(range(2), range(3)):
    idxy = (k, l)
    power_ms = get_data(
        results_df, 3, d=d, approx_type=approx_types[a], kernel_type=kernel_types[k]
    )
    power = get_data(
        results_df,
        1,
        d=d,
        approx_type=approx_types[a],
        kernel_type=kernel_types[k],
        l_minus_l_plus=L[l],
    )
    plotting_functions.plot_fig_3_4(*idxy, f, axs, power, power_ms, *L[l])

axs[1, 1].legend(
    fontsize=fs,
    ncol=4,
    handleheight=0.5,
    labelspacing=0.05,
    columnspacing=0.6,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.65),
)
f.savefig(user_or_paper + "figures/figure_3.png", dpi=300, bbox_inches="tight")

# Figure 4
mult = 3
width = 433.62 / 72.27 * mult
height = width * (5 ** 0.5 - 1) / 2 * (2 / 3)
f, axs = plt.subplots(2, 3, figsize=(width, height + 2), sharey=True, sharex=True)
f.tight_layout()
f.subplots_adjust(wspace=0.03, hspace=0.03)

L = [(-6, -2), (-4, 0), (-2, 2)]
d = 2
a = 0
for k, l in itertools.product(range(2), range(3)):
    idxy = (k, l)
    power_ms = get_data(
        results_df, 3, d=d, approx_type=approx_types[a], kernel_type=kernel_types[k]
    )
    power = get_data(
        results_df,
        1,
        d=d,
        approx_type=approx_types[a],
        kernel_type=kernel_types[k],
        l_minus_l_plus=L[l],
    )
    plotting_functions.plot_fig_3_4(*idxy, f, axs, power, power_ms, *L[l])

axs[1, 1].legend(
    fontsize=fs,
    ncol=4,
    handleheight=0.5,
    labelspacing=0.05,
    columnspacing=0.6,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.65),
)
f.savefig(user_or_paper + "figures/figure_4.png", dpi=300, bbox_inches="tight")

# Figure 5
mult = 3
width = 433.62 / 72.27 * mult
height = width * (5 ** 0.5 - 1) / 2 * (2 / 3)
f, axs = plt.subplots(2, 3, figsize=(width, height + 2.2), sharey=True, sharex=True)
f.tight_layout()
f.subplots_adjust(wspace=0.03, hspace=0.03)

L = [[(8, 12), (10, 14), (12, 16)], [(10, 14), (12, 16), (14, 18)]]
a = 0
for k, l in itertools.product(range(2), range(3)):
    idxy = (k, l)
    power_ms = get_data(
        results_df, 4, approx_type=approx_types[a], kernel_type=kernel_types[k]
    )
    power = get_data(
        results_df,
        2,
        approx_type=approx_types[a],
        kernel_type=kernel_types[k],
        l_minus_l_plus=L[k][l],
    )
    plotting_functions.plot_fig_5(*idxy, f, axs, power, power_ms)

axs[1, 1].legend(
    fontsize=fs,
    ncol=4,
    handleheight=0.5,
    labelspacing=0.05,
    columnspacing=0.6,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.65),
)
f.savefig(user_or_paper + "figures/figure_5.png", dpi=300, bbox_inches="tight")

# Figure 6
mult = 3
width = 433.62 / 72.27 * mult
height = width * (5 ** 0.5 - 1) / 2 * (2 / 3)
f, axs = plt.subplots(
    2, 3, figsize=(width, height + 2), sharey=True, sharex=True
)  # ,figsize=(5.326,3.562)) ,sharey=True)
f.tight_layout()
f.subplots_adjust(wspace=0.03, hspace=0.03)

a = 0
for k, e in itertools.product(range(2), range(3)):
    idxy = (k, e)
    if e == 2:
        power = get_data(
            results_df, 10, approx_type=approx_types[a], kernel_type=kernel_types[k]
        )
    else:
        power = get_data(
            results_df,
            9,
            d=e + 1,
            approx_type=approx_types[a],
            kernel_type=kernel_types[k],
        )
    plotting_functions.plot_fig_6(*idxy, f, axs, power)

axs[1, 1].legend(
    fontsize=fs,
    ncol=2,
    handleheight=0.5,
    labelspacing=0.05,
    columnspacing=0.6,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.57),
)
f.savefig(user_or_paper + "figures/figure_6.png", dpi=300, bbox_inches="tight")

# Figure 7
mult = 3
width = 433.62 / 72.27 * mult
height = width * (5 ** 0.5 - 1) / 2 * (2 / 3)
f, axs = plt.subplots(2, 3, figsize=(width, height + 2), sharey=True)
f.tight_layout()
f.subplots_adjust(wspace=0.03, hspace=0.03)

L = [(-4, 0), [(10, 14), (12, 16)]]
a = 0
for k, e in itertools.product(range(2), range(3)):
    idxy = (k, e)
    if e == 2:
        a = 0
        power_ms_wb = get_data(
            results_df, 4, approx_type=approx_types[a], kernel_type=kernel_types[k]
        )[:-1]
        power_wb = get_data(
            results_df,
            2,
            approx_type=approx_types[a],
            kernel_type=kernel_types[k],
            l_minus_l_plus=L[1][k],
        )
        a = 1
        power_ms_p = get_data(
            results_df, 4, approx_type=approx_types[a], kernel_type=kernel_types[k]
        )[:-1]
        power_p = get_data(
            results_df,
            2,
            approx_type=approx_types[a],
            kernel_type=kernel_types[k],
            l_minus_l_plus=L[1][k],
        )
        power_ms = power_ms_wb - power_ms_p
        power = power_wb - power_p
    else:
        a = 0
        power_ms_wb = get_data(
            results_df,
            3,
            d=e + 1,
            approx_type=approx_types[a],
            kernel_type=kernel_types[k],
        )[:-1]
        power_wb = get_data(
            results_df,
            1,
            d=e + 1,
            approx_type=approx_types[a],
            kernel_type=kernel_types[k],
            l_minus_l_plus=L[0],
        )
        a = 1
        power_ms_p = get_data(
            results_df,
            3,
            d=e + 1,
            approx_type=approx_types[a],
            kernel_type=kernel_types[k],
        )[:-1]
        power_p = get_data(
            results_df,
            1,
            d=e + 1,
            approx_type=approx_types[a],
            kernel_type=kernel_types[k],
            l_minus_l_plus=L[0],
        )
        power_ms = power_ms_wb - power_ms_p
        power = power_wb - power_p
    plotting_functions.plot_fig_7(*idxy, f, axs, power, power_ms)

axs[1, 1].legend(
    fontsize=fs,
    ncol=3,
    handleheight=0.5,
    labelspacing=0.05,
    columnspacing=0.6,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.67),
)
f.savefig(user_or_paper + "figures/figure_7.png", dpi=300, bbox_inches="tight")

# Figure 8
mult = 3
width = 433.62 / 72.27 * mult
height = width * (5 ** 0.5 - 1) / 2 * (2 / 3)
f, axs = plt.subplots(2, 3, figsize=(width, height + 2), sharey=True)
f.tight_layout()
f.subplots_adjust(wspace=0.03, hspace=0.03)

a = 1
for k, e in itertools.product(range(2), range(3)):
    idxy = (k, e)
    if e == 2:
        power = get_data(
            results_df, 8, approx_type=approx_types[a], kernel_type=kernel_types[k]
        )
    else:
        power = get_data(
            results_df,
            7,
            d=e + 1,
            approx_type=approx_types[a],
            kernel_type=kernel_types[k],
        )
    plotting_functions.plot_fig_8(*idxy, f, axs, power)

axs[1, 1].legend(
    fontsize=fs,
    ncol=2,
    handleheight=0.5,
    labelspacing=0.05,
    columnspacing=0.6,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.81),
)
f.savefig(user_or_paper + "figures/figure_8.png", dpi=300, bbox_inches="tight")

# Figure 9
mult = 3
width = 433.62 / 72.27 * mult
height = width * (5 ** 0.5 - 1) / 2 * (2 / 3)
f, axs = plt.subplots(2, 3, figsize=(width, height + 2), sharey=True)
f.tight_layout()
f.subplots_adjust(wspace=0.03, hspace=0.03)

L = [(-4, 0), [(10, 14), (12, 16)]]
a = 0
for k, e in itertools.product(range(2), range(3)):
    idxy = (k, e)
    if e == 2:
        power = get_data(results_df, 12, kernel_type=kernel_types[k])
        power_u = get_data(
            results_df,
            2,
            approx_type=approx_types[a],
            kernel_type=kernel_types[k],
            l_minus_l_plus=L[1][k],
        )[0]
        power_o = get_data(
            results_df,
            2,
            approx_type=approx_types[a],
            kernel_type=kernel_types[k],
            l_minus_l_plus=L[1][k],
        )[4]
    else:
        power = get_data(results_df, 11, d=e + 1, kernel_type=kernel_types[k])
        power_u = get_data(
            results_df,
            1,
            d=e + 1,
            approx_type=approx_types[a],
            kernel_type=kernel_types[k],
            l_minus_l_plus=L[0],
        )[0]
        power_o = get_data(
            results_df,
            1,
            d=e + 1,
            approx_type=approx_types[a],
            kernel_type=kernel_types[k],
            l_minus_l_plus=L[0],
        )[4]
    plotting_functions.plot_fig_9(*idxy, f, axs, power, power_u, power_o)

axs[1, 0].legend(
    fontsize=fs,
    ncol=1,
    handleheight=0.5,
    labelspacing=0.05,
    handlelength=1,
    columnspacing=0.2,
    loc="lower center",
    bbox_to_anchor=(0.5, -1.12),
)
axs[1, 1].legend(
    fontsize=fs,
    ncol=1,
    handleheight=0.5,
    labelspacing=0.05,
    handlelength=1,
    columnspacing=0.2,
    loc="lower center",
    bbox_to_anchor=(0.5, -1.12),
)
axs[1, 2].legend(
    fontsize=fs,
    ncol=1,
    handleheight=0.5,
    labelspacing=0.05,
    handlelength=1,
    columnspacing=0.2,
    loc="lower center",
    bbox_to_anchor=(0.5, -1.239),
)
f.savefig(user_or_paper + "figures/figure_9.png", dpi=300, bbox_inches="tight")

print(
    "Figures have been generated using " + user_or_paper + "raw/results.pkl"
    " and have been saved in " + user_or_paper + "figures."
)
