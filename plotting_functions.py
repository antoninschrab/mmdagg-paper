import numpy as np
import itertools
from weights import create_weights
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sampling import G, f_theta, f0

# https://gist.github.com/thriveth/8560036
CB_color_cycle = [
    "#377eb8",
    "#ff7f00",
    "#4daf4a",
    "#f781bf",
    "#a65628",
    "#984ea3",
    "#999999",
    "#e41a1c",
    "#dede00",
]

# Parameters for plots
linewidth = 2.5
markersize = 8
fs = 32


def plot_fig_1(f, axs):
    """
    Plot the figure labelled Figure 1 in our paper given the data.
    """
    idx = 0
    N = 5
    x_values = [i + 1 for i in range(N)]
    axs[idx].plot(
        x_values,
        create_weights(N, "uniform"),
        CB_color_cycle[1],
        marker="o",
        label=r"\texttt{uniform}",
        linewidth=linewidth,
        markersize=markersize,
    )
    axs[idx].plot(
        x_values,
        create_weights(N, "centred"),
        CB_color_cycle[7],
        marker="o",
        label=r"\texttt{centred}",
        linewidth=linewidth,
        markersize=markersize,
    )
    axs[idx].plot(
        x_values,
        create_weights(N, "increasing"),
        CB_color_cycle[2],
        marker="o",
        label=r"\texttt{increasing}",
        linewidth=linewidth,
        markersize=markersize,
    )
    axs[idx].plot(
        x_values,
        create_weights(N, "decreasing"),
        CB_color_cycle[0],
        marker="o",
        label=r"\texttt{decreasing}",
        linewidth=linewidth,
        markersize=markersize,
    )
    axs[idx].tick_params(
        axis="x", which="both", bottom=False, top=False, labelbottom=False
    )
    axs[idx].set_ylabel("Weights", labelpad=10)
    axs[idx].set_title(str(N) + " bandwidths", pad=8, fontsize=fs)
    axs[idx].set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    axs[idx].set_ylim([-0.025, 0.525])
    idx = 1
    N = 6
    x_values = [i + 1 for i in range(N)]
    axs[idx].plot(
        x_values,
        create_weights(N, "uniform"),
        CB_color_cycle[1],
        marker="o",
        label=r"\texttt{MMDAgg uniform}",
        linewidth=linewidth,
        markersize=markersize,
    )
    axs[idx].plot(
        x_values,
        create_weights(N, "centred"),
        CB_color_cycle[7],
        marker="o",
        label=r"\texttt{MMDAgg centred}",
        linewidth=linewidth,
        markersize=markersize,
    )
    axs[idx].plot(
        x_values,
        create_weights(N, "increasing"),
        CB_color_cycle[2],
        marker="o",
        label=r"\texttt{MMDAgg increasing}",
        linewidth=linewidth,
        markersize=markersize,
    )
    axs[idx].plot(
        x_values,
        create_weights(N, "decreasing"),
        CB_color_cycle[0],
        marker="o",
        label=r"\texttt{MMDAgg decreasing}",
        linewidth=linewidth,
        markersize=markersize,
    )
    axs[idx].tick_params(
        axis="x", which="both", bottom=False, top=False, labelbottom=False
    )
    axs[idx].set_title(str(N) + " bandwidths", pad=8, fontsize=fs)
    f.text(0.5, 0.065, "Bandwidths sorted in increasing order", ha="center")


def plot_fig_2(f, axs):
    """
    Plot the figure labelled Figure 2 in our paper given the data.
    """
    idx = 0
    start = -1
    stop = 0
    n_points = 300
    xplot = np.linspace(start, stop, n_points)
    Gplot = [G(x) for x in xplot]
    axs[idx].plot(xplot, Gplot, "k")
    axs[idx].set_yticks([-0.4, 0, 0.4])
    axs[idx].set_ylim(-0.4, 0.4)
    axs[idx].set_xlim(-1, 0)
    axs[idx].set_xticks([-1, -0.5, 0])
    axs[idx].set_yticklabels(["", 0, ""])
    axs[idx].set_yticks([-0.37, 0.37], minor=True)
    axs[idx].set_yticklabels(["$-0.4$", "$0.4$"], minor=True)
    axs[idx].set_xticklabels(["", "$-0.5$", ""])
    axs[idx].set_xticks([-0.97, -0.06], minor=True)
    axs[idx].set_xticklabels(["$-1.0$", "$0.0$"], minor=True)
    axs[idx].set_title("(a)", pad=12, fontsize=fs)
    for line in axs[idx].yaxis.get_minorticklines():
        line.set_visible(False)
    for line in axs[idx].xaxis.get_minorticklines():
        line.set_visible(False)
    idx = 1
    s = 1
    d = 1
    perturbation_multiplier = 2.7
    start = 0
    stop = 1
    n_points = 300
    xplot = np.linspace(start, stop, n_points)
    f0plot = [f0(x) for x in xplot]
    zeroplot = [0 for x in xplot]
    axs[idx].plot(xplot, f0plot, "k")
    colors = ["blue", "red", "purple", "orange"]
    for p in range(1, 5):
        fplot = [f_theta(x, p, s, perturbation_multiplier, p + 5) for x in xplot]
        axs[idx].plot(xplot, fplot, colors[p - 1])
    axs[idx].set_xticks([0, 0.5, 1])
    axs[idx].set_yticks([0, 1, 2])
    axs[idx].set_yticklabels(["", 1, ""])
    axs[idx].set_yticks([0.06, 1.94], minor=True)
    axs[idx].set_yticklabels(["$0$", "$2$"], minor=True)
    axs[idx].set_xticklabels(["", "$0.5$", ""])
    axs[idx].set_xticks([0.06, 0.94], minor=True)
    axs[idx].set_xticklabels(["$0.0$", "$1.0$"], minor=True)
    axs[idx].set_ylim(0, 2)
    axs[idx].set_xlim(0, 1)
    axs[idx].set_title("(b)", pad=12, fontsize=fs)
    for line in axs[idx].yaxis.get_minorticklines():
        line.set_visible(False)
    for line in axs[idx].xaxis.get_minorticklines():
        line.set_visible(False)
    idx = 2
    start = 0
    stop = 1
    n_points = 100
    p = 2
    s = 1
    d = 2
    perturbation_multiplier = 7.3
    x = np.linspace(start, stop, n_points)
    y = np.linspace(start, stop, n_points)
    z = np.array(
        [
            f_theta(
                np.concatenate((np.atleast_1d(j), np.atleast_1d(i))),
                p,
                s,
                perturbation_multiplier,
                5,
            )
            for j in y
            for i in x
        ]
    )
    Z = z.reshape(n_points, n_points)
    im1 = axs[idx].imshow(
        Z, origin="lower", interpolation="bilinear", extent=[0, 1, 0, 1], cmap="bwr"
    )
    axs[idx].set_title("(c)", pad=12, fontsize=fs)
    divider = make_axes_locatable(axs[idx])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = f.colorbar(im1, cax=cax, orientation="vertical")
    cbar.set_ticks([0.6, 1, 1.4])
    cbar.set_ticklabels([0.6, 1, 1.4])
    axs[idx].set_yticks([0, 0.5, 1])
    axs[idx].set_xticks([0, 0.5, 1])
    axs[idx].set_yticklabels(["", 0.5, ""])
    axs[idx].set_yticks([0.03, 0.97], minor=True)
    axs[idx].set_yticklabels(["$0.0$", "$1.0$"], minor=True)
    axs[idx].set_xticklabels(["", "$0.5$", ""])
    axs[idx].set_xticks([0.06, 0.94], minor=True)
    axs[idx].set_xticklabels(["$0.0$", "$1.0$"], minor=True)

    for line in axs[idx].yaxis.get_minorticklines():
        line.set_visible(False)
    for line in axs[idx].xaxis.get_minorticklines():
        line.set_visible(False)


def plot_fig_3_4(idx, idy, f, axs, power, power_ms, l_minus, l_plus):
    """
    Plot the figures labelled Figures 3 and 4 in our paper given the data.
    """
    x_values = [i for i in range(1, 1 + len(power[0]))]
    axs[idx, idy].plot(
        x_values,
        power[0],
        CB_color_cycle[1],
        marker="o",
        label=r"\texttt{MMDAgg uniform}",
        linewidth=linewidth,
        markersize=markersize,
    )
    axs[idx, idy].plot(
        x_values,
        power[3],
        CB_color_cycle[7],
        marker="o",
        label=r"\texttt{MMDAgg centred}",
        linewidth=linewidth,
        markersize=markersize,
    )
    axs[idx, idy].plot(
        x_values,
        power[1],
        CB_color_cycle[2],
        marker="o",
        label=r"\texttt{MMDAgg increasing}",
        linewidth=linewidth,
        markersize=markersize,
    )
    axs[idx, idy].plot(
        x_values,
        power[2],
        CB_color_cycle[0],
        marker="o",
        label=r"\texttt{MMDAgg decreasing}",
        linewidth=linewidth,
        markersize=markersize,
    )
    axs[idx, idy].plot(
        x_values,
        power_ms[0],
        CB_color_cycle[6],
        marker="s",
        linestyle="--",
        label=r"\texttt{median}",
        linewidth=linewidth,
        markersize=markersize,
    )
    axs[idx, idy].plot(
        x_values,
        power[4],
        CB_color_cycle[5],
        marker="^",
        linestyle="--",
        label=r"\texttt{ost}",
        linewidth=linewidth,
        markersize=markersize,
    )
    axs[idx, idy].plot(
        x_values,
        power_ms[1],
        CB_color_cycle[4],
        marker="s",
        linestyle="--",
        label=r"\texttt{split}",
        linewidth=linewidth,
        markersize=markersize,
    )
    axs[idx, idy].plot(
        x_values,
        power_ms[2],
        CB_color_cycle[3],
        marker="v",
        linestyle=":",
        label=r"\texttt{oracle}",
        linewidth=linewidth,
        markersize=markersize,
    )
    axs[idx, idy].set_xticks(x_values)
    if idx == 0:
        axs[idx, idy].set_title(
            "$\Lambda(" + str(l_minus) + "," + str(l_plus) + ")$", fontsize=fs, pad=10
        )
    axs[idx, idy].set_ylim(-0.05, 1.05)
    if (idx, idy) == (1, 1):
        axs[idx, idy].set_xlabel("Number of perturbations", fontsize=fs)
    if (idx, idy) == (0, 0):
        axs[idx, idy].set_ylabel("Power \n Gaussian kernel", fontsize=fs, labelpad=10)
    if (idx, idy) == (1, 0):
        axs[idx, idy].set_ylabel("Power \n Laplace kernel", fontsize=fs, labelpad=10)
    axs[idx, idy].set_yticks([0, 0.25, 0.5, 0.75, 1])


def plot_fig_5(idx, idy, f, axs, power, power_ms):
    """
    Plot the figure labelled Figure 5 in our paper given the data.
    """
    x_values = ["$Q_1$", "$Q_2$", "$Q_3$", "$Q_4$", "$Q_5$"]
    axs[idx, idy].plot(
        x_values,
        power[0],
        CB_color_cycle[1],
        marker="o",
        label=r"\texttt{MMDAgg uniform}",
        linewidth=linewidth,
        markersize=markersize,
    )
    axs[idx, idy].plot(
        x_values,
        power[3],
        CB_color_cycle[7],
        marker="o",
        label=r"\texttt{MMDAgg centred}",
        linewidth=linewidth,
        markersize=markersize,
    )
    axs[idx, idy].plot(
        x_values,
        power[1],
        CB_color_cycle[2],
        marker="o",
        label=r"\texttt{MMDAgg increasing}",
        linewidth=linewidth,
        markersize=markersize,
    )
    axs[idx, idy].plot(
        x_values,
        power[2],
        CB_color_cycle[0],
        marker="o",
        label=r"\texttt{MMDAgg decreasing}",
        linewidth=linewidth,
        markersize=markersize,
    )
    axs[idx, idy].plot(
        x_values,
        power_ms[0],
        CB_color_cycle[6],
        marker="s",
        linestyle="--",
        label=r"\texttt{median}",
        linewidth=linewidth,
        markersize=markersize,
    )
    axs[idx, idy].plot(
        x_values,
        power[4],
        CB_color_cycle[5],
        marker="^",
        linestyle="--",
        label=r"\texttt{ost}",
        linewidth=linewidth,
        markersize=markersize,
    )
    axs[idx, idy].plot(
        x_values,
        power_ms[1],
        CB_color_cycle[4],
        marker="s",
        linestyle="--",
        label=r"\texttt{split}",
        linewidth=linewidth,
        markersize=markersize,
    )
    axs[idx, idy].plot(
        x_values,
        power_ms[2],
        CB_color_cycle[3],
        marker="v",
        linestyle=":",
        label=r"\texttt{oracle}",
        linewidth=linewidth,
        markersize=markersize,
    )
    axs[idx, idy].set_xticks(x_values)
    axs[idx, idy].xaxis.set_tick_params(pad=7)
    axs[idx, idy].set_yticks([0, 0.25, 0.5, 0.75, 1])
    axs[idx, idy].set_ylim(-0.05, 1.05)
    if (idx, idy) == (1, 1):
        axs[idx, idy].set_xlabel("Choice of alternative", fontsize=fs)
    if (idx, idy) == (0, 0):
        axs[idx, idy].set_ylabel("Power \n Gaussian kernel", fontsize=fs)
        axs[idx, idy].set_title(
            "$\Lambda(8,12)$, $\Lambda(10,14)$", fontsize=fs, pad=10
        )
    if (idx, idy) == (0, 1):
        axs[idx, idy].set_title(
            "$\Lambda(10,14)$, $\Lambda(12,16)$", fontsize=fs, pad=10
        )
    if (idx, idy) == (0, 2):
        axs[idx, idy].set_title(
            "$\Lambda(12,16)$, $\Lambda(14,18)$", fontsize=fs, pad=10
        )
    if (idx, idy) == (1, 0):
        axs[idx, idy].set_ylabel("Power \n Laplace kernel", fontsize=fs, labelpad=10)


def plot_fig_6(idx, idy, f, axs, power):
    """
    Plot the figure labelled Figure 6 in our paper given the data.
    """
    x_values = [3, 5, 7, 9, 11, 13, 15]
    axs[idx, idy].plot(
        x_values,
        power[0][1:],
        CB_color_cycle[1],
        marker="o",
        label=r"\texttt{MMDAgg uniform}",
        linewidth=linewidth,
        markersize=markersize,
    )
    axs[idx, idy].plot(
        x_values,
        power[1][1:],
        CB_color_cycle[7],
        marker="o",
        label=r"\texttt{MMDAgg centred}",
        linewidth=linewidth,
        markersize=markersize,
    )
    axs[idx, idy].set_yticks([0, 0.25, 0.5, 0.75, 1])
    axs[idx, idy].set_ylim([-0.05, 1.05])
    axs[idx, idy].set_xticks(x_values)
    if (idx, idy) == (0, 0):
        axs[idx, idy].set_ylabel("Power \n Gaussian kernel", fontsize=fs, labelpad=10)
        axs[idx, idy].set_title(
            "$d=1$, 3 perturbations\n$m=n=500$\n $\Lambda(-2-i,-2+i)$",
            fontsize=fs,
            pad=10,
        )
    if (idx, idy) == (1, 0):
        axs[idx, idy].set_ylabel("Power \n Laplace kernel", fontsize=fs, labelpad=10)
    if (idx, idy) == (0, 1):
        axs[idx, idy].set_title(
            "$d=2$, 2 perturbations\n$m=n=2\,000$\n $\Lambda(-2-i,-2+i)$",
            fontsize=fs,
            pad=10,
        )
    if (idx, idy) == (1, 1):
        axs[idx, idy].set_xlabel(
            "Number of bandwidths in the collection (corresponding to $i=1,\dots,7$)",
            labelpad=10,
            fontsize=fs,
        )
    if (idx, idy) == (0, 2):
        axs[idx, idy].set_title(
            "MNIST, $Q_3$, $m=n=500$\n $\Lambda(12-i,12+i)$\n $\Lambda(14-i,14+i)$",
            fontsize=fs,
            pad=10,
        )


def plot_fig_7(idx, idy, f, axs, power, power_ms):
    """
    Plot the figure labelled Figure 7 in our paper given the data.
    """
    if idy == 0:
        x_values = [i + 1 for i in range(4)]
    if idy == 1:
        x_values = [i + 1 for i in range(3)]
    if idy == 2:
        x_values = ["$Q_1$", "$Q_2$", "$Q_3$", "$Q_4$", "$Q_5$"]
    axs[idx, idy].plot(
        x_values,
        power[0],
        CB_color_cycle[1],
        marker="o",
        label=r"\texttt{MMDAgg uniform}",
        linewidth=linewidth,
        markersize=markersize,
    )
    axs[idx, idy].plot(
        x_values,
        power[3],
        CB_color_cycle[7],
        marker="o",
        label=r"\texttt{MMDAgg centred}",
        linewidth=linewidth,
        markersize=markersize,
    )
    axs[idx, idy].plot(
        x_values,
        power[1],
        CB_color_cycle[2],
        marker="o",
        label=r"\texttt{MMDAgg increasing}",
        linewidth=linewidth,
        markersize=markersize,
    )
    axs[idx, idy].plot(
        x_values,
        power[2],
        CB_color_cycle[0],
        marker="o",
        label=r"\texttt{MMDAgg decreasing}",
        linewidth=linewidth,
        markersize=markersize,
    )
    axs[idx, idy].plot(
        x_values,
        power_ms[0],
        CB_color_cycle[6],
        marker="s",
        linestyle="--",
        label=r"\texttt{median}",
        linewidth=linewidth,
        markersize=markersize,
    )
    axs[idx, idy].plot(
        x_values,
        power_ms[1],
        CB_color_cycle[4],
        marker="s",
        linestyle="--",
        label=r"\texttt{split}",
        linewidth=linewidth,
        markersize=markersize,
    )
    axs[idx, idy].set_yticks([-0.02, 0, 0.02])
    axs[idx, idy].set_ylim([-0.03, 0.03])
    axs[idx, idy].set_xticks(x_values)
    if (idx, idy) == (0, 0):
        axs[idx, idy].set_ylabel("Difference in power \n Gaussian kernel", fontsize=fs)
        axs[idx, idy].set_title(
            "$d=1$, $\Lambda(-4,0)$\n$m=n=500$", fontsize=fs, pad=10
        )
        axs[idx, idy].set_xticklabels(["", "", "", ""])
    if (idx, idy) == (1, 0):
        axs[idx, idy].set_ylabel("Difference in power \n Laplace kernel", fontsize=fs)
        axs[idx, idy].set_xlabel("Number of perturbations", fontsize=fs)
    if (idx, idy) == (0, 1):
        axs[idx, idy].set_title(
            "$d=2$, $\Lambda(-4,0)$\n$m=n=2\,000$", fontsize=fs, pad=10
        )
        axs[idx, idy].set_xticklabels(["", "", ""])
    if (idx, idy) == (1, 1):
        axs[idx, idy].set_xlabel("Number of perturbations", fontsize=fs)
    if (idx, idy) == (0, 2):
        axs[idx, idy].set_title(
            "MNIST, $\Lambda(10,14)$, $\Lambda(12,16)$\n$m=n=500$", fontsize=fs, pad=10
        )
        axs[idx, idy].set_xticklabels(["", "", "", "", ""])
    if (idx, idy) == (1, 2):
        axs[idx, idy].set_xlabel("Choice of alternative", fontsize=fs)
        axs[idx, idy].xaxis.set_tick_params(pad=7)


def plot_fig_8(idx, idy, f, axs, power):
    """
    Plot the figure labelled Figure 8 in our paper given the data.
    """
    sample_sizes = ["$1\,000$", "$2\,000$", "$3\,000$", "$4\,000$", "$5\,000$"]
    if idy == 2:
        sample_sizes = ["200", "400", "600", "800", "$1\,000$"]
    axs[idx, idy].plot(
        sample_sizes,
        power[0],
        CB_color_cycle[1],
        marker="o",
        label=r"\texttt{MMDAgg uniform}",
        linewidth=linewidth,
        markersize=markersize,
    )
    axs[idx, idy].plot(
        sample_sizes,
        power[3],
        CB_color_cycle[7],
        marker="o",
        label=r"\texttt{MMDAgg centred}",
        linewidth=linewidth,
        markersize=markersize,
    )
    axs[idx, idy].plot(
        sample_sizes,
        power[1],
        CB_color_cycle[2],
        marker="o",
        label=r"\texttt{MMDAgg increasing}",
        linewidth=linewidth,
        markersize=markersize,
    )
    axs[idx, idy].plot(
        sample_sizes,
        power[2],
        CB_color_cycle[0],
        marker="o",
        label=r"\texttt{MMDAgg decreasing}",
        linewidth=linewidth,
        markersize=markersize,
    )
    axs[idx, idy].set_yticks([0, 0.25, 0.5, 0.75, 1])
    axs[idx, idy].set_ylim([-0.05, 1.05])
    axs[idx, idy].set_xticks(sample_sizes)
    axs[idx, idy].tick_params(axis="x", labelrotation=90)
    if (idx, idy) == (0, 0):
        axs[idx, idy].set_ylabel("Power \n Gaussian kernel", fontsize=fs, labelpad=10)
        axs[idx, idy].set_title(
            "$d=1$, 3 perturbations\n$m=100$, $\Lambda(-4,0)$", fontsize=fs, pad=10
        )
        axs[idx, idy].set_xticklabels(["", "", "", "", ""])
    if (idx, idy) == (1, 0):
        axs[idx, idy].set_ylabel("Power \n Laplace kernel", fontsize=fs, labelpad=10)
    if (idx, idy) == (0, 1):
        axs[idx, idy].set_title(
            "$d=2$, 2 perturbations\n$m=250$, $\Lambda(-4,0)$", fontsize=fs, pad=10
        )
        axs[idx, idy].set_xticklabels(["", "", "", "", ""])
    if (idx, idy) == (1, 1):
        axs[idx, idy].set_xlabel("Sample size $n$", labelpad=12, fontsize=fs)
    if (idx, idy) == (0, 2):
        axs[idx, idy].set_title(
            "MNIST, $Q_3$, $m=100$\n$\Lambda(10,14)$, $\Lambda(12,16)$",
            fontsize=fs,
            pad=10,
        )
        axs[idx, idy].set_xticklabels(["", "", "", "", ""])


def plot_fig_9(idx, idy, f, axs, power, power_u, power_o):
    """
    Plot the figure labelled Figure 9 in our paper given the data.
    """
    # https://github.com/matplotlib/matplotlib/issues/9460
    colors = ["#1845fb", "#578dff", "#86c8dd", "#adad7d", "#656364"]
    mn = ["500", "$2\,000$", "500"]
    if idy == 0:
        x_values = [i + 1 for i in range(4)]
        sample_sizes = [
            "$25\,000$",
            "$50\,000$",
            "$75\,000$",
            "$100\,000$",
        ]  # = [int(i * 10**5) for i in [0.25, 0.5, 0.75, 1]]
    if idy == 1:
        x_values = [i + 1 for i in range(3)]
        sample_sizes = [
            "$250\,000$",
            "$500\,000$",
            "$750\,000$",
            "$1\,000\,000$",
        ]  # = [int(i * 10**5) for i in [2.5, 5, 7.5, 10]]
    if idy == 2:
        x_values = ["$Q_1$", "$Q_2$", "$Q_3$", "$Q_4$", "$Q_5$"]
        sample_sizes = [
            "$50\,000$",
            "$100\,000$",
            "$150\,000$",
            "$200\,000$",
            "$250\,000$",
        ]  # = [int(i * 10**5) for i in [0.5, 1, 1.5, 2, 2.5]]
    axs[idx, idy].plot(
        x_values,
        power_u,
        CB_color_cycle[1],
        marker="o",
        label=r"\texttt{MMDAgg uniform} " + str(mn[idy]),
        linewidth=linewidth,
        markersize=markersize,
    )
    axs[idx, idy].plot(
        x_values,
        power_o,
        CB_color_cycle[5],
        marker="^",
        linestyle="--",
        label=r"\texttt{ost} " + str(mn[idy]),
        linewidth=linewidth,
        markersize=markersize,
    )
    axs[idx, idy].plot(
        x_values,
        power[0],
        colors[0],
        marker="^",
        linestyle="--",
        label=r"\texttt{ost} " + str(sample_sizes[0]),
        linewidth=linewidth,
        markersize=markersize,
    )
    axs[idx, idy].plot(
        x_values,
        power[1],
        colors[1],
        marker="^",
        linestyle="--",
        label=r"\texttt{ost} " + str(sample_sizes[1]),
        linewidth=linewidth,
        markersize=markersize,
    )
    axs[idx, idy].plot(
        x_values,
        power[2],
        colors[2],
        marker="^",
        linestyle="--",
        label=r"\texttt{ost} " + str(sample_sizes[2]),
        linewidth=linewidth,
        markersize=markersize,
    )
    axs[idx, idy].plot(
        x_values,
        power[3],
        colors[3],
        marker="^",
        linestyle="--",
        label=r"\texttt{ost} " + str(sample_sizes[3]),
        linewidth=linewidth,
        markersize=markersize,
    )
    if len(sample_sizes) == 5:
        axs[idx, idy].plot(
            x_values,
            power[4],
            colors[4],
            marker="^",
            linestyle="--",
            label=r"\texttt{ost} " + str(sample_sizes[4]),
            linewidth=linewidth,
            markersize=markersize,
        )
    axs[idx, idy].set_yticks([0, 0.25, 0.5, 0.75, 1])
    axs[idx, idy].set_ylim([-0.05, 1.05])
    axs[idx, idy].set_xticks(x_values)
    if (idx, idy) == (0, 0):
        axs[idx, idy].set_ylabel("Power \n Gaussian kernel", fontsize=fs, labelpad=10)
        axs[idx, idy].set_title("$d=1$, $\Lambda(-4,0)$", fontsize=fs, pad=10)
        axs[idx, idy].set_xticklabels(["", "", "", ""])
    if (idx, idy) == (1, 0):
        axs[idx, idy].set_ylabel("Power \n Laplace kernel", fontsize=fs, labelpad=10)
        axs[idx, idy].set_xlabel("Number of perturbations", fontsize=fs)
    if (idx, idy) == (0, 1):
        axs[idx, idy].set_title("$d=2$, $\Lambda(-4,0)$", fontsize=fs, pad=10)
        axs[idx, idy].set_xticklabels(["", "", ""])
    if (idx, idy) == (1, 1):
        axs[idx, idy].set_xlabel("Number of perturbations", fontsize=fs)
    if (idx, idy) == (0, 2):
        axs[idx, idy].set_title(
            "MNIST, $\Lambda(10,14)$, $\Lambda(12,16)$", fontsize=fs, pad=10
        )
        axs[idx, idy].set_xticklabels(["", "", "", "", ""])
    if (idx, idy) == (1, 2):
        axs[idx, idy].set_xlabel("Choice of alternative", fontsize=fs)
        axs[idx, idy].xaxis.set_tick_params(pad=7)
