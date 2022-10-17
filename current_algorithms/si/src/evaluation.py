import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import seaborn as sns

# colors: http://colorbrewer2.org

# diagnostic for interventions
def iv_diagnostic(interventions, post_df, df_dict):
    # get number of units and interventions
    I = len(interventions)

    # columns to be dropped
    columns = ["intervention", "unit", "metric"]

    # initialize
    diag = pd.DataFrame()

    for df_name, df in df_dict.items():
        # initialize R2 values
        R2 = np.zeros(I)

        # loop through all interventions
        for i, iv in enumerate(interventions):
            unit_ids = post_df.loc[post_df.intervention == iv, "unit"].values
            baseline_error_sum = 0
            estimated_error_sum = 0
            for unit in unit_ids:
                y = (
                    post_df.loc[(post_df.unit == unit) & (post_df.intervention == iv)]
                    .drop(columns=columns)
                    .values
                )
                y_hat = (
                    df.loc[(df.unit == unit) & (df.intervention == iv)]
                    .drop(columns=columns)
                    .values
                )
                baseline_error_sum += ((y.mean(axis=1) - y) ** 2).sum()
                estimated_error_sum += ((y_hat - y) ** 2).sum()
            R2[i] = 1 - estimated_error_sum / baseline_error_sum
        diag.insert(0, "{} scores".format(df_name), R2)
    diag.insert(0, "intervention", interventions)
    return diag


# diagnostic for units
# def unit_diagnostic(units, post_df, df_dict):
#     # get number of units
#     N = len(units)

#     # columns to be dropped
#     columns = ['intervention', 'unit', 'metric']

#     # initialize
#     diag = pd.DataFrame()

#     for df_name, df in df_dict.items():
#         # initialize R2 values
#         R2 = np.zeros(N)

#         for i, unit in enumerate(units):
#             ivs = post_df.loc[post_df.unit==unit, 'intervention'].values
#             baseline_error_sum = 0
#             estimated_error_sum = 0
#             for iv in ivs:
#                 y = post_df.loc[(post_df.unit==unit) & (post_df.intervention==iv)].drop(columns=columns).values
#                 y_hat = df.loc[(df.unit==unit) & (df.intervention==iv)].drop(columns=columns).values
#                 baseline_error_sum += ((y.mean(axis=1) - y)**2).sum()
#                 estimated_error_sum += ((y_hat - y)**2).sum()
#             R2[i] = 1 - estimated_error_sum / baseline_error_sum
#         diag.insert(0, '{} scores'.format(df_name), R2)
#     diag.insert(0, 'unit', units)
#     return diag


def remove_pre_results(df_dict, pre_range):
    # remove pre-intervention results from the final output
    new_df_dict = {}
    for df_name, df in df_dict.items():
        indices = df.columns.get_loc(pre_range[0]), df.columns.get_loc(pre_range[1])
        cols = list(range(indices[0], indices[1] + 1))
        new_df = df.drop(df.columns[cols], axis=1)
        new_df_dict[df_name] = new_df

    return new_df_dict


def unit_diagnostic(units, post_df, df_dict):
    # get number of units
    N = len(units)
    # columns to be dropped
    columns = ["intervention", "unit", "metric"]
    post_columns = list(columns)
    if "donor" in post_df.columns:
        post_columns = columns + ["donor"]
    # initialize
    diag = pd.DataFrame()
    for df_name, df in df_dict.items():
        # initialize R2 values
        R2 = np.zeros(N)
        for i, unit in enumerate(units):
            ivs = post_df.loc[post_df.unit == unit, "intervention"].values
            baseline_error_sum = 0
            estimated_error_sum = 0
            for iv in ivs:
                y = (
                    post_df.loc[(post_df.unit == unit) & (post_df.intervention == iv)]
                    .drop(columns=post_columns)
                    .values
                )
                y_hat = (
                    df.loc[(df.unit == unit) & (df.intervention == iv)]
                    .drop(columns=columns)
                    .values
                )
                baseline_error_sum += ((y.mean(axis=1) - y) ** 2).sum()
                estimated_error_sum += ((y_hat - y) ** 2).sum()
            R2[i] = 1 - estimated_error_sum / baseline_error_sum
        diag.insert(0, "{} scores".format(df_name), R2)
    diag.insert(0, "unit", units)
    return diag


###### PLOTS #########


def boxplots(data_dict, boxColors, xlabel, ylabel, title, top=1, bottom=0, scale=0.1):
    # unpack data_dict
    data = list(data_dict.values())
    xticklabels = list(data_dict.keys())

    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.canvas.set_window_title(title)
    plt.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

    bp = plt.boxplot(data, notch=0, sym="+", vert=1, whis=1.5)
    plt.setp(bp["boxes"], color="black")
    plt.setp(bp["whiskers"], color="black")
    plt.setp(bp["fliers"], color="red", marker="+")

    # Add a horizontal grid to the plot, but make it very light in color
    # so we can use it for reading data values but not be distracting
    ax1.yaxis.grid(True, linestyle="-", which="major", color="lightgrey", alpha=0.5)

    # Hide these grid behind plot objects
    ax1.set_axisbelow(True)
    ax1.set_title(title)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)

    # Now fill the boxes with desired colors
    numBoxes = len(data)
    medians = list(range(numBoxes))
    for i in range(numBoxes):
        box = bp["boxes"][i]
        boxX = []
        boxY = []
        for j in range(5):
            boxX.append(box.get_xdata()[j])
            boxY.append(box.get_ydata()[j])
        boxCoords = list(zip(boxX, boxY))
        boxPolygon = Polygon(boxCoords, facecolor=boxColors[i])
        ax1.add_patch(boxPolygon)
        # Now draw the median lines back over what we just filled in
        med = bp["medians"][i]
        medianX = []
        medianY = []
        for j in range(2):
            medianX.append(med.get_xdata()[j])
            medianY.append(med.get_ydata()[j])
            plt.plot(medianX, medianY, "k")
            medians[i] = medianY[0]
        # Finally, overplot the sample averages, with horizontal alignment
        # in the center of each box
        plt.plot(
            [np.average(med.get_xdata())],
            [np.average(data[i])],
            color="w",
            marker="*",
            markeredgecolor="k",
        )

    # Set the axes ranges and axes labels
    ax1.set_xlim(0.5, numBoxes + 0.5)
    ax1.set_ylim(bottom, top)
    xtickNames = plt.setp(ax1, xticklabels=xticklabels)
    plt.setp(xtickNames, rotation=45)
    pos = np.arange(numBoxes) + 1
    upperLabels = [str(np.round(s, 2)) for s in medians]
    weights = ["bold", "semibold"]
    for tick, label in zip(range(numBoxes), ax1.get_xticklabels()):
        ax1.text(
            pos[tick],
            top - (top * scale),
            upperLabels[tick],
            horizontalalignment="center",
            color=boxColors[tick],
        )  # weight=weights[k]

    # Finally, add a basic legend
    plt.figtext(
        0.80,
        0.015,
        "*",
        color="white",
        backgroundcolor="silver",
        weight="roman",
        size="medium",
    )
    plt.figtext(
        0.815, 0.013, " Mean Value", color="black", weight="roman", size="x-small"
    )
    plt.show()


def heatmap(df, size1, size2, title):
    plt.figure(figsize=(size1, size2))
    plt.title(title)
    sns.heatmap(df, annot=True, cmap="RdYlGn")
    plt.yticks(rotation=0)
    b, t = plt.ylim()
    b += 0.5
    t -= 0.5
    plt.ylim(b, t)
    plt.show()
