import logging
logging.basicConfig(level=logging.INFO)

import numpy as onp
import os

import plotly.graph_objects as go

import pandas as pd
from jax import numpy as np
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots

import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "generalIVmodel/"))

import skbio.stats.composition as cmp


colours=["#e41a1c",
"#377eb8",
"#4daf4a",
"#984ea3",
"#ff7f00",
"#ffff33",
"#a65628",
"#f781bf"]

plotly_colors=['#636EFA',
     '#EF553B',
     '#00CC96',
     '#AB63FA',
     '#FFA15A',
     '#19D3F3',
     '#FF6692',
     '#B6E880',
     '#FF97FF',
     '#FECB52',
               '#636EFA',
               '#EF553B',
               '#00CC96',
               '#AB63FA',
               '#FFA15A',
               '#19D3F3',
               '#FF6692',
               '#B6E880',
               '#FF97FF',
               '#FECB52'
               ]

col_rgba = ['rgba(31, 119, 180, 0.2)',
       'rgba(255, 127, 14, 0.2)',
        'rgba(44, 160, 44, 0.2)',
       'rgba(214, 39, 40, 0.2)',
    'rgba(148, 103, 189, 0.2)',
       'rgba(140, 86, 75, 0.2)',
        'rgba(227, 119, 194, 0.2)',
       'rgba(127, 127, 127, 0.2)',
        'rgba(188, 189, 34, 0.2)',
       'rgba(23, 190, 207, 0.2)']


col_sat_rgba = ['rgba(31, 119, 180, 1)',
       'rgba(255, 127, 14, 1)',
        'rgba(44, 160, 44, 1)',
       'rgba(214, 39, 40, 1)',
    'rgba(148, 103, 189, 1)',
       'rgba(140, 86, 75, 1)',
        'rgba(227, 119, 194, 1)',
       'rgba(127, 127, 127, 1)',
        'rgba(188, 189, 34, 1)',
       'rgba(23, 190, 207, 1)']


img_path = "/Users/elisabeth.ailer/Projekte/P1_Microbiom/Code/fig"

def update_layout(fig):
    layout = go.Layout(
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="serif", size=32),
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showline=True, linewidth=2, linecolor="black"),  # gridcolor="grey"),
        yaxis=dict(showline=True, linewidth=2, linecolor="black")  # , gridcolor="grey")
    )
    fig.update_traces(marker_line_width=2, marker_size=5, line_width=4)
    fig.update_layout(layout)

    return fig

def update_layout_px(fig):
    layout = go.Layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        ternary=dict(bgcolor='rgba(0,0,0,0)',
                     aaxis=dict(gridcolor="grey", linecolor="grey"),
                     baxis=dict(gridcolor="grey", linecolor="grey"),
                     caxis=dict(gridcolor="grey", linecolor="grey"),),
        font=dict(family="serif", size=26),
        xaxis=dict(showline=True, linewidth=2, linecolor="black"),  # gridcolor="grey"),
        yaxis=dict(showline=True, linewidth=2, linecolor="black")  # , gridcolor="grey")
    )
    #fig.update_traces(marker_line_width=2, marker_size=5, line_width=4)
    fig.update_layout(layout)

    return fig


def plot_ilr_X_vs_Y(X_sim_ilr, X_star_ilr, Y_sim, Y_star):
    """plot ilr X vs Y for the different components"""

    fig = make_subplots(rows=1, cols=min(X_sim_ilr.shape[1], 5),
                        #horizontal_spacing=0.3
                        )
    fig = update_layout(fig)
    nter = 0

    for i in range(X_sim_ilr.shape[1]):
        if nter < 5:
            fig.add_trace(
                go.Scatter(x=X_sim_ilr[:, i], y=Y_sim, mode="markers", name="Simulated data",
                           opacity=0.75, marker_color=plotly_colors[0],
                           showlegend=(False if i > 0 else True)),
                row=1, col=i + 1
            )

            fig.add_trace(
                go.Scatter(x=X_star_ilr[:, i], y=Y_star, mode="markers", name="True Effect",
                           opacity = .75, marker_color=plotly_colors[4], showlegend=(False if i > 0 else True)),
                row=1, col=i + 1
            )
            nter+=1

    fig.update_layout(xaxis2=dict(showline=True, linewidth=2, linecolor="black"))
    fig.update_layout(yaxis2=dict(showline=True, linewidth=2, linecolor="black"))
    fig.update_xaxes(title="ilr(X)")
    fig.update_layout(yaxis1=dict(title="Y"))

    return fig

def get_Y_trendline(X, Y):
    p = onp.poly1d(onp.polyfit(onp.array(X).squeeze(), onp.array(Y).squeeze(), 1))
    Yhat = p(X)
    return Yhat


# TODO : check if we still need this function
def plot_results(X_sim, Y_sim, X_star, Y_star, Yhat, title="", component_plot=5, path=""):
    X_star_ilr = cmp.ilr(X_star)
    X_sim_ilr = cmp.ilr(X_sim)

    n, p = X_sim.shape
    plt.figure(figsize=(20, 10))
    plt.suptitle(title)
    df = pd.DataFrame(X_sim_ilr)
    idx = df.mean().sort_values(ascending=False).index
    len_idx = np.min(np.array([len(idx), component_plot]))
    n_iter = 1

    for n_iter, i in enumerate(idx):
        if n_iter >= len_idx:
            break
        else:
            plt.subplot(1, len_idx, n_iter+1)
            plt.title(str(n_iter) + " Component")
            plt.scatter(X_sim_ilr[:, i], Y_sim, label="Data", c=colours[0])

            plt.scatter(X_star_ilr[:, i], Y_star, alpha=.1, label="True Effect", c=colours[2])
            plt.plot(X_star_ilr[:, i], get_Y_trendline(X_star_ilr[:, i], Y_star), label="Trendline True Effect", c=colours[2])

            plt.scatter(X_star_ilr[:, i], Yhat, alpha=0.1, label="Fitted Effect", c=colours[1])
            plt.plot(X_star_ilr[:, i], get_Y_trendline(X_star_ilr[:, i], Yhat), label="Trendline Fitted Effect", c=colours[1])

            plt.xlabel("X - Transformed Microbiome Data")
            plt.ylabel("Y - Weigth")
            plt.legend()

            plt.savefig(os.path.join(path, title + ".png"))

    plt.show()

def plot_mse_results(df_mse,
                     filter_list=["ALR+LC", "ILR+LC", "ONLY Second LC", "ONLY Second ILR",
                                  "DIR+LC", "KIV+KIV", "KIVmanual", "ILR+ILR", "ONLY Second KIV"],
                     sort_to_filter=False):
    """plot results coming out from the multiple application of running all methods,
    please note that betaT is in ilr coordinates and will be translated to log in the first step
    """
    mask = df_mse["Method"].isin(filter_list)

    # sort dataframe according to filter_kust
    df_mse_mask = df_mse[mask]

    if sort_to_filter:
        df_mse_mask.Method = df_mse_mask.Method.astype("category")
        sorter = filter_list
        df_mse_mask.Method.cat.set_categories(sorter, inplace=True)
        df_mse_mask = df_mse_mask.sort_values(["Method"])

    fig = go.Figure()

    fig.add_trace(
        go.Box(x=df_mse_mask.Method, y=df_mse_mask.MSE,
            marker=dict(
                       color=plotly_colors[0],
                       line_color=plotly_colors[0]),

    )

    )
    fig.update_yaxes(rangemode="tozero")
    fig.update_layout(yaxis=dict(title="OOS MSE"))
    fig = update_layout(fig)
    return fig



def plot_beta_results(df_beta, betaT,
                      filter_list=["ALR+LC", "ILR+LC", "ONLY Second LC", "ONLY Second ILR",
                                   "DIR+LC", "KIV+KIV", "KIVmanual", "ILR+ILR", "ONLY Second KIV"],
                      beta_zero=False,
                      sort_to_filter=False
                      ):



    unique_beta = np.round(np.unique(betaT), 3).tolist()
    betaT = np.round(betaT, 3)
    colours_sat_beta = zip(plotly_colors[2:len(unique_beta)+2], unique_beta)
    colours_sat_beta = pd.DataFrame(colours_sat_beta).set_index(1)

    fig = go.Figure()
    if beta_zero:
        num_coef_list = set((np.abs(betaT) < 0.2) * np.array(range(len(betaT)))) - {0}
    else:
        num_coef_list = set((np.abs(betaT) > 0.2) * np.array(range(len(betaT))))



    mask = df_beta["Method"].isin(filter_list)
    df_beta_mask = df_beta[mask]

    if sort_to_filter:
        df_beta_mask.Method = df_beta_mask.Method.astype("category")
        sorter = filter_list
        df_beta_mask.Method.cat.set_categories(sorter, inplace=True)
        df_beta_mask = df_beta_mask.sort_values(["Method"])

    for num_coef in num_coef_list:
        fig.add_trace(
            go.Box(x=df_beta_mask["Method"], y=[item[num_coef] for item in df_beta_mask["Beta"]],
                   name= str(num_coef) + ". Beta (Estimates)",
                   marker_color=colours_sat_beta.loc[float(betaT[num_coef]), 0],
                   marker=dict(
                       color=colours_sat_beta.loc[float(betaT[num_coef]), 0],
                       line_color=colours_sat_beta.loc[float(betaT[num_coef]), 0]),
                  legendgroup=str(betaT[num_coef]))

        )
        fig.add_hline(y=betaT[num_coef], line_dash="dash",
                      line_color=colours_sat_beta.loc[float(betaT[num_coef]), 0],
                      #annotation_text="Ground Truth",
                      name="Ground Truth")

    fig = update_layout(fig)

    return fig


def plot_beta_results_2(df_beta, betaT,
                      filter_list=["ALR+LC", "ILR+LC", "ONLY Second LC", "ONLY Second ILR",
                                   "DIR+LC", "KIV+KIV", "KIVmanual", "ILR+ILR", "ONLY Second KIV"],
                      beta_zero=False,
                      sort_to_filter=False
                      ):
    """plot results coming out from the multiple application of running all methods,
    please note that betaT is in ilr coordinates and will be translated to log in the first step
    """

    fig = go.Figure()
    if beta_zero:
        num_coef_list = set((np.abs(betaT) < 0.2) * np.array(range(len(betaT)))) - {0}
    else:
        num_coef_list = set((np.abs(betaT) > 0.2) * np.array(range(len(betaT))))
    mask = df_beta["Method"].isin(filter_list)

    if sort_to_filter:
        df_beta.Method = df_beta.Method.astype("category")
        sorter = filter_list
        df_beta.Method.cat.set_categories(sorter, inplace=True)

    for num_coef in num_coef_list:
        fig.add_trace(
            go.Box(x=df_beta["Method"][mask], y=[item[num_coef] for item in df_beta["Beta"][mask]],
                   name=str(num_coef) + ". Beta (Estimates)")

        )
        if not beta_zero:
            fig.add_hline(y=betaT[num_coef], line_dash="dash", annotation_text=str(num_coef) + " Ground Truth")
    if beta_zero:
        fig.add_hline(y=0, line_dash="dash", annotation_text="Ground Truth")
    fig.update_layout(yaxis=dict(title="Beta Coefficient"))
    fig = update_layout(fig)

    return fig


def plot_diversity_methods(x, y, xstar, xstar_bound, ystar_ols, ystar_2sls, ystar_kiv, results, ytrue=None):
    fig = go.Figure()
    # for i in np.array([0, 1]):
    #    fig.add_trace(
    #        go.Scatter(x=x[z==i].squeeze(), y=y[z==i].squeeze(),
    #                   mode="markers", marker=dict(color="black"),
    #                  name="Control" if i==0 else "STAT")
    #    )



    fig.add_trace(
        go.Scatter(x=x.squeeze(), y=y.squeeze(),
                   mode="markers", marker=dict(color="black"),
                   marker_symbol="x-thin",
                   name="Data")
    )

    if ytrue is not None:
        fig.add_trace(
            go.Scatter(x=x.squeeze(), y=y.squeeze(),
                       mode="markers", marker=dict(color="lightgrey"),
                       name="True Effect")
        )
    fig.add_trace(
        go.Scatter(x=xstar, y=ystar_ols, name="OLS", mode="lines",
                   line=dict(dash="dot"),
                   line_color=plotly_colors[1])#, line_color=colours[0])
    )

    fig.add_trace(
        go.Scatter(x=xstar, y=ystar_2sls, name="2SLS", mode="lines",
                   line=dict(dash="dash"),
                   line_color=plotly_colors[2])#, line_color=colours[1])
    )
    fig.add_trace(
        go.Scatter(x=xstar, y=ystar_kiv, name="KIV", mode="lines", line=dict(dash="dashdot"),
                   line_color=plotly_colors[3])#, line_color=colours[2])
    )

    if results is not None:
        fig.add_trace(
            go.Scatter(x=xstar_bound, y=results["objective"][:, 0],
                       #marker_symbol="x",
                        line_color=plotly_colors[4],
                       marker=dict(color=plotly_colors[4],
                                   line=dict(color=plotly_colors[4])),
                       name="GB lower",)# line_color=colours[3])
        )

        fig.add_trace(
            go.Scatter(x=xstar_bound, y=results["objective"][:, 1],
                       #marker_symbol="x",
                        line_color=plotly_colors[5],
                       marker=dict(color=plotly_colors[5],
                                   line=dict(color=plotly_colors[5])
                                   ),
                       name="GB upper",)# line_color=colours[4])

        )

    fig.update_layout(yaxis=dict(title="Weight (standardized)"),
                      xaxis=dict(title="Diversity (standardized)"))

    fig = update_layout(fig)
    return fig