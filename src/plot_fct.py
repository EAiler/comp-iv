import logging
logging.basicConfig(level=logging.INFO)

import numpy as onp
import os
import plotly.graph_objects as go
import pandas as pd
from jax import numpy as np
from plotly.subplots import make_subplots

import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "generalIVmodel/"))


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




def update_layout(fig):
    """update layout for the paper

    Parameters
    ----------
    fig : plotly figure

    Returns
    -------
    fig : plotly figure
        input figure with updated layout

    """

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
    """update layout for the paper

        Parameters
        ----------
        fig : plotly express figure

        Returns
        -------
        fig : plotly express figure
            input figure with updated layout

        """
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
    """plot ilr X vs Y for the different components

    Parameters
    ----------
    X_sim_ilr : np.ndarray
        simulated X in ilr coordinates
    X_star_ilr : np.ndarray
        interventional X in ilr coordinates
    Y_sim : np.ndarray
        simulated Y
    Y_star : np.ndarray
        Y outcome from intervential X_star_ilr

    Returns
    -------
    fig : plotly figure


    """

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
    """ returns linear trendline for given X and Y

    Parameters
    ----------
    X : np.ndarray
        input X
    Y : np.ndarray
        input Y

    Returns
    -------
    Yhat : np.ndarray
        trendline for Y on given X values

    """
    p = onp.poly1d(onp.polyfit(onp.array(X).squeeze(), onp.array(Y).squeeze(), 1))
    Yhat = p(X)
    return Yhat


def plot_mse_results(df_mse,
                     filter_list=["ALR+LC", "ILR+LC", "ONLY Second LC", "ONLY Second ILR",
                                  "DIR+LC", "KIV+KIV", "KIVmanual", "ILR+ILR", "ONLY Second KIV"],
                     sort_to_filter=False):
    """plot oos mse results coming out from the multiple application of running all methods in the function run_methods_all

    Parameters
    ----------
    df_mse : pd.DataFrame
        output dataframe from run_methods_all function for the oos mse
    filter_list : list=["ALR+LC", "ILR+LC", "ONLY Second LC", "ONLY Second ILR", "DIR+LC", "KIV+KIV", "KIVmanual",
    "ILR+ILR", "ONLY Second KIV"]
        list of possible methods to plot
    sort_to_filter : bool=False
        whether to sort the methods in the plot according to the order provided in filter_list

    Returns
    -------
    fig : plotly figure


    See also
    --------
    run_methods_all

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
    """plot beta results coming out from the multiple application of running all methods in the function run_methods_all

    Parameters
    ----------
    df_beta : pd.DataFrame
        output dataframe from run_methods_all function for the beta values
    betaT : np.ndarray
        true causal beta values
    filter_list : list=["ALR+LC", "ILR+LC", "ONLY Second LC", "ONLY Second ILR", "DIR+LC", "KIV+KIV", "KIVmanual",
    "ILR+ILR", "ONLY Second KIV"]
        list of possible methods to plot
    beta_zero : bool=False
        whether to plot the influential beta values (False) or to plot the non-influential/zero beta values (True)
    sort_to_filter : bool=False
        whether to sort the methods in the plot according to the order provided in filter_list

    Returns
    -------
    fig : plotly figure


    See also
    --------
    run_methods_all

    """
    unique_beta = np.round(np.unique(betaT), 3).tolist()
    betaT = np.round(betaT, 3)
    colours_sat_beta = zip(plotly_colors[2:len(unique_beta)+2], unique_beta)
    colours_sat_beta = pd.DataFrame(colours_sat_beta).set_index(1)

    fig = go.Figure()
    if beta_zero:
        num_coef_list = set((np.abs(betaT) < 0.09) * np.array(range(len(betaT)))) - {0}
    else:
        num_coef_list = set((np.abs(betaT) > 0.09) * np.array(range(len(betaT))))



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



def plot_diversity_methods(x, y, xstar, xstar_bound, ystar_ols, ystar_2sls, ystar_kiv, results, ytrue=None):
    """
    plot the outcome from the diversity methods, if some methods where not computed, the values should be set to None

    Parameters
    ----------
    x : np.ndarray
        diversity of the data points
    y : np.ndarray
        outcome with respect to the diversity in the data points, i.e. weight
    xstar : np.ndarray
        evaluated data points for predictions
    xstar_bound : np.ndarray
        evaluated data points for prediction in bounding mehtod
    ystar_ols : np.ndarray
        predicted y values for OLS method
    ystar_2sls : np.ndarray
        predicted y values for 2SLS method
    ystar_kiv : np.ndarray
        predicted y values for KIV method
    results : dict
        predicted y for bounding method, contained in dictionary according to original implementation
    ytrue : np.ndarray=None
        true causal y values if applicable

    Results
    -------
    fig : plotly figure

    """

    fig = go.Figure()

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