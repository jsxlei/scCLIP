#!/usr/bin/env python

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import os
import scanpy as sc

import colorcet as cc

# palette=sns.color_palette(cc.glasbey)
# colormap = sns.color_palette(cc.glasbey, n_colors=len(np.unique(labels)))

import pynndescent
import numba


@numba.njit(fastmath=True)
def correct_alternative_cosine(ds):
    result = np.empty_like(ds)
    for i in range(ds.shape[0]):
        result[i] = 1.0 - np.power(2.0, ds[i])
    return result


pynn_dist_fns_fda = pynndescent.distances.fast_distance_alternatives
pynn_dist_fns_fda["cosine"]["correction"] = correct_alternative_cosine
pynn_dist_fns_fda["dot"]["correction"] = correct_alternative_cosine


def plot_umap(
    adata,
    color=None,
    save=None,
    n_neighbors=30,
    min_dist=0.5,
    metric="euclidean",  #'correlation',
    use_rep="X",
    # **tl_kwargs,
    **pl_kwargs,
):
    sc.settings.set_figure_params(
        dpi=80, facecolor="white", figsize=(4, 4), frameon=True
    )
    sc.pp.neighbors(
        adata, metric=metric, use_rep=use_rep
    )  # , n_neighbors=n_neighbors),
    sc.tl.leiden(adata)
    sc.tl.umap(adata)  # , min_dist=min_dist)

    if "show" in pl_kwargs and not pl_kwargs["show"]:
        axis = sc.pl.umap(
            adata, color=color, save=save, wspace=0.4, ncols=4, **pl_kwargs
        )
        return axis[0].figure if isinstance(axis, list) else axis.figure
    else:
        sc.pl.umap(
            adata, color=color, save=save, wspace=0.65, ncols=4, show=False, **pl_kwargs
        )

    # plt.close('all')
    # return


def plot_paired_umap(
    adata,
    color=["cell_type", "modality"],
    save=None,
    n_neighbors=30,
    min_dist=0.5,
    metric="euclidean",  #'correlation',
    use_rep="X",
    # **tl_kwargs,
    **pl_kwargs,
):
    sc.settings.set_figure_params(
        dpi=80, facecolor="white", figsize=(4, 4), frameon=True
    )
    sc.pp.neighbors(
        adata, metric=metric, use_rep=use_rep
    )  # , n_neighbors=n_neighbors),
    sc.tl.leiden(adata)
    sc.tl.umap(adata)  # , min_dist=min_dist)

    ncols = 2
    nrows = 1
    figsize = 4
    wspace = 0.5
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(ncols * figsize + figsize * wspace * (ncols - 1), nrows * figsize),
    )
    plt.subplots_adjust(wspace=wspace)

    sc.pl.umap(adata, color="cell_type", ax=axs[0], show=False, legend_loc="on data")
    sc.pl.umap(adata, color="modality", ax=axs[1], show=False)
    concat = adata.obsm["X_umap"]
    plt.plot(
        concat[:, 0].reshape(2, -1),
        concat[:, 1].reshape(2, -1),
        color="gray",
        linestyle="dashed",
        linewidth=0.5,
    )
    plt.tight_layout()

    if save:
        plt.savefig(save)
    else:
        return plt.gcf()


def plot_heatmap(
    X,
    obs_names=None,
    var_names=None,
    col_cluster=False,
    row_cluster=False,
    col_title="",
    row_title="",
    sort=False,
    cmap="viridis",
    cax_visible=True,
    **kwargs,
):  # TO DO: support obs_names as pd.DataFrame
    if obs_names is not None:
        if sort:
            index = sort_index_by_classes(obs_names)
            X = X[index]
            obs_names = obs_names[index]
        row_colors, row_colors_dict = map_color(obs_names, return_dict=True)
        row_legend = [
            mpatches.Patch(color=color, label=c) for c, color in row_colors_dict.items()
        ]
    else:
        row_colors = None

    if var_names is not None:
        if sort:
            index = sort_index_by_classes(var_names)
            X = X[:, index]
            var_names = var_names[index]
        col_colors, col_colors_dict = map_color(var_names, return_dict=True)
        col_legend = [
            mpatches.Patch(color=color, label=c) for c, color in col_colors_dict.items()
        ]
    else:
        col_colors = None

    g = sns.clustermap(
        X,
        row_cluster=row_cluster,
        col_cluster=col_cluster,
        row_colors=row_colors,
        col_colors=col_colors,
        cmap=cmap,
        **kwargs,
    )
    if col_colors is not None:
        g.ax_col_dendrogram.legend(
            title=col_title,
            loc="center",
            handles=col_legend,
            frameon=False,
            ncol=4,
            bbox_to_anchor=(0.5, 0.85),
        )
        # g.ax_heatmap.legend(loc='upper center', handles=row_legend, frameon=False, ncol=4, bbox_to_anchor=(0.5, 1.2))

    if row_colors is not None:
        g.ax_row_dendrogram.legend(
            title=row_title,
            loc="center",
            handles=row_legend,
            frameon=False,
            ncol=1,
            bbox_to_anchor=(0.2, 0.5),
        )
        # g.ax_heatmap.legend(loc='upper center', handles=col_legend, frameon=False, ncol=4, bbox_to_anchor=(0.5, 1.3))
    g.cax.set_visible(cax_visible)

    # if col_cluster:
    #     col_ind = g.dendrogram_col.reordered_ind
    #     return plt.gcf(), col_ind
    return plt.gcf()


# import scanpy as sc
# sc.settings.set_figure_params(dpi=80, facecolor='white',figsize=(3,3),frameon=True)


def plot_maps(
    maps,
    title=None,
    xticklabels=None,
    yticklabels=None,
    n_col=4,
    figsize=4,
    save=None,
    **kwargs,
):  # vmin=0, origin='lower'):
    """
    maps: (n_layer n_head d_q d_k) or (n d_q d_k)
    """
    if len(maps.shape) == 3:
        n, d_q, d_k = maps.shape
        n_row = (n - 1) // n_col + 1
        maps = maps.reshape(n_row, n_col, d_q, d_k)
    elif len(maps.shape) == 4:
        n_row, n_col, d_q, d_k = maps.shape
        n = n_row * n_col

    fig, ax = plt.subplots(n_row, n_col, figsize=(n_col * figsize, n_row * figsize))
    if title is None:
        title = np.array(["Sample {}".format(i) for i in range(n)]).reshape(
            n_row, n_col
        )
    else:
        title = title.reshape(n_row, n_col)

    for row in range(n_row):
        for column in range(n_col):
            ax[row][column].imshow(maps[row][column], **kwargs)
            ax[row][column].set_title(title[row][column])

            # if xticklabels is not None:
            #     ax[row][column].set_xticks(list(range(d_k)))
            #     ax[row][column].set_xticklabels(xticklabels, rotation=45, ha='right')
            # if yticklabels is not None:
            #     ax[row][column].set_yticks(list(range(d_q)))
            #     ax[row][column].set_yticklabels(yticklabels)

    # fig.subplots_adjust(hspace=0.5)
    plt.tight_layout()

    if save:
        os.makedirs(os.path.dirname(save), exist_ok=True)
        plt.savefig(save, format="png", bbox_inches="tight")
    else:
        plt.show()
    # return fig


def plot_correlation(
    X=None, corr=None, xticklabels=None, yticklabels=None, vmin=None
):  # , vmin=0):
    if corr is None:
        corr = np.corrcoef(X)
    # np.fill_diagonal(corr, 0)
    fig, ax = plt.subplots(1, 1)
    ax.imshow(corr, vmin=vmin)
    if xticklabels is not None:
        ax.set_xticks(list(range(len(xticklabels))))
        ax.set_xticklabels(xticklabels, rotation=45, ha="right", fontsize=5)
    if yticklabels is not None:
        ax.set_yticks(list(range(len(yticklabels))))
        ax.set_yticklabels(yticklabels, fontsize=5)
    plt.tight_layout()
    # sns.heatmap(corr, vmin=0, square=True)
    return plt.gcf()


def map_color(labels, return_dict=False):
    # colors = sns.color_palette(cmap, n_colors=len(labels))
    if len(np.unique(labels)) <= 20:
        colormap = plt.cm.tab20
        colors_dict = {c: colormap(i) for i, c in enumerate(np.unique(labels))}
    else:
        import colorcet as cc

        colormap = sns.color_palette(cc.glasbey, n_colors=len(np.unique(labels)))
        colors_dict = {c: colormap[i] for i, c in enumerate(np.unique(labels))}
    colors = [colors_dict[c] for c in labels]
    if return_dict:
        return colors, colors_dict
    return colors


def sort_index_by_classes(y):
    index = []
    for c in np.unique(y):
        ind = np.where(y == c)[0]
        index.append(ind)
    index = np.concatenate(index)
    return index


def plot_reorder_module(attn, ncol=4, vmin=0, vmax=None):
    nrow = (attn.shape[0] - 1) // ncol + 1
    fig, axs = plt.subplots(nrow, ncol, figsize=(ncol * 4, nrow * 3), squeeze=False)
    for i in range(attn.shape[0]):
        if i == 0:
            grid = sns.clustermap(attn[i])
            row_ind = grid.dendrogram_row.reordered_ind
            col_ind = grid.dendrogram_col.reordered_ind
            sns.heatmap(
                attn[i][row_ind][:, col_ind],
                square=True,
                vmin=vmin,
                vmax=vmax,
                cbar=True,
                ax=axs[i // ncol, i % ncol],
            )
        else:
            sns.heatmap(
                attn[i][row_ind][:, col_ind],
                square=True,
                vmin=vmin,
                vmax=vmax,
                cbar=True,
                ax=axs[i // ncol, i % ncol],
            )
    plt.show()

    fig, ax = plt.subplots(n_row, n_col, figsize=(n_col * figsize, n_row * figsize))
    if title is None:
        title = np.array(["Sample {}".format(i) for i in range(n)]).reshape(
            n_row, n_col
        )
    else:
        title = title.reshape(n_row, n_col)

    for row in range(n_row):
        for column in range(n_col):
            ax[row][column].imshow(maps[row][column], **kwargs)
            ax[row][column].set_title(title[row][column])

            # if xticklabels is not None:
            #     ax[row][column].set_xticks(list(range(d_k)))
            #     ax[row][column].set_xticklabels(xticklabels, rotation=45, ha='right')
            # if yticklabels is not None:
            #     ax[row][column].set_yticks(list(range(d_q)))
            #     ax[row][column].set_yticklabels(yticklabels)

    # fig.subplots_adjust(hspace=0.5)
    plt.tight_layout()

    if save:
        os.makedirs(os.path.dirname(save), exist_ok=True)
        plt.savefig(save, format="png", bbox_inches="tight")
    else:
        plt.show()
    # return fig


def sort_index_by_classes(y):
    index = []
    for c in np.unique(y):
        ind = np.where(y == c)[0]
        index.append(ind)
    index = np.concatenate(index)
    return index


def plot_reorder_module(attn, ncol=4, vmin=0, vmax=None):
    nrow = (attn.shape[0] - 1) // ncol + 1
    fig, axs = plt.subplots(nrow, ncol, figsize=(ncol * 4, nrow * 3), squeeze=False)
    for i in range(attn.shape[0]):
        if i == 0:
            grid = sns.clustermap(attn[i])
            row_ind = grid.dendrogram_row.reordered_ind
            col_ind = grid.dendrogram_col.reordered_ind
            sns.heatmap(
                attn[i][row_ind][:, col_ind],
                square=True,
                vmin=vmin,
                vmax=vmax,
                cbar=True,
                ax=axs[i // ncol, i % ncol],
            )
        else:
            sns.heatmap(
                attn[i][row_ind][:, col_ind],
                square=True,
                vmin=vmin,
                vmax=vmax,
                cbar=True,
                ax=axs[i // ncol, i % ncol],
            )
    plt.show()
