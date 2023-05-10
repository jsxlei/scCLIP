#!/usr/bin/env python
"""
# Author: Lei Xiong
# Contact: jsxlei@gmail.com
# File Name: plot.py
# Created Time : Fri 21 Jan 2022 04:12:32 AM CST
# Description:

"""

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


# import scanpy as sc

def plot_paired_scatter(concat, cmap=plt.cm.tab10.colors): # TO DO
    plt.figure(figsize=(4,4))
    plt.plot(concat[:, 0].reshape(2, -1), concat[:, 1].reshape(2, -1), 
            color='gray', linestyle='dashed', linewidth=0.5)
    # concat = np.concatenate([X, Y])
    plt.scatter(
        concat[:, 0],
        concat[:, 1],
        s=min(10, 12000/len(concat)),
        c=[cmap[0]]*(len(concat)//2)+[cmap[1]]*(len(concat)//2)
    )

    plt.gca().set_facecolor('white')
    plt.xticks([], [])
    plt.yticks([], [])
    # plt.set_xticks([])
    # plt.set_yticks([])
    # plt.grid(False)
    # plt.axis('off')
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.tight_layout()
    return plt.gcf()


def plot_umap(
    adata, 
    color=None,
    save=None,
    n_neighbors=30, 
    min_dist=0.5, 
    metric='euclidean', #'correlation',  
    use_rep='X',
    # **tl_kwargs,
    **pl_kwargs,
):
    sc.settings.set_figure_params(dpi=80, facecolor='white',figsize=(4,4), frameon=True)
    sc.pp.neighbors(adata, metric=metric, use_rep=use_rep) #, n_neighbors=n_neighbors),
    sc.tl.leiden(adata)
    sc.tl.umap(adata) #, min_dist=min_dist)

    
    if 'show' in pl_kwargs and not pl_kwargs['show']:
        axis = sc.pl.umap(adata, color=color, save=save, wspace=0.4, ncols=4, **pl_kwargs)
        # concat = adata.obsm['X_umap']
        # plt.plot(concat[:, 0].reshape(2, -1), concat[:, 1].reshape(2, -1), 
        # color='gray', linestyle='dashed', linewidth=0.5)
        return axis[0].figure if isinstance(axis, list) else axis.figure
    else:
        sc.pl.umap(adata, color=color, save=save, wspace=0.65, ncols=4, show=False, **pl_kwargs)

    # plt.close('all')
    # return 
        
def plot_paired_umap(
    adata, 
    color=['cell_type', 'modality'],
    save=None,
    n_neighbors=30, 
    min_dist=0.5, 
    metric='euclidean', #'correlation',  
    use_rep='X',
    # **tl_kwargs,
    **pl_kwargs,
):
    sc.settings.set_figure_params(dpi=80, facecolor='white',figsize=(4,4), frameon=True)
    sc.pp.neighbors(adata, metric=metric, use_rep=use_rep) #, n_neighbors=n_neighbors),
    sc.tl.leiden(adata)
    sc.tl.umap(adata) #, min_dist=min_dist)

    ncols=2
    nrows=1
    figsize=4
    wspace=0.5
    fig,axs = plt.subplots(nrows=nrows, ncols=ncols,
                           figsize=(ncols*figsize+figsize*wspace*(ncols-1),nrows*figsize))
    # plt.subplots_adjust(wspace=wspace)

    sc.pl.umap(adata, color='cell_type', ax=axs[0], show=False, legend_loc='on data')
    sc.pl.umap(adata, color='modality', ax=axs[1], show=False)
    concat = adata.obsm['X_umap']
    plt.plot(concat[:, 0].reshape(2, -1), concat[:, 1].reshape(2, -1), 
            color='gray', linestyle='dashed', linewidth=0.5)
    plt.tight_layout()
    
    if save:
        plt.savefig(save)
    else:
        return plt.gcf()
    

def plot_heatmap(X, obs_names=None,  var_names=None,  col_cluster=False, row_cluster=False, col_title='', row_title='', sort=False,cmap='viridis', cax_visible=True, **kwargs): # TO DO: support obs_names as pd.DataFrame
    if obs_names is not None:
        if sort:
            index = sort_index_by_classes(obs_names)
            X = X[index]
            obs_names = obs_names[index]
        row_colors, row_colors_dict = map_color(obs_names, return_dict=True)
        row_legend = [mpatches.Patch(color=color, label=c) for c, color in row_colors_dict.items()]
    else:
        row_colors = None

    if var_names is not None:
        if sort:
            index = sort_index_by_classes(var_names)
            X = X[:, index]
            var_names = var_names[index]
        col_colors, col_colors_dict = map_color(var_names, return_dict=True)
        col_legend = [mpatches.Patch(color=color, label=c) for c, color in col_colors_dict.items()]
    else:
        col_colors = None
    
    g = sns.clustermap(X, row_cluster=row_cluster, col_cluster=col_cluster, row_colors=row_colors, col_colors=col_colors, cmap=cmap, **kwargs)
    if col_colors is not None:
        g.ax_col_dendrogram.legend(title=col_title, loc='center', handles=col_legend, frameon=False, ncol=4, bbox_to_anchor=(0.5, 0.85))
        # g.ax_heatmap.legend(loc='upper center', handles=row_legend, frameon=False, ncol=4, bbox_to_anchor=(0.5, 1.2))

    if row_colors is not None:
        g.ax_row_dendrogram.legend(title=row_title, loc='center', handles=row_legend, frameon=False, ncol=1, bbox_to_anchor=(0.2, 0.5))
        # g.ax_heatmap.legend(loc='upper center', handles=col_legend, frameon=False, ncol=4, bbox_to_anchor=(0.5, 1.3))
    g.cax.set_visible(cax_visible)
    
    # if col_cluster:
    #     col_ind = g.dendrogram_col.reordered_ind
    #     return plt.gcf(), col_ind
    return plt.gcf()



def dim_reduction(X, method='UMAP', metric='correlation'):
    if method == 'TSNE':
        from sklearn.manifold import TSNE
        X = TSNE(n_components=2).fit_transform(X)
    if method == 'UMAP':
        from umap import UMAP
        X = UMAP(n_neighbors=30, min_dist=0.1, metric=metric).fit_transform(X)
    if method == 'PCA':
        from sklearn.decomposition import PCA
        X = PCA(n_components=2).fit_transform(X)
    return X

def plot_embedding(
    X, 
    labels=None, 
    classes=None, 
    cmap=None, 
    save=False, 
    show_legend=True, 
    method='UMAP', 
    metric='correlation', 
    **legend_params
):
        
    plt.figure(figsize=(8, 8))
    labels = np.array(labels)
    if classes is None:
        classes = np.unique(labels)

    if len(classes) <= 10:
        cmap = 'tab10'
    elif len(classes) <= 20:
        cmap = 'tab20'
    else:
        cmap = 'husl'
    colors = sns.color_palette(cmap, n_colors=len(classes))

    if X.shape[1] > 2:
        X = dim_reduction(X, method=method, metric=metric)
        
    for i, c in enumerate(classes):
        plt.scatter(X[labels==c, 0], X[labels==c, 1], color=colors[i], \
                                    label=c, s=50, edgecolors='none', alpha=0.8)
#     plt.axis("off")
    
    legend_params_ = {'loc': 'center left',
                     'bbox_to_anchor':(1.0, 0.45),
                     'fontsize': 10,
                     'ncol': 1,
                     'frameon': False,
                     'markerscale': 1.5
                    }
    legend_params.update(legend_params_)
    if show_legend:
        plt.legend(**legend_params)
    if save:
        # if os.path.isfile(save):
        #     print('remove previous figure {}'.format(save))
        #     os.remove(save)
        plt.savefig(save, format='pdf', bbox_inches='tight')
    else:
        plt.show()

def plot_maps(maps, title=None, xticklabels=None, yticklabels=None,
                       n_col=4, figsize=4, save=None, **kwargs): #vmin=0, origin='lower'):
    """
    maps: (n_layer n_head d_q d_k) or (n d_q d_k)
    """
    if len(maps.shape) == 3:
        n, d_q, d_k = maps.shape
        n_row = (n-1) // n_col + 1
        maps = maps.reshape(n_row, n_col, d_q, d_k)
    elif len(maps.shape) == 4:
        n_row, n_col, d_q, d_k = maps.shape
        n = n_row * n_col

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import os

# import scanpy as sc
# sc.settings.set_figure_params(dpi=80, facecolor='white',figsize=(3,3),frameon=True)


def plot_maps(maps, title=None, xticklabels=None, yticklabels=None,
                       n_col=4, figsize=4, save=None, **kwargs): #vmin=0, origin='lower'):
    """
    maps: (n_layer n_head d_q d_k) or (n d_q d_k)
    """
    if len(maps.shape) == 3:
        n, d_q, d_k = maps.shape
        n_row = (n-1) // n_col + 1
        maps = maps.reshape(n_row, n_col, d_q, d_k)
    elif len(maps.shape) == 4:
        n_row, n_col, d_q, d_k = maps.shape
        n = n_row * n_col

    fig, ax = plt.subplots(n_row, n_col, figsize=(n_col*figsize, n_row*figsize))
    if title is None:
        title = np.array(['Sample {}'.format(i) for i in range(n)]).reshape(n_row, n_col)
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
        plt.savefig(save, format='png', bbox_inches='tight')
    else:
        plt.show()
    # return fig





def plot_correlation(X=None, corr=None, xticklabels=None, yticklabels=None, vmin=None):#, vmin=0):
    if corr is None:
        corr = np.corrcoef(X)
    # np.fill_diagonal(corr, 0)
    fig, ax = plt.subplots(1,1)
    ax.imshow(corr, vmin=vmin)
    if xticklabels is not None:
        ax.set_xticks(list(range(len(xticklabels))))
        ax.set_xticklabels(xticklabels, rotation=45, ha='right', fontsize=5)
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
        colors_dict = {c:colormap(i) for i, c in enumerate(np.unique(labels))}
    else:
        import colorcet as cc
        colormap = sns.color_palette(cc.glasbey, n_colors=len(np.unique(labels)))
        colors_dict = {c:colormap[i] for i, c in enumerate(np.unique(labels))}
    colors = [ colors_dict[c] for c in labels ]
    if return_dict:
        return colors, colors_dict
    return colors 

if __name__ == '__main__':
    maps = np.random.rand(16, 20, 30)
    # fig = plot_attention_map(maps)
    # fig.show()

    
    
# def plot_model_embedding(model, save=None):
    # peak_embedding = model.peak_embedding.weight.detach().cpu().numpy()
# import pickle
# from anndata import AnnData
# import scanpy as sc
# def plot_embedding(X, obs=None, var=None, out_dir=None, strategy=None):
#     sc.settings.set_figure_params(dpi=80, facecolor='white',figsize=(3,3),frameon=True)

#     adata = AnnData(X, obs=obs, var=var)
#     sc.pp.neighbors(adata, n_neighbors=30)
#     sc.tl.leiden(adata)
#     sc.tl.umap(adata, min_dist=0.1)
#     if 'annotation' in adata.obs:
#         colors = ['leiden', 'annotation']
#     else:
#         colors = ['leiden']
#     if out_dir:
#         sc.settings.figdir, name = out_dir.rsplit('/', 1)
#         sc.pl.umap(adata, color=colors, save='_'+name.split('.')[0], wspace=0.4, ncols=4)
#         # pickle.dump(adata, open(out_dir, 'wb'))
#         adata.write(out_dir) 
#     else:
#         return adata


      
    
def sort_index_by_classes(y):
    index = []
    for c in np.unique(y):
        ind = np.where(y==c)[0]
        index.append(ind)
    index = np.concatenate(index)
    return index

    
def plot_reorder_module(attn, ncol=4, vmin=0, vmax=None):
    nrow = (attn.shape[0]-1)//ncol + 1
    fig, axs = plt.subplots(nrow, ncol, figsize=(ncol*4, nrow*3), squeeze=False)
    for i in range(attn.shape[0]):
        if i == 0:
            grid = sns.clustermap(attn[i])
            row_ind = grid.dendrogram_row.reordered_ind
            col_ind = grid.dendrogram_col.reordered_ind
            sns.heatmap(attn[i][row_ind][:, col_ind],  square=True, vmin=vmin, vmax=vmax, cbar=True, ax=axs[i//ncol, i%ncol])
        else:
            sns.heatmap(attn[i][row_ind][:, col_ind],  square=True, vmin=vmin, vmax=vmax, cbar=True, ax=axs[i//ncol, i%ncol])
    plt.show()

    fig, ax = plt.subplots(n_row, n_col, figsize=(n_col*figsize, n_row*figsize))
    if title is None:
        title = np.array(['Sample {}'.format(i) for i in range(n)]).reshape(n_row, n_col)
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
        plt.savefig(save, format='png', bbox_inches='tight')
    else:
        plt.show()
    # return fig





def plot_correlation(X=None, corr=None, xticklabels=None, yticklabels=None, vmin=None):#, vmin=0):
    if corr is None:
        corr = np.corrcoef(X)
    # np.fill_diagonal(corr, 0)
    fig, ax = plt.subplots(1,1)
    ax.imshow(corr, vmin=vmin)
    if xticklabels is not None:
        ax.set_xticks(list(range(len(xticklabels))))
        ax.set_xticklabels(xticklabels, rotation=45, ha='right', fontsize=5)
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
        colors_dict = {c:colormap(i) for i, c in enumerate(np.unique(labels))}
    else:
        import colorcet as cc
        colormap = sns.color_palette(cc.glasbey, n_colors=len(np.unique(labels)))
        colors_dict = {c:colormap[i] for i, c in enumerate(np.unique(labels))}
    colors = [ colors_dict[c] for c in labels ]
    if return_dict:
        return colors, colors_dict
    return colors 

if __name__ == '__main__':
    maps = np.random.rand(16, 20, 30)
    # fig = plot_attention_map(maps)
    # fig.show()

    
    
# def plot_model_embedding(model, save=None):
    # peak_embedding = model.peak_embedding.weight.detach().cpu().numpy()
# import pickle
# from anndata import AnnData
# import scanpy as sc
# def plot_embedding(X, obs=None, var=None, out_dir=None, strategy=None):
#     sc.settings.set_figure_params(dpi=80, facecolor='white',figsize=(3,3),frameon=True)

#     adata = AnnData(X, obs=obs, var=var)
#     sc.pp.neighbors(adata, n_neighbors=30)
#     sc.tl.leiden(adata)
#     sc.tl.umap(adata, min_dist=0.1)
#     if 'annotation' in adata.obs:
#         colors = ['leiden', 'annotation']
#     else:
#         colors = ['leiden']
#     if out_dir:
#         sc.settings.figdir, name = out_dir.rsplit('/', 1)
#         sc.pl.umap(adata, color=colors, save='_'+name.split('.')[0], wspace=0.4, ncols=4)
#         # pickle.dump(adata, open(out_dir, 'wb'))
#         adata.write(out_dir) 
#     else:
#         return adata


      
    
def sort_index_by_classes(y):
    index = []
    for c in np.unique(y):
        ind = np.where(y==c)[0]
        index.append(ind)
    index = np.concatenate(index)
    return index

    
def plot_reorder_module(attn, ncol=4, vmin=0, vmax=None):
    nrow = (attn.shape[0]-1)//ncol + 1
    fig, axs = plt.subplots(nrow, ncol, figsize=(ncol*4, nrow*3), squeeze=False)
    for i in range(attn.shape[0]):
        if i == 0:
            grid = sns.clustermap(attn[i])
            row_ind = grid.dendrogram_row.reordered_ind
            col_ind = grid.dendrogram_col.reordered_ind
            sns.heatmap(attn[i][row_ind][:, col_ind],  square=True, vmin=vmin, vmax=vmax, cbar=True, ax=axs[i//ncol, i%ncol])
        else:
            sns.heatmap(attn[i][row_ind][:, col_ind],  square=True, vmin=vmin, vmax=vmax, cbar=True, ax=axs[i//ncol, i%ncol])
    plt.show()


#!/usr/bin/env python
"""
# Author: Lei Xiong
# Contact: jsxlei@gmail.com
# File Name: regnet/pl/track.py
# Created Time : Thu 10 Mar 2022 01:56:58 PM EST
# Description:

"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np



def plot_tracks(tracks, interval, height=1.5):
    fig, axes = plt.subplots(len(tracks), 1, figsize=(20, height * len(tracks)), sharex=True)
    for ax, (title, y) in zip(axes, tracks.items()):
        ax.fill_between(np.linspace(interval[0], interval[1], num=len(y)), y)
        ax.set_title(title)
        sns.despine(top=True, right=True, bottom=True)
    ax.set_xlabel(str(interval))
    plt.tight_layout()

    



# =======================================================
#                   Vis sequence from Deep List
# =======================================================
def plot_a(ax, base, left_edge, height, color):
    a_polygon_coords = [
        np.array([
           [0.0, 0.0],
           [0.5, 1.0],
           [0.5, 0.8],
           [0.2, 0.0],
        ]),
        np.array([
           [1.0, 0.0],
           [0.5, 1.0],
           [0.5, 0.8],
           [0.8, 0.0],
        ]),
        np.array([
           [0.225, 0.45],
           [0.775, 0.45],
           [0.85, 0.3],
           [0.15, 0.3],
        ])
    ]
    for polygon_coords in a_polygon_coords:
        ax.add_patch(matplotlib.patches.Polygon((np.array([1,height])[None,:]*polygon_coords
                                                 + np.array([left_edge,base])[None,:]),
                                                facecolor=color, edgecolor=color))


def plot_c(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge+0.65, base+0.5*height], width=1.3, height=height,
                                            facecolor=color, edgecolor=color))
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge+0.65, base+0.5*height], width=0.7*1.3, height=0.7*height,
                                            facecolor='white', edgecolor='white'))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+1, base], width=1.0, height=height,
                                            facecolor='white', edgecolor='white', fill=True))


def plot_g(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge+0.65, base+0.5*height], width=1.3, height=height,
                                            facecolor=color, edgecolor=color))
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge+0.65, base+0.5*height], width=0.7*1.3, height=0.7*height,
                                            facecolor='white', edgecolor='white'))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+1, base], width=1.0, height=height,
                                            facecolor='white', edgecolor='white', fill=True))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+0.825, base+0.085*height], width=0.174, height=0.415*height,
                                            facecolor=color, edgecolor=color, fill=True))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+0.625, base+0.35*height], width=0.374, height=0.15*height,
                                            facecolor=color, edgecolor=color, fill=True))


def plot_t(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+0.4, base],
                  width=0.2, height=height, facecolor=color, edgecolor=color, fill=True))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge, base+0.8*height],
                  width=1.0, height=0.2*height, facecolor=color, edgecolor=color, fill=True))

default_colors = {0:'green', 1:'blue', 2:'orange', 3:'red'}
default_plot_funcs = {0:plot_a, 1:plot_c, 2:plot_g, 3:plot_t}
def plot_weights_given_ax(ax, array,
                 height_padding_factor,
                 length_padding,
                 subticks_frequency,
                 highlight,
                 colors=default_colors,
                 plot_funcs=default_plot_funcs):
    if len(array.shape)==3:
        array = np.squeeze(array)
    assert len(array.shape)==2, array.shape
    if (array.shape[0]==4 and array.shape[1] != 4):
        array = array.transpose(1,0)
    assert array.shape[1]==4
    max_pos_height = 0.0
    min_neg_height = 0.0
    heights_at_positions = []
    depths_at_positions = []
    for i in range(array.shape[0]):
        #sort from smallest to highest magnitude
        acgt_vals = sorted(enumerate(array[i,:]), key=lambda x: abs(x[1]))
        positive_height_so_far = 0.0
        negative_height_so_far = 0.0
        for letter in acgt_vals:
            plot_func = plot_funcs[letter[0]]
            color=colors[letter[0]]
            if (letter[1] > 0):
                height_so_far = positive_height_so_far
                positive_height_so_far += letter[1]                
            else:
                height_so_far = negative_height_so_far
                negative_height_so_far += letter[1]
            plot_func(ax=ax, base=height_so_far, left_edge=i, height=letter[1], color=color)
        max_pos_height = max(max_pos_height, positive_height_so_far)
        min_neg_height = min(min_neg_height, negative_height_so_far)
        heights_at_positions.append(positive_height_so_far)
        depths_at_positions.append(negative_height_so_far)

    #now highlight any desired positions; the key of
    #the highlight dict should be the color
    for color in highlight:
        for start_pos, end_pos in highlight[color]:
            assert start_pos >= 0.0 and end_pos <= array.shape[0]
            min_depth = np.min(depths_at_positions[start_pos:end_pos])
            max_height = np.max(heights_at_positions[start_pos:end_pos])
            ax.add_patch(
                matplotlib.patches.Rectangle(xy=[start_pos,min_depth],
                    width=end_pos-start_pos,
                    height=max_height-min_depth,
                    edgecolor=color, fill=False))
            
    ax.set_xlim(-length_padding, array.shape[0]+length_padding)
    ax.xaxis.set_ticks(np.arange(0.0, array.shape[0]+1, subticks_frequency))
    height_padding = max(abs(min_neg_height)*(height_padding_factor),
                         abs(max_pos_height)*(height_padding_factor))
    ax.set_ylim(min_neg_height-height_padding, max_pos_height+height_padding)


def plot_weights(array,
                 figsize=(50,2),
                 height_padding_factor=0.2,
                 length_padding=1.0,
                 subticks_frequency=10,
                 colors=default_colors,
                 plot_funcs=default_plot_funcs,
                 highlight={}):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111) 
    plot_weights_given_ax(ax=ax, array=array,
        height_padding_factor=height_padding_factor,
        length_padding=length_padding,
        subticks_frequency=subticks_frequency,
        colors=colors,
        plot_funcs=plot_funcs,
        highlight=highlight)
    plt.show()
    
