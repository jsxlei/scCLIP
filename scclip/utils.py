from regnet.nn.utils import pearsonr
import torch
import scanpy as sc
import os
from anndata import AnnData, concat
import pandas as pd
import numpy as np
from tqdm import tqdm
sc.settings.set_figure_params(dpi=80, facecolor='white',figsize=(4, 4),frameon=True)

from .plot import plot_umap
from .logger import create_logger
# from .data import PairedDataset
# from scalex.metrics import silhouette_score, batch_entropy_mixing_score
# from .metrics import compute_metrics


def subsample(adata, size):
    if isinstance(adata, AnnData):
        n = adata.shape[0]
    else:
        n = adata[0].shape[0]
    if isinstance(size, float):
        size = n * size
    if n > size:
        index = np.random.choice(n, size=size, replace=False)
        if isinstance(adata, AnnData):
            adata = adata[index].copy()
        else:
            adata = [ann[index].copy() for ann in adata]
    return adata


def _plot_umap(adata, args, name, out_dir=None): 
    if out_dir is None:
        out_dir = args.default_root_dir + '/{}'.format(args.paired)
    os.makedirs(out_dir, exist_ok=True)
    sc.settings.figdir = out_dir
    color = [i for i in ['leiden', args.cell_type, 'modality', 'batch'] if i in adata.obs]
    # print(f'use metric: {args.metric}', flush=True)

    plot_umap(adata, color=color, metric=args.metric, save='_{}.png'.format(name))
    adata.write(out_dir+'/{}.h5ad'.format(name))



def get_result_from_single_omics(filename, modality, model, out_dir, args):
    fn = model.to('cuda').get_atac_features if modality == 'atac' else model.to('cuda').get_rna_features
    dataloader = PairedDataset(filename, modality=modality).dataloader()
    if args.fast_dev_run:
        batch = next(iter(dataloader))
        adata = fn(batch[modality].to('cuda')).detach().cpu().numpy()
    else:
        adata = torch.concat([fn(batch[modality].to('cuda')).detach().cpu() for batch in tqdm(dataloader, desc=modality)]).numpy()
    
    adata = AnnData(adata, obs=dataloader.dataset.mdata.mod[modality].obs[:len(adata)])
    adata.write(out_dir+f'/{modality}.h5ad')

    _plot_umap(adata, args, modality, out_dir)
    return adata


def get_results(model, dataloader, trainer, args, out_dir=None):

    paired, atac, rna = args.paired, args.atac, args.rna
    if paired is not None: #and (args.atac is None and args.rna is None):
        out_dir = args.default_root_dir + '/{}'.format(paired) if out_dir is None else out_dir
        os.makedirs(out_dir, exist_ok=True)
        log = create_logger(paired, fh=out_dir+'/log.txt')
        log.info(f'model global_step: {model.global_step}')

        output = trainer.predict(model=model, dataloaders=dataloader)
        atac_embeds, rna_embeds, log_dict = zip(*output)
        # out = pd.DataFrame(trainer.predict(model=model, dataloaders=dataloader))
        if isinstance(dataloader.dataset, torch.utils.data.Subset):
            indices = dataloader.dataset.indices
            atac = dataloader.dataset.dataset.mdata.mod['atac'][indices]
            rna = dataloader.dataset.dataset.mdata.mod['rna'][indices]
        else:
            atac = dataloader.dataset.mdata.mod['atac']
            rna = dataloader.dataset.mdata.mod['rna']


        log_dict = pd.DataFrame(log_dict).mean(0).astype(float)
        log.info(f'\n{log_dict.to_string()}') 

        atac_embeds = torch.cat(atac_embeds).detach().cpu().numpy()
        rna_embeds = torch.cat(rna_embeds).detach().cpu().numpy()

        atac_embeds = AnnData(atac_embeds, obs=atac.obs[:len(atac_embeds)])
        atac_embeds.write(out_dir+'/atac_embeds.h5ad')
        rna_embeds = AnnData(rna_embeds, obs=rna.obs[:len(rna_embeds)])
        rna_embeds.write(out_dir+'/rna_embeds.h5ad')
        concat_embeds = concat([atac_embeds, rna_embeds], label='modality', keys=["atac", "rna"], index_unique='#')
        concat_embeds.write(out_dir+'/combined_embeds.h5ad')

        _plot_umap(atac_embeds, args, 'atac_embeds', out_dir)
        _plot_umap(rna_embeds, args, 'rna_embeds', out_dir)
        _plot_umap(concat_embeds, args, 'combined_embeds', out_dir)
        
        index = np.random.choice(len(atac_embeds), size=min(10000, len(atac_embeds)), replace=False)
        concat_ = concat([atac_embeds[index], rna_embeds[index]], label='modality', keys=["atac", "rna"], index_unique='#')
        # metrics = pd.Series(compute_metrics(paired=concat_, use_rep='X_umap'))
        
        if 'pred_rna' in output:
            log.info(f'compute prediction correlation')
            pred_rna = torch.cat(list(out.pred_rna.values)).detach().cpu().numpy()
            pred_rna = AnnData(pred_rna, obs=rna.obs[:len(pred_rna)], var=rna.var)
            pred_rna.write(out_dir+'/pred_rna.h5ad')

            pearsonr_rna = pearsonr(
                torch.Tensor(pred_rna.X), 
                torch.Tensor(rna.X[:len(pred_rna)].toarray()), 
                batch_first=True,
            ).nanmean().item()
            log.info(f'pearsonr RNA: {pearsonr_rna}')

        if 'pred_atac' in output:
            log.info(f'compute prediction correlation')
            pred_atac = torch.cat(list(out.pred_atac.values)).detach().cpu().numpy()
            pred_atac = AnnData(pred_atac, obs=rna.obs[:len(pred_atac)], var=atac.var)
            pred_atac.write(out_dir+'/pred_atac.h5ad')

            pearsonr_atac = pearsonr(
                torch.Tensor(pred_atac.X), 
                torch.Tensor(atac.X[:len(pred_atac)].toarray()), 
                batch_first=True,
            ).nanmean().item()
            log.info(f'pearsonr atac: {pearsonr_atac}')

        log.info(f'\n{metrics.to_string()}')
        metrics.to_csv(out_dir+'/metrics.txt', sep='\t', header=None) 

    else:
        out_dir = args.default_root_dir + '/{}'.format(atac+'_'+rna) if out_dir is None else out_dir
        os.makedirs(out_dir, exist_ok=True)
        if atac:
            atac = get_result_from_single_omics(atac, 'atac', model, out_dir, args)
        if rna:
            rna = get_result_from_single_omics(rna, 'rna', model, out_dir, args)
        if atac and rna:
            combined = concat([atac, rna], label='modality', keys=["atac", "rna"], index_unique='#')
            _plot_umap(combined, args, 'combined', out_dir)
            metrics = pd.Series(compute_metrics(paired=combined, is_paired=False))
            log = create_logger('combined', fh=out_dir+'/log.txt')
            log.info(f'\n{metrics.to_string()}')
            metrics.to_csv(out_dir+'/metrics.txt', sep='\t', header=None)



 