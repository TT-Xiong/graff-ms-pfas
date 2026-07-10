import numpy as np
import numpy.random as npr
import scipy.sparse as sp
import pandas as pd
from tqdm import tqdm
import os
import torch
from torch import nn
from torch.nn import functional as F
import torch_geometric as pyg
from torch_geometric.loader import DataLoader
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
from pyteomics.mass import Composition
from pytorch_lightning import seed_everything
from pandarallel import pandarallel
from multiprocessing import cpu_count
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from src.smiles import from_mol
from src.gnn import add_virtual_node, graph_laplacian
from src.graff import GrAFF
from src.graff import (
    build_covariates,
    ce_ids,
    instruments,
    isotope_types,
    pfas_precursor_types,
    precursor_types,
)


################################################################
# greedy product/loss vocabulary selection
################################################################

def learn_vocabulary(df, vocab_size=None):
    annots = df[['InChIKey','products','losses','intensities','peaks']].copy()
    # equally split peak height among all compatible annotations
    annots['intensities'] = annots[['intensities','peaks']].apply(
        lambda item: item.intensities[item.peaks] / item.intensities.sum() \
                     / np.bincount(item.peaks)[item.peaks], axis=1)
    annots = annots.drop(columns=['peaks']).explode(['products','losses','intensities'])
    annots = annots.groupby(['InChIKey','products','losses']).sum()
    annots['intensities'] /= annots['intensities'].sum()
    annots = annots.reset_index()

    # separately rank products and losses by how much peak height each explains
    products = annots.groupby('products')['intensities'].sum().sort_values()[::-1].to_frame()
    products['kind'] = 'product'
    losses = annots.groupby('losses')['intensities'].sum().sort_values()[::-1].to_frame()
    losses['kind'] = 'loss'

    # take top vocab_size of either type
    vocab = pd.concat([products,losses],axis=0).sort_values('intensities',ascending=False)
    vocab.index.name = 'formula'
    vocab = vocab.reset_index()
    
    if vocab_size:
        vocab = vocab.head(vocab_size)
    
    pt = Chem.GetPeriodicTable()
    mws = {pt.GetElementSymbol(n): pt.GetMostCommonIsotopeMass(n) for n in range(1,119)}

    def formula_mz(formula):
        if not formula:
            return 0.0
        comp = Composition(formula=formula)
        return sum(mws[a] * n for a, n in comp.items())

    vocab['mz'] = vocab['formula'].map(formula_mz)
    
    return vocab


def formula_mz_table(formulas):
    pt = Chem.GetPeriodicTable()
    mws = {pt.GetElementSymbol(n): pt.GetMostCommonIsotopeMass(n) for n in range(1, 119)}

    def formula_mz(formula):
        if not formula:
            return 0.0
        comp = Composition(formula=formula)
        return sum(mws[a] * n for a, n in comp.items())

    return formulas.map(formula_mz)


def aggregate_pfas_formula_intensities(df):
    """Sum normalized peak-intensity fractions per (formula, kind) on the train split."""
    annots = df[['InChIKey', 'products', 'losses', 'intensities', 'peaks']].copy()
    annots['intensities'] = annots[['intensities', 'peaks']].apply(
        lambda item: item.intensities[item.peaks] / item.intensities.sum()
        / np.bincount(item.peaks)[item.peaks],
        axis=1,
    )
    annots = annots.drop(columns=['peaks']).explode(['products', 'losses', 'intensities'])

    products = (
        annots.groupby('products')['intensities']
        .sum()
        .to_frame()
    )
    products['kind'] = 'product'
    products.index.name = 'formula'

    losses = (
        annots.groupby('losses')['intensities']
        .sum()
        .to_frame()
    )
    losses['kind'] = 'loss'
    losses.index.name = 'formula'

    ranked = pd.concat([products, losses], axis=0).reset_index()
    ranked = ranked[ranked['formula'].astype(str).str.len() > 0]
    return ranked


def rank_nist_vocab_by_pfas_intensity(nist_vocab, df):
    """Attach PFAS train intensity to each NIST (formula, kind) row and sort descending."""
    pfas = aggregate_pfas_formula_intensities(df)
    pfas_lut = {
        (row.formula, row.kind): row.intensities
        for row in pfas.itertuples(index=False)
    }
    ranked = nist_vocab.copy()
    ranked['pfas_intensity'] = ranked.apply(
        lambda row: pfas_lut.get((row.formula, row.kind), 0.0),
        axis=1,
    )
    return ranked.sort_values(
        ['pfas_intensity', 'formula', 'kind'],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def prune_nist_vocabulary(nist_vocab, df, keep_n):
    """Keep the top *keep_n* NIST entries by PFAS train annotation intensity."""
    if keep_n is None or keep_n <= 0:
        return nist_vocab.reset_index(drop=True)
    ranked = rank_nist_vocab_by_pfas_intensity(nist_vocab, df)
    if keep_n >= len(ranked):
        return ranked.drop(columns=['pfas_intensity']).reset_index(drop=True)
    pruned = ranked.head(keep_n).drop(columns=['pfas_intensity'])
    n_zero = int((ranked.head(keep_n)['pfas_intensity'] == 0).sum())
    print(
        f'Pruned NIST vocab: kept {keep_n}/{len(nist_vocab)} entries '
        f'({n_zero} with zero PFAS train intensity)',
        flush=True,
    )
    return pruned.reset_index(drop=True)


def learn_extension_vocabulary(df, nist_vocab, max_extensions=None, exclude_keys=None):
    """Rank (formula, kind) pairs in PFAS train absent from *exclude_keys* (default: full NIST)."""
    if exclude_keys is None:
        exclude_keys = set(zip(nist_vocab['formula'], nist_vocab['kind']))
    else:
        exclude_keys = set(exclude_keys)

    annots = df[['InChIKey', 'products', 'losses', 'intensities', 'peaks']].copy()
    annots['intensities'] = annots[['intensities', 'peaks']].apply(
        lambda item: item.intensities[item.peaks] / item.intensities.sum()
        / np.bincount(item.peaks)[item.peaks],
        axis=1,
    )
    annots = annots.drop(columns=['peaks']).explode(['products', 'losses', 'intensities'])

    products = (
        annots.groupby('products')['intensities']
        .sum()
        .sort_values()[::-1]
        .to_frame()
    )
    products['kind'] = 'product'
    products.index.name = 'formula'

    losses = (
        annots.groupby('losses')['intensities']
        .sum()
        .sort_values()[::-1]
        .to_frame()
    )
    losses['kind'] = 'loss'
    losses.index.name = 'formula'

    extensions = pd.concat([products, losses], axis=0).sort_values('intensities', ascending=False)
    extensions = extensions.reset_index()
    extensions = extensions[
        ~extensions.apply(lambda row: (row['formula'], row['kind']) in exclude_keys, axis=1)
    ]
    extensions = extensions[extensions['formula'].astype(str).str.len() > 0]
    extensions['mz'] = formula_mz_table(extensions['formula'])
    extensions = extensions.reset_index(drop=True)
    if max_extensions is not None and max_extensions > 0:
        extensions = extensions.head(max_extensions)
    return extensions


def merge_union_vocabulary(nist_vocab, extensions):
    """Keep NIST index order; append PFAS-only entries at the end."""
    union = pd.concat(
        [nist_vocab.reset_index(drop=True), extensions.reset_index(drop=True)],
        ignore_index=True,
    )
    return union


def _torch_load_checkpoint(checkpoint_path):
    """Load a Lightning checkpoint; compatible with torch 1.12 and 2.6+."""
    try:
        return torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location='cpu')


def load_nist_vocab_from_checkpoint(checkpoint_path):
    ckpt = _torch_load_checkpoint(checkpoint_path)
    hp = ckpt.get('hyper_parameters', ckpt.get('hparams', {}))
    if 'vocab' not in hp:
        raise KeyError(f'No vocab in checkpoint hyperparameters: {checkpoint_path}')
    vocab = hp['vocab']
    if not isinstance(vocab, pd.DataFrame):
        vocab = pd.DataFrame(vocab)
    return vocab.reset_index(drop=True)


def _nist_vocab_row_lut(nist_vocab):
    """Map (formula, kind) to NIST clf row index (row 0 reserved for pad)."""
    return {
        (row.formula, row.kind): i + 1
        for i, row in nist_vocab.reset_index(drop=True).iterrows()
    }


def _transfer_clf_by_formula(model_sd, pretrained_sd, target_vocab, nist_vocab):
    """Copy clf rows where PFAS vocab entries match NIST (formula, kind) keys."""
    nist_lut = _nist_vocab_row_lut(nist_vocab)
    n_old_clf = pretrained_sd['clf.weight'].shape[0]
    matched = 0

    for j, row in target_vocab.reset_index(drop=True).iterrows():
        nist_row = nist_lut.get((row.formula, row.kind))
        pfas_row = j + 1
        if nist_row is None or nist_row >= n_old_clf:
            continue
        model_sd['clf.weight'][pfas_row] = pretrained_sd['clf.weight'][nist_row]
        model_sd['clf.bias'][pfas_row] = pretrained_sd['clf.bias'][nist_row]
        matched += 1

    return matched


def parse_ce_loss_weights(spec, *, ce_id_values):
    """Parse '0:3.0,1:0.8' into a list indexed by CE_ID."""
    if not spec:
        return None
    lut = [1.0] * (max(ce_id_values) + 1)
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        key, value = part.split(':')
        ce_id = int(key.strip())
        if ce_id not in ce_id_values:
            raise ValueError(f'Unknown CE_ID {ce_id} in --ce_loss_weights (expected {ce_id_values})')
        lut[ce_id] = float(value.strip())
    return lut


def transfer_graff_weights(
    model,
    checkpoint_path,
    *,
    target_vocab,
    vocab_mode='union',
    transfer_mode='auto',
):
    """
    Transfer NIST-pretrained weights into a PFAS model.

    - GNN / decoder / isotope_shift: copy when shapes match
    - cov_emb / cov_film / ce_embed: skip (NIST 7-dim vs PFAS 11-dim; CE-FiLM/embed are PFAS-only)
    - clf (union): copy first NIST rows; append random rows for extensions
    - clf (pfas / clf_map): copy rows matched by (formula, kind)
    - clf (backbone): skip clf entirely
    - vocab buffers: skip (rebuilt from target vocab)
    """
    ckpt = _torch_load_checkpoint(checkpoint_path)
    pretrained_sd = ckpt['state_dict']
    model_sd = model.state_dict()
    nist_vocab = load_nist_vocab_from_checkpoint(checkpoint_path)

    n_old_clf = pretrained_sd['clf.weight'].shape[0]
    n_new_clf = model_sd['clf.weight'].shape[0]

    if transfer_mode == 'auto':
        transfer_mode = 'union' if vocab_mode == 'union' else 'clf_map'

    transferred = []
    skipped = []
    clf_copied = 0
    clf_mode = transfer_mode

    for key, value in pretrained_sd.items():
        if key.startswith('cov_emb.') or key.startswith('cov_film.') or key.startswith('ce_embed.') or key.startswith('vocab_mzs') or key.startswith('vocab_kinds'):
            skipped.append(key)
            continue
        if key not in model_sd:
            skipped.append(key)
            continue

        if key in ('clf.weight', 'clf.bias'):
            continue

        if model_sd[key].shape != value.shape:
            skipped.append(key)
            continue

        model_sd[key] = value
        transferred.append(key)

    if clf_mode == 'backbone':
        print('clf: skipped (backbone-only transfer)', flush=True)
    elif clf_mode == 'union':
        if n_new_clf < n_old_clf:
            raise ValueError(
                f'Union vocab ({n_new_clf - 1} entries) is smaller than NIST '
                f'checkpoint vocab ({n_old_clf - 1} entries).'
            )
        model_sd['clf.weight'][:n_old_clf] = pretrained_sd['clf.weight']
        model_sd['clf.bias'][:n_old_clf] = pretrained_sd['clf.bias']
        clf_copied = n_old_clf - 1
        print(
            f'clf: copied first {clf_copied} NIST rows; '
            f'{n_new_clf - n_old_clf} extension rows left randomly initialized',
            flush=True,
        )
    elif clf_mode == 'clf_map':
        clf_copied = _transfer_clf_by_formula(
            model_sd, pretrained_sd, target_vocab, nist_vocab,
        )
        print(
            f'clf: mapped {clf_copied}/{len(target_vocab)} PFAS vocab rows from NIST '
            f'by (formula, kind)',
            flush=True,
        )
    else:
        raise ValueError(f'Unknown transfer_mode: {transfer_mode}')

    model.load_state_dict(model_sd)
    print(
        f'Loaded checkpoint: {len(transferred)} backbone tensors copied, '
        f'{len(skipped)} skipped (cov_emb / buffers / clf handled separately).',
        flush=True,
    )

################################################################
# load hyperparameters from command line
################################################################

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('df_path')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--vocab_size', type=int, default=10000)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--learning_rate', type=float, default=5e-4)
parser.add_argument('--grad_clipping', type=int, default=100)
parser.add_argument('--max_epochs', type=int, default=100)
parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--precision', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--encoder_dim', type=int, default=512)
parser.add_argument('--decoder_dim', type=int, default=1024)
parser.add_argument('--encoder_depth', type=int, default=6)
parser.add_argument('--decoder_depth', type=int, default=2)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--num_eigs', type=int, default=8)
parser.add_argument('--eig_dim', type=int, default=32)
parser.add_argument('--eig_depth', type=int, default=2)
parser.add_argument('--min_probability', type=float, default=0)
parser.add_argument('--min_mz', type=float, default=0)
parser.add_argument('--subsample', type=int, default=0)
parser.add_argument('--cache_path', type=str, default=None)
parser.add_argument(
    '--dataset',
    choices=['nist', 'pfas'],
    default=None,
    help='Covariate layout (default: auto-detect from .pkl columns)',
)
parser.add_argument(
    '--checkpoint',
    type=str,
    default=None,
    help='NIST (or compatible) Lightning checkpoint for partial weight transfer',
)
parser.add_argument(
    '--vocab_mode',
    choices=['pfas', 'union'],
    default=None,
    help='Vocabulary: pfas-only from train split, or union with NIST checkpoint vocab '
         '(default: union when --checkpoint is set, else pfas for PFAS data)',
)
parser.add_argument(
    '--pfas_extension_size',
    type=int,
    default=2000,
    help='Max PFAS-only (formula, kind) entries appended in union vocab mode',
)
parser.add_argument(
    '--nist_vocab_keep',
    type=int,
    default=None,
    help='Union mode: keep this many NIST (formula, kind) rows ranked by PFAS train '
         'intensity; removed slots can be filled via --pfas_extension_size. '
         'Use with --transfer_mode clf_map.',
)
parser.add_argument(
    '--transfer_mode',
    choices=['auto', 'union', 'clf_map', 'backbone'],
    default='auto',
    help='Weight transfer for clf: auto (union slice or PFAS formula map), '
         'union (NIST row prefix), clf_map (match formula/kind), backbone (skip clf)',
)
parser.add_argument(
    '--freeze_backbone',
    action='store_true',
    help='Freeze GNN/decoder/isotope_shift; train cov_emb + clf only (use with checkpoint).',
)
parser.add_argument(
    '--clf_lr',
    type=float,
    default=None,
    help='Learning rate for clf (default: same as --learning_rate, or 1e-3 with --freeze_backbone).',
)
parser.add_argument(
    '--cov_emb_lr',
    type=float,
    default=None,
    help='Learning rate for cov_emb (default: same as --learning_rate).',
)
parser.add_argument(
    '--early_stopping_patience',
    type=int,
    default=0,
    help='Stop if val/loss does not improve for this many epochs (0 = disabled).',
)
parser.add_argument(
    '--early_stopping_min_delta',
    type=float,
    default=0.0,
    help='Minimum val/loss decrease to count as improvement for early stopping.',
)
parser.add_argument(
    '--cov_conditioning',
    choices=['add', 'film_decoder', 'both'],
    default='add',
    help='PFAS covariate fusion: add (default, optional wide MLP via --cov_emb_dim), '
         'film_decoder (compact cov_emb + FiLM on decoder), '
         'both (768-d hidden cov_emb + FiLM).',
)
parser.add_argument(
    '--cov_emb_dim',
    type=int,
    default=None,
    help='Hidden width of cov_emb MLP (default: encoder_dim). Try 768 with encoder_dim=512.',
)
parser.add_argument(
    '--ce_loss_weights',
    type=str,
    default=None,
    help='PFAS only: per-CE_ID training loss weights, e.g. 0:3.0,1:0.8,2:0.5,3:2.5. '
         'Validation loss stays unweighted.',
)
parser.add_argument(
    '--ce_embed_dim',
    type=int,
    default=None,
    help='PFAS only: learnable CE_ID embedding size (replaces CE one-hot in covariates). '
         'Try 32 with film_decoder.',
)
args = parser.parse_args()

if args.freeze_backbone and args.clf_lr is None:
    args.clf_lr = 1e-3
    print('freeze_backbone: default clf_lr=1e-3', flush=True)

if args.cov_conditioning != 'add':
    print(
        f'Covariate conditioning: {args.cov_conditioning} '
        f'(cov_emb_dim={args.cov_emb_dim or args.encoder_dim})',
        flush=True,
    )

seed_everything(args.seed, workers=True)
use_parallel = args.num_workers > 0
if use_parallel:
    pandarallel.initialize(progress_bar=False, verbose=0, nb_workers=args.num_workers)
else:
    print('num_workers=0: sequential pandas apply for preprocessing', flush=True)


def frame_apply(df, func, axis=1):
    if use_parallel:
        return df.parallel_apply(func, axis=axis)
    return df.apply(func, axis=axis)

################################################################
# load prepared dataframe
################################################################

df = pd.read_pickle(args.df_path)
df = df.query('split!=""')

if args.dataset is None:
    if 'CE_ID' in df.columns and 'Dissociation_id' in df.columns:
        args.dataset = 'pfas'
    else:
        args.dataset = 'nist'

if args.dataset == 'pfas':
    active_precursor_types = pfas_precursor_types
    df = df[df['Precursor_type'].isin(active_precursor_types)]
    df = df[df['Dissociation_id'].notna() & df['CE_ID'].notna()]
else:
    active_precursor_types = precursor_types
    df = df[df['Precursor_type'].isin(active_precursor_types)]
    if 'Instrument' in df.columns:
        df = df[df['Instrument'].isin(instruments)]

print(f'Dataset mode: {args.dataset} ({len(df)} spectra)')

if args.ce_embed_dim:
    if args.dataset != 'pfas':
        raise ValueError('--ce_embed_dim is only supported for PFAS training.')
    print(f'CE embedding: dim={args.ce_embed_dim} (replaces CE one-hot)', flush=True)

if args.vocab_mode is None:
    args.vocab_mode = 'union' if args.checkpoint else 'pfas'

if args.checkpoint and args.vocab_mode == 'pfas' and args.transfer_mode == 'auto':
    print(
        'PFAS vocab + checkpoint: clf rows will be mapped by (formula, kind); '
        'unmatched rows stay randomly initialized.',
        flush=True,
    )
elif args.checkpoint and args.vocab_mode == 'pfas' and args.transfer_mode == 'union':
    raise ValueError('transfer_mode=union requires vocab_mode=union.')

if args.vocab_mode == 'union' and not args.checkpoint:
    raise ValueError('--vocab_mode union requires --checkpoint with a NIST vocab.')

if args.nist_vocab_keep is not None and args.vocab_mode != 'union':
    raise ValueError('--nist_vocab_keep requires --vocab_mode union.')

if args.nist_vocab_keep is not None and args.transfer_mode == 'union':
    raise ValueError(
        '--nist_vocab_keep reshuffles NIST rows; use --transfer_mode clf_map, not union.',
    )

if args.nist_vocab_keep is not None and args.transfer_mode == 'auto':
    args.transfer_mode = 'clf_map'
    print(
        'Pruned NIST union: auto-selected transfer_mode=clf_map',
        flush=True,
    )

if args.subsample:
    df = df.sample(n=args.subsample, random_state=args.seed)

train_df = df.query('split=="train"')

################################################################
# compute the fixed vocabulary from training spectra
################################################################

if args.vocab_mode == 'union':
    print(f'Building union vocabulary (NIST checkpoint + PFAS train)... ', end='', flush=True)
    full_nist_vocab = load_nist_vocab_from_checkpoint(args.checkpoint)
    nist_vocab = prune_nist_vocabulary(
        full_nist_vocab, train_df, args.nist_vocab_keep,
    )
    kept_nist_keys = set(zip(nist_vocab['formula'], nist_vocab['kind']))
    extensions = learn_extension_vocabulary(
        train_df,
        full_nist_vocab,
        max_extensions=args.pfas_extension_size,
        exclude_keys=kept_nist_keys,
    )
    vocab = merge_union_vocabulary(nist_vocab, extensions)
    vocab_size = len(vocab)
    n_pruned = len(full_nist_vocab) - len(nist_vocab)
    print(
        f'{vocab_size} formulas ({len(nist_vocab)} NIST'
        f'{f", pruned {n_pruned}" if n_pruned else ""}'
        f' + {len(extensions)} PFAS-only)',
        flush=True,
    )
else:
    print(f'Selecting vocabulary (K={args.vocab_size})... ', end='', flush=True)
    vocab = learn_vocabulary(train_df, args.vocab_size)
    vocab_size = len(vocab)
    if vocab_size < args.vocab_size:
        print(f'using {vocab_size} formulas (requested {args.vocab_size})... ', end='', flush=True)
    print('done', flush=True)

assert vocab_size > 0, 'Empty vocabulary!'

################################################################
# generate indices into the fixed vocabulary for each peak
################################################################

# reserve zero index for the pad (row position in vocab, not pandas index label)
vocab = vocab.reset_index(drop=True)
product_lut = {
    row.formula: i + 1 for i, row in vocab.iterrows() if row.kind == 'product'
}
loss_lut = {
    row.formula: i + 1 for i, row in vocab.iterrows() if row.kind == 'loss'
}
iso_lut = {v:i for i,v in enumerate(sorted(isotope_types))}

def index_peaks(item):
    peaks = pd.DataFrame(item[['products','losses','isotopes','peaks']]).T
    peaks = peaks.explode([*peaks.columns])
    peaks['product_idx'] = peaks['products'].map(product_lut)
    peaks['loss_idx'] = peaks['losses'].map(loss_lut)
    peaks['isotope_idx'] = peaks['isotopes'].map(iso_lut)
    # don't predict rare higher isotope peaks
    peaks = peaks.dropna(subset=['isotope_idx'])
    peaks = peaks[['product_idx','loss_idx','isotope_idx','peaks']]
    # drop peaks not in either product or loss vocabulary
    peaks = peaks.loc[~(peaks['product_idx'].isna() & peaks['loss_idx'].isna())]
    # if a product cannot be explained as a loss, or vice-versa,
    # (which should usually be the case), assign the missing index to the pad
    peaks = peaks.fillna(0).astype(int)
    return peaks.values.T

# this is not necessary if cached
print(f'Indexing peaks... ',end='')
df['product_idx'], df['loss_idx'], df['isotope_idx'], df['peak_idx'] = (
    zip(*frame_apply(df, index_peaks, axis=1))
)
print('done')

# remove any (rare!) spectra from training or validation that have no explanations
df.loc[(df['product_idx'].str.len()==0)&(df['split'].isin(['train','val'])),'split'] = ''

################################################################
# precompute graph features
################################################################

max_peaks = df['intensities'].str.len().max()
max_annots = df['product_idx'].str.len().max()

def pad1d(x, n):
    return F.pad(x, (0, n-len(x)))

def featurize_spectrum(item):
    # graph features
    mol = Chem.MolFromSmiles(item.SMILES)
    g = from_mol(mol)
    g = graph_laplacian(g, args.num_eigs)
    g = add_virtual_node(g)
    # must pad the eigenfeatures for the virtual node
    eig_pad = torch.zeros(g.num_nodes-g.eigvecs.shape[0],g.eigvecs.shape[1],
                         dtype=g.eigvecs.dtype,device=g.eigvecs.device)
    g.eigvecs = torch.cat([g.eigvecs,eig_pad],0)

    if args.dataset == 'pfas':
        covariates = build_covariates(
            dataset='pfas',
            precursor_type=item.Precursor_type,
            has_isotopes=item.has_isotopes,
            dissociation_id=int(item.Dissociation_id),
            ce_id=int(item.CE_ID),
            precursor_types_list=active_precursor_types,
            ce_ids_list=ce_ids,
            ce_embed_dim=args.ce_embed_dim,
        )
    else:
        covariates = build_covariates(
            dataset='nist',
            precursor_type=item.Precursor_type,
            has_isotopes=item.has_isotopes,
            nce=item.NCE,
            instrument=item.Instrument,
            precursor_types_list=active_precursor_types,
            instruments_list=instruments,
        )
    
    # if an annotation matched both the product and loss vocabularies, it's doubled
    double_counted = np.zeros(vocab_size+1,dtype=bool)
    double_counted[item.product_idx] |= (item.loss_idx > 0)
    double_counted[item.loss_idx] |= (item.product_idx > 0)
    double_counted[0] = False

    g.spectrum = str(item.Spectrum)
    g.split = item.split
    g.ce_id = torch.tensor([int(item.CE_ID)], dtype=torch.long)
    g.precursor_mz = item.PrecursorMZ
    g.covariates = torch.FloatTensor(covariates).view(1,-1)
    g.product_idx = pad1d(torch.LongTensor(item.product_idx), max_annots).view(1,-1)
    g.loss_idx = pad1d(torch.LongTensor(item.loss_idx), max_annots).view(1,-1)
    g.peak_idx = pad1d(torch.LongTensor(item.peak_idx), max_annots).view(1,-1)
    g.isotope_idx = pad1d(torch.LongTensor(item.isotope_idx), max_annots).view(1,-1)
    g.mzs = pad1d(torch.FloatTensor(item.mzs), max_peaks).view(1,-1)
    g.intensities = pad1d(torch.FloatTensor(item.intensities), max_peaks).view(1,-1)
    g.double_counted = torch.BoolTensor(double_counted).view(1,-1)

    return g

print('Featurizing spectra... ',end='')
if args.cache_path:
    if os.path.exists(args.cache_path):
        items = pd.read_pickle(args.cache_path)
    else:
        items = frame_apply(df, featurize_spectrum, axis=1)
        items.to_pickle(args.cache_path)
else:
    items = frame_apply(df, featurize_spectrum, axis=1)
print('done')

################################################################
# split data
################################################################

datasets = {}
for split in ['train','val','test']:
    datasets[split] = [x for x in items if x.split == split]

loaders = {}
for split in datasets:
    loaders[split] = DataLoader(
        datasets[split],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=split=='train',
        drop_last=split=='train'
    )

################################################################
# fit model
################################################################

callbacks = [
    ModelCheckpoint(
        monitor='val/loss',
        mode='min',
        save_top_k=1,
    ),
]
if args.early_stopping_patience > 0:
    callbacks.append(
        EarlyStopping(
            monitor='val/loss',
            mode='min',
            patience=args.early_stopping_patience,
            min_delta=args.early_stopping_min_delta,
            verbose=True,
        )
    )
    print(
        f'Early stopping: patience={args.early_stopping_patience}, '
        f'min_delta={args.early_stopping_min_delta}',
        flush=True,
    )

trainer = pl.Trainer(
    accelerator='gpu' if args.gpus else 'cpu', 
    devices=args.gpus if args.gpus else None,
    strategy='ddp' if args.gpus>1 else None,
    precision=args.precision,
    max_epochs=args.max_epochs,
    gradient_clip_val=args.grad_clipping,
    logger=TensorBoardLogger(
        'lightning_logs', 
        default_hp_metric=False,
        name='graff'
    ),
    callbacks=callbacks,
)

ce_loss_weight_lut = None
if args.ce_loss_weights:
    if args.dataset != 'pfas':
        raise ValueError('--ce_loss_weights is only supported for PFAS training.')
    ce_loss_weight_lut = parse_ce_loss_weights(
        args.ce_loss_weights,
        ce_id_values=ce_ids,
    )
    print(f'CE loss weights (train only): {ce_loss_weight_lut}', flush=True)

_graff_hparams = {
    k: v for k, v in args.__dict__.items()
    if k not in {
        'df_path', 'dataset', 'checkpoint', 'vocab_mode',
        'pfas_extension_size', 'nist_vocab_keep', 'transfer_mode',
        'early_stopping_patience', 'early_stopping_min_delta',
        'ce_loss_weights',
    }
}

model = GrAFF(
    vocab=vocab,
    precursor_types=active_precursor_types,
    instruments=instruments if args.dataset == 'nist' else [],
    dataset=args.dataset,
    ce_ids=ce_ids if args.dataset == 'pfas' else [],
    ce_loss_weights=ce_loss_weight_lut,
    **_graff_hparams,
)

if args.checkpoint:
    transfer_graff_weights(
        model,
        args.checkpoint,
        target_vocab=vocab,
        vocab_mode=args.vocab_mode,
        transfer_mode=args.transfer_mode,
    )

trainer.fit(model, loaders['train'], loaders['val'])
