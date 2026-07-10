#!/usr/bin/env python3
"""Preview pruned-union vocabulary stats without importing the full training stack."""
import argparse

import numpy as np
import pandas as pd
import torch
from pyteomics.mass import Composition
from rdkit import Chem


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
    annots = df[['InChIKey', 'products', 'losses', 'intensities', 'peaks']].copy()
    annots['intensities'] = annots[['intensities', 'peaks']].apply(
        lambda item: item.intensities[item.peaks] / item.intensities.sum()
        / np.bincount(item.peaks)[item.peaks],
        axis=1,
    )
    annots = annots.drop(columns=['peaks']).explode(['products', 'losses', 'intensities'])

    products = annots.groupby('products')['intensities'].sum().to_frame()
    products['kind'] = 'product'
    products.index.name = 'formula'

    losses = annots.groupby('losses')['intensities'].sum().to_frame()
    losses['kind'] = 'loss'
    losses.index.name = 'formula'

    ranked = pd.concat([products, losses], axis=0).reset_index()
    return ranked[ranked['formula'].astype(str).str.len() > 0]


def rank_nist_vocab_by_pfas_intensity(nist_vocab, df):
    pfas = aggregate_pfas_formula_intensities(df)
    pfas_lut = {(r.formula, r.kind): r.intensities for r in pfas.itertuples(index=False)}
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
    ranked = rank_nist_vocab_by_pfas_intensity(nist_vocab, df)
    if keep_n >= len(ranked):
        return ranked.drop(columns=['pfas_intensity']).reset_index(drop=True)
    pruned = ranked.head(keep_n).drop(columns=['pfas_intensity'])
    n_zero = int((ranked.head(keep_n)['pfas_intensity'] == 0).sum())
    print(
        f'Pruned NIST vocab: kept {keep_n}/{len(nist_vocab)} entries '
        f'({n_zero} with zero PFAS train intensity)',
    )
    return pruned.reset_index(drop=True)


def learn_extension_vocabulary(df, exclude_keys, max_extensions=None):
    annots = df[['InChIKey', 'products', 'losses', 'intensities', 'peaks']].copy()
    annots['intensities'] = annots[['intensities', 'peaks']].apply(
        lambda item: item.intensities[item.peaks] / item.intensities.sum()
        / np.bincount(item.peaks)[item.peaks],
        axis=1,
    )
    annots = annots.drop(columns=['peaks']).explode(['products', 'losses', 'intensities'])

    products = annots.groupby('products')['intensities'].sum().sort_values()[::-1].to_frame()
    products['kind'] = 'product'
    products.index.name = 'formula'

    losses = annots.groupby('losses')['intensities'].sum().sort_values()[::-1].to_frame()
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


def load_nist_vocab_from_checkpoint(checkpoint_path):
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location='cpu')
    hp = ckpt.get('hyper_parameters', ckpt.get('hparams', {}))
    vocab = hp['vocab']
    if not isinstance(vocab, pd.DataFrame):
        vocab = pd.DataFrame(vocab)
    return vocab.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pkl_path')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--nist_vocab_keep', type=int, default=5000)
    parser.add_argument('--pfas_extension_size', type=int, default=5000)
    args = parser.parse_args()

    df = pd.read_pickle(args.pkl_path).query('split != ""')
    train_df = df.query('split == "train"')

    full_nist = load_nist_vocab_from_checkpoint(args.checkpoint)
    nist = prune_nist_vocabulary(full_nist, train_df, args.nist_vocab_keep)
    kept = set(zip(nist['formula'], nist['kind']))
    ext = learn_extension_vocabulary(
        train_df, kept, max_extensions=args.pfas_extension_size,
    )
    vocab = pd.concat([nist, ext], ignore_index=True)

    pfas = aggregate_pfas_formula_intensities(train_df)
    vocab_keys = set(zip(vocab['formula'], vocab['kind']))
    in_vocab = sum(r.intensities for r in pfas.itertuples(index=False) if (r.formula, r.kind) in vocab_keys)
    total = pfas['intensities'].sum()

    removed_nist = full_nist[
        ~full_nist.apply(lambda r: (r.formula, r.kind) in kept, axis=1)
    ]
    ext_keys = set(zip(ext['formula'], ext['kind']))
    readded = removed_nist[
        removed_nist.apply(lambda r: (r.formula, r.kind) in ext_keys, axis=1)
    ]

    print(f'Full NIST vocab:     {len(full_nist)}')
    print(f'Kept NIST:           {len(nist)}')
    print(f'PFAS extensions:     {len(ext)}')
    print(f'Union total:         {len(vocab)}')
    print(f'Train intensity in vocab: {100 * in_vocab / total:.1f}%')
    print(f'Removed NIST re-added via PFAS slots: {len(readded)}')
    print('\nTop 10 PFAS extensions:')
    print(ext.head(10).to_string(index=False))


if __name__ == '__main__':
    main()
