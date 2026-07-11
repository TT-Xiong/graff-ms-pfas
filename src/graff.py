import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
import torch_geometric as pyg
from torch_scatter import scatter_logsumexp

from .gnn import GINE, ResBlock, CanonicalOneHot, SignNet

atom_types = sorted(['C','H','N','O','P','S','F','Cl','Br','I'])
isotope_types = [0,1,2]
neutron_mass = 1.008665

# NIST training covariates
precursor_types = ['[M+H]+','[M-H]-']
instruments = [
    'Orbitrap Fusion Lumos',
    'Thermo Finnigan Elite Orbitrap',
    'Thermo Finnigan Velos Orbitrap'
]

# PFAS training covariates (preprocess-pfas.py)
pfas_precursor_types = ['[M-H]-', '[M]+', '[M+H]+', '[M-2H]-']
dissociation_types = ['HCD', 'CID']
dissociation_id = {'HCD': 0, 'CID': 1}
ce_ids = [0, 1, 2, 3]
ce_bins = [15, 30, 45, 60]


def normalize_ce_value(
    *,
    ce_id=None,
    eV=None,
    ce_max_ev=60.0,
    ce_clip_min=None,
    ce_clip_max=None,
) -> float:
    """Map collision energy to a monotonic scalar, default eV / ce_max_ev."""
    if eV is not None:
        value = float(eV)
        if ce_clip_min is not None:
            value = max(float(ce_clip_min), value)
        if ce_clip_max is not None:
            value = min(float(ce_clip_max), value)
        return value / float(ce_max_ev)
    if ce_id is not None:
        return float(ce_bins[int(ce_id)]) / float(ce_max_ev)
    raise ValueError("Either ce_id or eV is required")


def compute_covariates_dim(
    dataset,
    *,
    precursor_types,
    instruments,
    ce_ids,
    ce_embed_dim=None,
    ce_encoding='onehot',
):
    if dataset == 'pfas':
        if ce_embed_dim:
            ce_part = 0
        elif ce_encoding == 'continuous':
            ce_part = 1
        else:
            ce_part = len(ce_ids)
        return len(dissociation_types) + ce_part + len(precursor_types) + 1
    return len(instruments) + len(precursor_types) + 1 + 1


def build_covariates(
    *,
    dataset='nist',
    precursor_type,
    has_isotopes,
    nce=None,
    instrument=None,
    dissociation_id=None,
    ce_id=None,
    precursor_types_list=None,
    instruments_list=None,
    ce_ids_list=None,
    ce_embed_dim=None,
    ce_encoding='onehot',
    eV=None,
    ce_max_ev=60.0,
    ce_clip_min=None,
    ce_clip_max=None,
):
    """Build the covariate vector consumed by ``GrAFF.cov_emb``."""
    if dataset == 'pfas':
        pt_list = precursor_types_list or pfas_precursor_types
        ce_list = ce_ids_list or ce_ids
        diss_idx = int(dissociation_id)
        ce_idx = int(ce_id)
        diss_onehot = [0.0] * len(dissociation_types)
        diss_onehot[diss_idx] = 1.0
        pt_onehot = [0.0] * len(pt_list)
        pt_onehot[pt_list.index(precursor_type)] = 1.0
        if ce_embed_dim:
            return np.array([*diss_onehot, *pt_onehot, float(has_isotopes)], dtype=np.float32)
        if ce_encoding == 'continuous':
            ce_value = normalize_ce_value(
                ce_id=ce_idx,
                eV=eV,
                ce_max_ev=ce_max_ev,
                ce_clip_min=ce_clip_min,
                ce_clip_max=ce_clip_max,
            )
            return np.array([*diss_onehot, ce_value, *pt_onehot, float(has_isotopes)], dtype=np.float32)
        ce_onehot = [0.0] * len(ce_list)
        ce_onehot[ce_idx] = 1.0
        return np.array([*diss_onehot, *ce_onehot, *pt_onehot, float(has_isotopes)], dtype=np.float32)

    inst_list = instruments_list or instruments
    pt_list = precursor_types_list or precursor_types
    inst_onehot = [0.0] * len(inst_list)
    inst_onehot[inst_list.index(instrument)] = 1.0
    pt_onehot = [0.0] * len(pt_list)
    pt_onehot[pt_list.index(precursor_type)] = 1.0
    return np.array([*inst_onehot, *pt_onehot, float(nce), float(has_isotopes)], dtype=np.float32)

class GrAFF(pl.LightningModule):
    def __init__(
        self,
        *,
        vocab,
        encoder_dim,
        decoder_dim,
        encoder_depth,
        decoder_depth,
        num_eigs,
        eig_dim,
        eig_depth,
        dropout,
        learning_rate,
        weight_decay,
        precursor_types,
        instruments,
        dataset='nist',
        dissociation_types=None,
        ce_ids=None,
        min_probability,
        min_mz,
        freeze_backbone=False,
        clf_lr=None,
        cov_emb_lr=None,
        cov_conditioning='add',
        cov_emb_dim=None,
        ce_loss_weights=None,
        ce_embed_dim=None,
        ce_encoding='onehot',
        ce_max_ev=60.0,
        ce_clip_min=None,
        ce_clip_max=None,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        self.dataset = dataset
        self.vocab = vocab
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.encoder_depth = encoder_depth
        self.decoder_depth = decoder_depth
        self.num_eigs = num_eigs
        self.eig_dim = eig_dim
        self.eig_depth = eig_depth
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.freeze_backbone = freeze_backbone
        self.clf_lr = clf_lr
        self.cov_emb_lr = cov_emb_lr
        self.cov_conditioning = cov_conditioning
        self.ce_encoding = ce_encoding
        self.ce_max_ev = float(ce_max_ev)
        self.ce_clip_min = ce_clip_min
        self.ce_clip_max = ce_clip_max
        self.use_ce_loss_weights = ce_loss_weights is not None
        if self.use_ce_loss_weights:
            self.register_buffer(
                'ce_loss_weight_lut',
                torch.tensor(ce_loss_weights, dtype=torch.float32),
            )
        if cov_emb_dim is not None:
            self.cov_emb_dim = cov_emb_dim
        elif cov_conditioning == 'both':
            self.cov_emb_dim = 768
        else:
            self.cov_emb_dim = encoder_dim

        if dataset == 'pfas':
            self.precursor_types = precursor_types or pfas_precursor_types
            self.instruments = instruments or []
            self.dissociation_types = dissociation_types or globals()['dissociation_types']
            self.ce_ids = ce_ids or globals()['ce_ids']
        else:
            self.precursor_types = precursor_types or globals()['precursor_types']
            self.instruments = instruments or globals()['instruments']
            self.dissociation_types = dissociation_types or []
            self.ce_ids = ce_ids or []

        covariates_dim = compute_covariates_dim(
            dataset,
            precursor_types=self.precursor_types,
            instruments=self.instruments,
            ce_ids=self.ce_ids,
            ce_embed_dim=ce_embed_dim,
            ce_encoding=ce_encoding,
        )
        vocab_size = len(vocab)
        self.vocab_size = vocab_size
        self.covariates_dim = covariates_dim
        self.ce_embed_dim = ce_embed_dim
        self.cov_in_dim = covariates_dim + (ce_embed_dim or 0)
        if ce_embed_dim:
            self.ce_embed = nn.Embedding(len(self.ce_ids), ce_embed_dim)
        else:
            self.ce_embed = None
        
        # inference time only
        self.min_probability = min_probability
        self.min_mz = min_mz
        
        vocab_mzs = vocab['mz'].values
        self.register_buffer('vocab_mzs', torch.FloatTensor(vocab_mzs))
        vocab_kinds = vocab['kind'].values
        self.register_buffer('vocab_kinds', torch.BoolTensor(vocab_kinds == 'product'))
        self.register_buffer('isotope_mzs', torch.FloatTensor(isotope_types) * neutron_mass)
        
        self.log_epsilon = -10
        
        # embedding layers
        extra_node_feats = {
            'is_virtual_node': {False: 0, True: 1},
        }
        extra_edge_feats = {
            'is_virtual_in_edge': {False: 0, True: 1},
            'is_virtual_out_edge': {False: 0, True: 1},
        }
        self.onehot = CanonicalOneHot(extra_node_feats, extra_edge_feats)
        self.node_emb = nn.Sequential(
            nn.Linear(self.onehot.node_dim, encoder_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, encoder_dim)
        )
        self.edge_emb = nn.Sequential(
            nn.Linear(self.onehot.edge_dim, encoder_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, encoder_dim)
        )
        
        if num_eigs > 0:
            self.signnet = SignNet(
                num_eigs=num_eigs,
                embed_dim=encoder_dim,
                rho_dim=encoder_dim,
                rho_depth=eig_depth,
                phi_dim=eig_dim,
                phi_depth=eig_depth,
                dropout=dropout
            )
        else:
            self.signnet = None
        
        self.encoder = GINE(
            node_dim=encoder_dim,
            edge_dim=encoder_dim,
            model_dim=encoder_dim, 
            model_depth=encoder_depth,
            dropout=dropout
        )
        
        if cov_conditioning == 'film_decoder':
            self.cov_emb = nn.Sequential(
                nn.Linear(self.cov_in_dim, encoder_dim),
                nn.SiLU(inplace=True),
                nn.Dropout(dropout),
                nn.LayerNorm(encoder_dim),
                nn.Linear(encoder_dim, encoder_dim),
            )
        else:
            self.cov_emb = nn.Sequential(
                nn.Linear(self.cov_in_dim, self.cov_emb_dim),
                nn.SiLU(inplace=True),
                nn.Dropout(dropout),
                nn.LayerNorm(self.cov_emb_dim),
                nn.Linear(self.cov_emb_dim, encoder_dim),
            )

        if cov_conditioning in ('film_decoder', 'both'):
            self.cov_film = nn.Sequential(
                nn.Linear(self.cov_in_dim, decoder_dim),
                nn.SiLU(inplace=True),
                nn.Linear(decoder_dim, decoder_dim * 2),
            )
        else:
            self.cov_film = None
        
        self.attn = nn.Linear(encoder_dim, 1)
        
        layers = []
        layers.append(nn.Linear(encoder_dim, decoder_dim))
        for _ in range(decoder_depth):
            layers.append(ResBlock(decoder_dim, dropout))
        self.decoder = nn.Sequential(*layers)
        
        self.isotope_shift = nn.Linear(decoder_dim, len(isotope_types))
        self.clf = nn.Linear(decoder_dim, vocab_size + 1)

        if self.freeze_backbone:
            self._set_backbone_requires_grad(False)

    _BACKBONE_PREFIXES = (
        'onehot.',
        'node_emb.',
        'edge_emb.',
        'signnet.',
        'encoder.',
        'attn.',
        'decoder.',
        'isotope_shift.',
    )

    def _set_backbone_requires_grad(self, requires_grad):
        for name, param in self.named_parameters():
            if any(name.startswith(prefix) for prefix in self._BACKBONE_PREFIXES):
                param.requires_grad = requires_grad

    def _optimizer_param_groups(self):
        cov_emb_params = list(self.cov_emb.parameters())
        if self.ce_embed is not None:
            cov_emb_params += list(self.ce_embed.parameters())
        if self.cov_film is not None:
            cov_emb_params += list(self.cov_film.parameters())
        clf_params = list(self.clf.parameters())
        use_split_lr = (
            self.freeze_backbone
            or self.clf_lr is not None
            or self.cov_emb_lr is not None
        )
        if not use_split_lr:
            return [{'params': [p for p in self.parameters() if p.requires_grad], 'lr': self.learning_rate}]

        cov_lr = self.cov_emb_lr if self.cov_emb_lr is not None else self.learning_rate
        clf_lr = self.clf_lr if self.clf_lr is not None else self.learning_rate
        groups = [
            {'params': cov_emb_params, 'lr': cov_lr, 'name': 'cov_emb'},
            {'params': clf_params, 'lr': clf_lr, 'name': 'clf'},
        ]
        if not self.freeze_backbone:
            backbone_params = [
                param for name, param in self.named_parameters()
                if param.requires_grad
                and not name.startswith('cov_emb.')
                and not name.startswith('cov_film.')
                and not name.startswith('ce_embed.')
                and not name.startswith('clf.')
            ]
            if backbone_params:
                groups.insert(0, {'params': backbone_params, 'lr': self.learning_rate, 'name': 'backbone'})
        return groups

    def configure_optimizers(self):
        groups = self._optimizer_param_groups()
        for group in groups:
            group.pop('name', None)
        opt = torch.optim.Adam(groups, weight_decay=self.weight_decay)
        if self.freeze_backbone:
            n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            n_total = sum(p.numel() for p in self.parameters())
            print(
                f'freeze_backbone: training cov_emb + clf only '
                f'({n_trainable:,}/{n_total:,} parameters)',
                flush=True,
            )
        elif self.clf_lr is not None or self.cov_emb_lr is not None:
            lr_msg = ', '.join(
                f'{g.get("name", "group")} lr={g["lr"]}'
                for g in self._optimizer_param_groups()
            )
            print(f'Layer-wise learning rates: {lr_msg}', flush=True)
        return opt

    def _cov_condition_input(self, covariates, ce_id=None):
        cov_in = covariates.view(covariates.shape[0], self.covariates_dim)
        if self.ce_embed is not None:
            if ce_id is None:
                raise ValueError('batch.ce_id is required when ce_embed is enabled')
            ce_vec = self.ce_embed(ce_id.view(-1).long())
            cov_in = torch.cat([cov_in, ce_vec], dim=-1)
        return cov_in

    def _apply_cov_film(self, z, covariates, ce_id=None):
        if self.cov_film is None:
            return z
        cov_in = self._cov_condition_input(covariates, ce_id)
        gamma, beta = self.cov_film(cov_in).chunk(2, dim=-1)
        return z * (1.0 + gamma) + beta
    
    def forward(self, g):
        batch_size = len(g.ptr) - 1
        device = g.x.device
        ce_id = getattr(g, 'ce_id', None)
        
        # embed node, edge, eigenfeatures
        x_atom, x_bond = self.onehot(g.x, g.edge_attr)
        x_atom = self.node_emb(x_atom)
        x_bond = self.edge_emb(x_bond)
        if self.num_eigs > 0:
            x_eig = self.signnet(g.eigvecs[:,:self.num_eigs],
                                 g.eigvals[:,:self.num_eigs][g.batch])
        else:
            x_eig = 0

        # run message passing
        x_mol = self.encoder(g, x_atom + x_eig, x_bond)
        # and attention-pool across atoms
        w = pyg.utils.softmax(self.attn(x_mol), g.batch)
        z = pyg.nn.global_add_pool(x_mol * w, g.batch)
        
        # condition molecule representation on covariates
        cov_in = self._cov_condition_input(g.covariates, ce_id)
        z = z + self.cov_emb(cov_in)
        # transform to spectrum representation
        z = self.decoder(z)
        z = self._apply_cov_film(z, g.covariates, ce_id)
        # and predict logits
        log_y_pred = self.clf(z)
        
        # add the (approximated) isotopic envelope
        log_y_pred = log_y_pred.view(batch_size, self.vocab_size+1, 1)
        isotope_shift = self.isotope_shift(z).view(batch_size, 1, len(isotope_types))
        log_y_pred = log_y_pred + isotope_shift            #同位素峰校正
        
        # correct intensities of double-counted formulas
        log_y_pred = log_y_pred - g.double_counted.unsqueeze(-1) * np.log(2)    #双计数校正
        
        return log_y_pred
        
    def step(self, batch, step):
        batch_size = len(batch.ptr) - 1
        device = batch.x.device
        
        log_y_pred = self(batch)
        
        y_pred = torch.softmax(log_y_pred.flatten(1), dim=1).view(log_y_pred.shape)
        
        ################################################################
        # calculate peak-marginal cross-entropy
        ################################################################
        
        # recall this has zero entries, which point to the pad element
        # they exactly should line up with the padding of y
        product_idx = batch.product_idx * len(isotope_types) + batch.isotope_idx
        loss_idx = batch.loss_idx * len(isotope_types) + batch.isotope_idx

        log_y_pred = torch.log_softmax(log_y_pred.flatten(1), dim=1).view(log_y_pred.shape)
        pad_mask = torch.zeros(1, self.vocab_size+1, 1,
                               device=device, dtype=log_y_pred.dtype)
        pad_mask[:,0] = self.log_epsilon
        log_y_pred = log_y_pred + pad_mask
        log_y_pred = torch.logsumexp(torch.stack([
            torch.gather(log_y_pred.flatten(1), 1, product_idx),
            torch.gather(log_y_pred.flatten(1), 1, loss_idx)
        ], dim=-1), dim=-1)
        
        is_pad = torch.maximum(batch.product_idx, batch.loss_idx) == 0
        peak_idx = batch.peak_idx + 1
        peak_idx[is_pad] = 0
        num_peaks = batch.intensities.shape[1]
        log_y_pred = scatter_logsumexp(log_y_pred, peak_idx, 1, dim_size=num_peaks+1)[:,1:]
        
        # predict a height of zero for peaks in intensities that weren't annotated / in vocab
        mask = torch.isinf(log_y_pred)
        log_y_pred = torch.where(mask, torch.zeros_like(log_y_pred) + self.log_epsilon, log_y_pred)
        
        per_sample_loss = -((batch.intensities * log_y_pred).sum(1))
        if step == 'train' and self.use_ce_loss_weights:
            ce_idx = batch.ce_id.view(-1).long()
            sample_weights = self.ce_loss_weight_lut[ce_idx]
            loss = (per_sample_loss * sample_weights).mean()
        else:
            loss = per_sample_loss.mean()
        
        self.log(f'{step}/loss', loss, batch_size=batch_size, sync_dist=step=='val')
        
        return loss
    
    def training_step(self, batch, batch_idx):
        return self.step(batch, 'train')
        
    def validation_step(self, batch, batch_idx):
        return self.step(batch, 'val')
    
    def predict_step(self, batch, batch_idx):
        # this should only be called at inference time - uses different batch structure
        
        log_y_pred = self(batch)
        
        x_pred = batch.mzs.unsqueeze(-1) + self.isotope_mzs.view(1,1,-1)
        
        y_pred = torch.softmax(log_y_pred.flatten(1), dim=1).view(log_y_pred.shape)
        
        # sparsify predictions
        xs = []
        ys = []
        for precursor_mz, has_isotopes, x, y in zip(
            batch.precursor_mz, batch.has_isotopes, x_pred, y_pred
        ):
            mask = (x >= self.min_mz)
            mask &= (x <= precursor_mz + self.isotope_mzs.max())
            mask &= (y > self.min_probability)
            mask[:,1:] &= has_isotopes # if we didn't ask for isotopes, don't predict them
            
            x = x[mask].flatten()
            y = y[mask].flatten() / y[mask].sum()
            
            # fast (approximate) way to deduplicate
            x = torch.round(x, decimals=4)
            x, idx = torch.unique(x, return_inverse=True)
            y = torch.zeros_like(x).scatter_add_(0, idx, y)
            
            xs.append(x)
            ys.append(y)

        return batch.spectrum, xs, ys
