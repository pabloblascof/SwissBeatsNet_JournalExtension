"""Classification heads on top of frozen encoder features.

Why this file exists. The anchor (submission 2813) fits its head inside
``mm_mae/classifiers/Classifiers.py::EnsembleClassifier`` -- a Lightning module
that is welded to the encoder, the Hydra config and a globally-cached config
loader (HANDOFF.md section 6). The journal branch trains heads on *cached*
features (``data/cache/features/<encoder_id>/cls.f32``) so that one encoder
pass serves every fold, seed and label policy. That needs the head as a plain
``nn.Module`` with the same parameterisation as the anchor's, and nothing else.

:class:`EnsembleHead` is that module. Per lead ``l`` it holds the anchor's
``non_linear`` head ``768 -> 384 -> ReLU -> Dropout(0.3) -> 1`` (or a single
``Linear`` for ``kind="linear"``), and the ensemble logit is the **unweighted
mean of the twelve per-lead logits**, exactly as ``EnsembleClassifier.forward``
does with ``torch.stack(logits_list).mean(dim=0)``. Parameter shapes match the
anchor's ``classifiers[<lead>]`` modules one-for-one, so a state_dict from the
anchor could be loaded lead by lead if that is ever needed.

:class:`LinearProbe` is the P4 shortcut probe used when a torch probe is
preferred to sklearn (``src/compass/p4_probes.py`` defaults to sklearn for
its convergence guarantees; this class is here so the torch path exists and
is tested).

Determinism: construction and forward are deterministic under
``torch.manual_seed`` -- initialisation draws from torch's global generator in
a fixed order, and dropout is the only stochastic op at train time.
"""

from __future__ import annotations

import math

from typing import Literal, Optional

import numpy as np
import torch
from torch import nn

HeadKind = Literal["non_linear", "linear"]

#: Every head kind :func:`build_head` can construct (D-026 A adds the last two).
HEAD_KINDS = ("non_linear", "linear", "lead_attention", "concat_mlp", "pooled_ensemble")
#: Head kinds that read per-lead *token* features instead of the CLS feature (S4C wave B H6, amendment B3).
#: Kept apart from :data:`HEAD_KINDS`, the CLS-feature kinds the container serves; the container gets a tokens
#: feature path only if H6 is chosen.
TOKEN_HEAD_KINDS = ("attn_pool",)
#: S4C H6 ``attn_pool`` reads per-lead token features ``[B, n_leads, n_tokens, emb_dim + TOKEN_MASK_CHANNELS]``:
#: the last channel is the token's validity (1 / 0, the same for every lead), so the mask travels through
#: every ordinary feature path (memmap views, minibatch gathers, ``predict_head``).
TOKEN_MASK_CHANNELS = 1

#: Ways :class:`PooledEnsembleHead` combines the twelve per-lead logits.
POOLS = ("mean", "max", "lse_fixed", "lse_learned", "noisy_or", "softmax_weights", "l1_weights")

#: The anchor's head hyperparameters (``mm_mae/config/chagas.yaml`` +
#: ``Classifiers.py``): ViT-B width, twelve leads, hidden = width // 2.
ANCHOR_N_LEADS = 12
ANCHOR_EMB_DIM = 768
ANCHOR_DROPOUT = 0.3


def _make_lead_head(emb_dim: int, kind: str, dropout: float) -> nn.Module:
    """One per-lead head with the anchor's layer order and shapes."""
    if kind == "non_linear":
        return nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim // 2, 1),
        )
    if kind == "linear":
        return nn.Linear(emb_dim, 1)
    raise ValueError(f"unknown head kind {kind!r}; expected 'non_linear' or 'linear'")


class EnsembleHead(nn.Module):
    """Twelve per-lead heads whose logits are averaged (the anchor's head).

    Parameters
    ----------
    n_leads, emb_dim
        Feature layout ``[B, n_leads, emb_dim]``. Anchor: 12 x 768.
    kind
        ``"non_linear"`` (anchor) or ``"linear"``.
    dropout
        Dropout after the hidden ReLU; only used by ``"non_linear"``.
    """

    def __init__(
        self,
        n_leads: int = ANCHOR_N_LEADS,
        emb_dim: int = ANCHOR_EMB_DIM,
        kind: HeadKind = "non_linear",
        dropout: float = ANCHOR_DROPOUT,
    ) -> None:
        super().__init__()
        if n_leads < 1:
            raise ValueError("n_leads must be >= 1")
        self.n_leads = int(n_leads)
        self.emb_dim = int(emb_dim)
        self.kind = kind
        self.dropout = float(dropout)
        self.leads = nn.ModuleList(
            [_make_lead_head(self.emb_dim, kind, self.dropout) for _ in range(self.n_leads)]
        )

    def _check(self, feats: torch.Tensor) -> None:
        if feats.dim() != 3 or feats.shape[1] != self.n_leads or feats.shape[2] != self.emb_dim:
            raise ValueError(
                f"expected feats of shape [B, {self.n_leads}, {self.emb_dim}], "
                f"got {tuple(feats.shape)}"
            )

    def per_lead_logits(self, feats: torch.Tensor) -> torch.Tensor:
        """``[B, n_leads, emb_dim] -> [B, n_leads]`` -- one logit per lead head.

        Exposed separately because the inter-lead study (S4) reads how much the
        twelve heads disagree, not only their mean.
        """
        self._check(feats)
        cols = [head(feats[:, i, :]) for i, head in enumerate(self.leads)]  # each [B, 1]
        return torch.cat(cols, dim=1)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """``[B, n_leads, emb_dim] -> [B]`` -- the mean of the per-lead logits."""
        return self.per_lead_logits(feats).mean(dim=1)


# ---------------------------------------------------------------------------
# D-026 A: heads that see all twelve leads at once
# ---------------------------------------------------------------------------


class LeadAttentionHead(nn.Module):
    """Inter-lead interaction at the head: a small transformer over 12 lead tokens.

    Why. The anchor's ensemble scores each lead in isolation and averages, so
    no parameter can express a contrast *between* leads -- yet the conduction
    phenotypes Chagas produces (RBBB + left anterior fascicular block) are
    defined by exactly such contrasts (V1 vs I/aVL axis), and S2.3/S4.1 found
    the per-lead embeddings already carry them. This head lets the classifier
    compare leads without touching the encoder.

    Layout: a projection ``emb_dim -> d_model`` shared across leads (LayerNorm
    after), plus a learned lead-identity embedding ``[n_leads, d_model]`` (the
    shared encoder gives every lead the same position code, so identity has to
    be added here), a learned CLS token, ``n_layers`` pre-norm
    ``TransformerEncoderLayer`` (``n_heads`` heads, dropout ``attn_dropout``,
    GELU, FFN ``4 * d_model``), a final LayerNorm and ``Linear(d_model, 1)``
    on the CLS token. ``forward(feats [B, n_leads, emb_dim]) -> [B]``.
    """

    def __init__(self, n_leads: int = ANCHOR_N_LEADS, emb_dim: int = ANCHOR_EMB_DIM, d_model: int = 256,
                 n_layers: int = 2, n_heads: int = 4, attn_dropout: float = 0.1, dropout: float = 0.1) -> None:
        super().__init__()
        self.kind = "lead_attention"
        self.n_leads, self.emb_dim = int(n_leads), int(emb_dim)
        self.d_model, self.n_layers, self.n_heads = int(d_model), int(n_layers), int(n_heads)
        self.attn_dropout, self.dropout = float(attn_dropout), float(dropout)
        self.proj = nn.Sequential(nn.Linear(self.emb_dim, self.d_model), nn.LayerNorm(self.d_model))
        self.lead_embed = nn.Parameter(torch.zeros(1, self.n_leads, self.d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        nn.init.trunc_normal_(self.lead_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        layer = nn.TransformerEncoderLayer(self.d_model, self.n_heads, dim_feedforward=4 * self.d_model,
                                           dropout=self.attn_dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, self.n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(self.d_model)
        self.drop = nn.Dropout(self.dropout)
        self.out = nn.Linear(self.d_model, 1)

    def spec(self) -> dict:
        return {"kind": self.kind, "n_leads": self.n_leads, "emb_dim": self.emb_dim, "d_model": self.d_model,
                "n_layers": self.n_layers, "n_heads": self.n_heads, "attn_dropout": self.attn_dropout,
                "dropout": self.dropout}

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        _check_shape(feats, self.n_leads, self.emb_dim)
        tok = self.proj(feats) + self.lead_embed
        tok = torch.cat([self.cls_token.expand(tok.shape[0], -1, -1), tok], dim=1)
        z = self.norm(self.encoder(tok)[:, 0])
        return self.out(self.drop(z)).squeeze(1)


class ConcatMLPHead(nn.Module):
    """Concatenate the twelve projected lead embeddings, then an MLP.

    A shared ``Linear(emb_dim, proj_dim) + LayerNorm + GELU`` per lead; the
    ``n_leads * proj_dim`` concatenation keeps lead identity by position, so
    the MLP ``-> hidden -> GELU -> Dropout -> 1`` can weigh leads against each
    other (the simplest head with inter-lead interaction).
    """

    def __init__(self, n_leads: int = ANCHOR_N_LEADS, emb_dim: int = ANCHOR_EMB_DIM, proj_dim: int = 128,
                 hidden: int = 256, dropout: float = 0.3) -> None:
        super().__init__()
        self.kind = "concat_mlp"
        self.n_leads, self.emb_dim = int(n_leads), int(emb_dim)
        self.proj_dim, self.hidden, self.dropout = int(proj_dim), int(hidden), float(dropout)
        self.proj = nn.Sequential(nn.Linear(self.emb_dim, self.proj_dim), nn.LayerNorm(self.proj_dim), nn.GELU())
        self.mlp = nn.Sequential(nn.Dropout(self.dropout), nn.Linear(self.n_leads * self.proj_dim, self.hidden),
                                 nn.GELU(), nn.Dropout(self.dropout), nn.Linear(self.hidden, 1))

    def spec(self) -> dict:
        return {"kind": self.kind, "n_leads": self.n_leads, "emb_dim": self.emb_dim, "proj_dim": self.proj_dim,
                "hidden": self.hidden, "dropout": self.dropout}

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        _check_shape(feats, self.n_leads, self.emb_dim)
        z = self.proj(feats).flatten(1)
        return self.mlp(z).squeeze(1)


class PooledEnsembleHead(nn.Module):
    """The anchor's twelve per-lead MLPs with a different pooling of their logits,
    optionally fed a *lead-contrast* input.

    Pools (``pool``), over per-lead logits ``l_1..l_12``:

    * ``mean`` -- the anchor (but see ``lead_contrast``);
    * ``max`` -- the most confident lead decides;
    * ``lse_fixed`` / ``lse_learned`` -- ``T * (logsumexp(l / T) - log n)``, a
      soft maximum that equals the mean as ``T -> inf`` and the max as
      ``T -> 0``; ``T = 1`` fixed, or learned (``softplus`` parameter, init 1);
    * ``noisy_or`` -- ``P = 1 - prod(1 - sigmoid(l_i))``, returned as a logit
      (any lead may carry the finding); lead biases start so that ``P = 0.5``;
    * ``softmax_weights`` -- ``sum_i softmax(a)_i l_i``, ``a`` learned, init 0
      (= mean);
    * ``l1_weights`` -- ``sum_i relu(w_i) l_i``, ``w`` learned, init ``1/n``
      (= mean), with :meth:`penalty` ``= l1 * sum relu(w)`` added to the loss by
      ``fit_head`` so leads can be switched off.

    ``lead_contrast``: each lead head reads ``[e_l, e_l - mean_{k != l} e_k]``
    (``2 * emb_dim``) -- what differs between this lead and the rest, the
    quantity the inter-lead argument of the paper is about, without attention.
    """

    def __init__(self, n_leads: int = ANCHOR_N_LEADS, emb_dim: int = ANCHOR_EMB_DIM, pool: str = "mean",
                 lead_contrast: bool = False, dropout: float = ANCHOR_DROPOUT, l1: float = 1e-3) -> None:
        super().__init__()
        if pool not in POOLS:
            raise ValueError(f"unknown pool {pool!r}; one of {POOLS}")
        self.kind = "pooled_ensemble"
        self.n_leads, self.emb_dim = int(n_leads), int(emb_dim)
        self.pool, self.lead_contrast = pool, bool(lead_contrast)
        self.dropout, self.l1 = float(dropout), float(l1)
        in_dim = self.emb_dim * (2 if self.lead_contrast else 1)
        self.leads = nn.ModuleList([_make_lead_head(in_dim, "non_linear", self.dropout) for _ in range(self.n_leads)])
        if pool == "noisy_or":
            # Start at P = 0.5 rather than 1 - 0.5**12 ~ 1 (saturated): each lead's
            # prior p0 solves 1 - (1 - p0)**n = 0.5.
            p0 = 1.0 - 0.5 ** (1.0 / self.n_leads)
            with torch.no_grad():
                for h in self.leads:
                    h[-1].bias.fill_(math.log(p0 / (1.0 - p0)))
        if pool == "lse_learned":
            self.t_raw = nn.Parameter(torch.tensor(math.log(math.e - 1.0)))  # softplus(t_raw) = 1
        elif pool == "softmax_weights":
            self.lead_logits = nn.Parameter(torch.zeros(self.n_leads))
        elif pool == "l1_weights":
            self.lead_weights = nn.Parameter(torch.full((self.n_leads,), 1.0 / self.n_leads))

    def spec(self) -> dict:
        return {"kind": self.kind, "n_leads": self.n_leads, "emb_dim": self.emb_dim, "pool": self.pool,
                "lead_contrast": self.lead_contrast, "dropout": self.dropout, "l1": self.l1}

    def per_lead_logits(self, feats: torch.Tensor) -> torch.Tensor:
        _check_shape(feats, self.n_leads, self.emb_dim)
        if self.lead_contrast:
            others = (feats.sum(dim=1, keepdim=True) - feats) / max(self.n_leads - 1, 1)
            feats = torch.cat([feats, feats - others], dim=-1)
        return torch.cat([h(feats[:, i, :]) for i, h in enumerate(self.leads)], dim=1)

    def lead_weights_now(self) -> Optional[torch.Tensor]:
        if self.pool == "softmax_weights":
            return torch.softmax(self.lead_logits, dim=0)
        if self.pool == "l1_weights":
            return torch.relu(self.lead_weights)
        return None

    def penalty(self) -> torch.Tensor:
        if self.pool == "l1_weights":
            return self.l1 * torch.relu(self.lead_weights).sum()
        return torch.zeros((), device=next(self.parameters()).device)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.pool_logits(self.per_lead_logits(feats))

    def pool_logits(self, l: torch.Tensor) -> torch.Tensor:
        """``[B, n_leads]`` per-lead logits -> ``[B]`` (the pooling of :meth:`forward`)."""
        n = l.shape[1]
        if self.pool == "mean":
            return l.mean(dim=1)
        if self.pool == "max":
            return l.max(dim=1).values
        if self.pool in ("lse_fixed", "lse_learned"):
            t = nn.functional.softplus(self.t_raw) + 1e-3 if self.pool == "lse_learned" else l.new_tensor(1.0)
            return t * (torch.logsumexp(l / t, dim=1) - math.log(n))
        if self.pool == "noisy_or":
            s = (-nn.functional.softplus(l).sum(dim=1)).clamp(max=-1e-6)  # log prod(1 - p_i) = log(1 - P)
            return torch.log(-torch.expm1(s)) - s  # logit(P) = log P - log(1 - P)
        return (l * self.lead_weights_now()).sum(dim=1)


class AttnPoolHead(nn.Module):
    """S4C wave B variant H6 ``attnpool`` (docs/journal/S4C_IMPROVE.md amendment B3): attentive per-lead pooling.

    Per lead ``l``, ``n_queries`` learned queries ``q_{l,m}`` (init ``N(0, query_std^2)``, so pooling starts
    close to a uniform average) attend over that lead's valid encoder output tokens ``x_t``:

    * ``a_{m,t} = softmax_t(q_{l,m} . LN(x_t) / sqrt(D))`` over the valid tokens (masked scores are ``-inf``,
      hence weight exactly 0). ``LN`` has no affine parameters; there are no key or value projections (the
      efficient-probing simplification of Psomas et al., arXiv:2506.10178).
    * ``pooled_l = mean_m sum_t a_{m,t} x_t`` (``D`` dims, the CLS head's input width).
    * Lead ``l``'s anchor MLP head reads ``pooled_l``; the record logit is the mean of the lead logits.

    Input: ``[B, n_leads, n_tokens, emb_dim + 1]``, the last channel the validity mask
    (:data:`TOKEN_MASK_CHANNELS`).
    """

    def __init__(self, n_leads: int = ANCHOR_N_LEADS, emb_dim: int = ANCHOR_EMB_DIM, n_queries: int = 4,
                 dropout: float = ANCHOR_DROPOUT, query_std: float = 0.02) -> None:
        super().__init__()
        if n_queries < 1:
            raise ValueError("n_queries must be >= 1")
        self.kind = "attn_pool"
        self.n_leads, self.emb_dim, self.n_queries = int(n_leads), int(emb_dim), int(n_queries)
        self.dropout, self.query_std = float(dropout), float(query_std)
        self.queries = nn.Parameter(torch.randn(self.n_leads, self.n_queries, self.emb_dim) * self.query_std)
        self.leads = nn.ModuleList([_make_lead_head(self.emb_dim, "non_linear", self.dropout)
                                    for _ in range(self.n_leads)])

    def spec(self) -> dict:
        return {"kind": self.kind, "n_leads": self.n_leads, "emb_dim": self.emb_dim, "n_queries": self.n_queries,
                "dropout": self.dropout, "query_std": self.query_std}

    def attention(self, feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(tokens [B, L, T, D], valid [B, T], weights [B, L, M, T])``."""
        d = self.emb_dim
        if feats.dim() != 4 or feats.shape[1] != self.n_leads or feats.shape[3] != d + TOKEN_MASK_CHANNELS:
            raise ValueError(f"expected token features [B, {self.n_leads}, T, {d} + {TOKEN_MASK_CHANNELS}], "
                             f"got {tuple(feats.shape)}")
        tok = feats[..., :d]
        valid = feats[:, 0, :, d] > 0.5
        keys = nn.functional.layer_norm(tok, (d,))
        scores = torch.einsum("lmd,bltd->blmt", self.queries.to(keys.dtype), keys) / math.sqrt(d)
        scores = scores.masked_fill(~valid[:, None, None, :], float("-inf"))
        return tok, valid, torch.softmax(scores, dim=-1)

    def pooled(self, feats: torch.Tensor) -> torch.Tensor:
        tok, _, a = self.attention(feats)
        return torch.einsum("blmt,bltd->bld", a, tok) / self.n_queries

    def per_lead_logits(self, feats: torch.Tensor) -> torch.Tensor:
        p = self.pooled(feats)
        return torch.cat([h(p[:, i, :]) for i, h in enumerate(self.leads)], dim=1)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.per_lead_logits(feats).mean(dim=1)


def _check_shape(feats: torch.Tensor, n_leads: int, emb_dim: int) -> None:
    if feats.dim() != 3 or feats.shape[1] != n_leads or feats.shape[2] != emb_dim:
        raise ValueError(f"expected feats of shape [B, {n_leads}, {emb_dim}], got {tuple(feats.shape)}")


def head_spec(head: nn.Module) -> dict:
    """The constructor arguments that rebuild ``head`` via :func:`build_head`."""
    if hasattr(head, "spec"):
        return head.spec()
    return {"kind": head.kind, "n_leads": head.n_leads, "emb_dim": head.emb_dim, "dropout": head.dropout}


def build_head(kind: str = "non_linear", n_leads: int = ANCHOR_N_LEADS, emb_dim: int = ANCHOR_EMB_DIM,
               dropout: Optional[float] = None, demographics: bool = False, input_norm: str = "none",
               norm_dim: Optional[int] = None, rhythm: bool = False, erase: str = "none", **kw) -> nn.Module:
    """Construct any head kind. ``build_head(**head_spec(h))`` rebuilds ``h``.

    ``non_linear``/``linear`` return the anchor's :class:`EnsembleHead`
    unchanged (so the anchor's state_dict layout is untouched); ``dropout``
    ``None`` means the kind's default. Unknown keyword arguments for a kind
    raise, so a spec cannot silently drop a hyperparameter. ``input_norm``
    (:data:`INPUT_NORMS`) wraps the head in :class:`InputNorm` -- innermost,
    so demographics and the auxiliary read-out sit outside it; ``emb_dim`` is
    always the width of the features the head is *fed*. ``rhythm`` wraps the
    result in :class:`WithRhythm` (outermost). ``erase`` (:data:`ERASURES`,
    S4C_IMPROVE H3) wraps the head in :class:`ConceptErasure`, which acts on the
    raw features: inside demographics / rhythm, outside any ``input_norm``.
    """
    if rhythm:
        return WithRhythm(build_head(kind, n_leads, emb_dim, dropout, demographics=demographics,
                                     input_norm=input_norm, norm_dim=norm_dim, erase=erase, **kw))
    if demographics:
        return WithDemographics(build_head(kind, n_leads, emb_dim, dropout, input_norm=input_norm,
                                           norm_dim=norm_dim, erase=erase, **kw))
    if erase not in ERASURES:
        raise ValueError(f"unknown erase {erase!r}; one of {ERASURES}")
    if erase != "none":
        return ConceptErasure(build_head(kind, n_leads, emb_dim, dropout, input_norm=input_norm,
                                         norm_dim=norm_dim, **kw), erase, n_leads, emb_dim)
    if input_norm not in INPUT_NORMS:
        raise ValueError(f"unknown input_norm {input_norm!r}; one of {INPUT_NORMS}")
    if input_norm != "none":
        out_dim = int(emb_dim if input_norm != "pca_whiten" else (norm_dim or DEFAULT_PCA_DIM))
        if out_dim < 1 or out_dim > emb_dim:
            raise ValueError(f"norm_dim must be in [1, {emb_dim}], got {out_dim}")
        return InputNorm(build_head(kind, n_leads, out_dim, dropout, **kw), input_norm, n_leads, emb_dim)
    if norm_dim is not None:
        raise ValueError("norm_dim is only meaningful with input_norm='pca_whiten'")
    if kind in ("non_linear", "linear"):
        if kw:
            raise ValueError(f"head kind {kind!r} takes no {sorted(kw)}")
        return EnsembleHead(n_leads, emb_dim, kind, ANCHOR_DROPOUT if dropout is None else dropout)
    if kind == "lead_attention":
        return LeadAttentionHead(n_leads, emb_dim, dropout=0.1 if dropout is None else dropout, **kw)
    if kind == "pooled_ensemble":
        return PooledEnsembleHead(n_leads, emb_dim, dropout=ANCHOR_DROPOUT if dropout is None else dropout, **kw)
    if kind == "concat_mlp":
        return ConcatMLPHead(n_leads, emb_dim, dropout=ANCHOR_DROPOUT if dropout is None else dropout, **kw)
    if kind == "attn_pool":
        return AttnPoolHead(n_leads, emb_dim, dropout=ANCHOR_DROPOUT if dropout is None else dropout, **kw)
    raise ValueError(f"unknown head kind {kind!r}; expected one of {HEAD_KINDS + TOKEN_HEAD_KINDS}")


# ---------------------------------------------------------------------------
# Head-input normalisation, fitted on the training rows and shipped in head.pt
# ---------------------------------------------------------------------------

#: ``none`` (the anchor: raw features), ``standardise`` (per-dimension z-score
#: shared across leads), ``standardise_per_lead`` (per-lead, per-dimension
#: z-score), ``pca_whiten`` (PCA whitening shared across leads, to ``norm_dim``).
INPUT_NORMS = ("none", "standardise", "standardise_per_lead", "pca_whiten")
DEFAULT_PCA_DIM = 256


class InputNorm(nn.Module):
    """A fixed affine transform of the per-lead features in front of any head.

    Why. The cached CLS features have per-dimension standard deviations that
    span about three orders of magnitude and a large shared offset, and the
    anchor's head reads them raw. The transform's statistics are *buffers*:
    fitted once on the training rows (``src.train.head.fit_input_norm``,
    never on validation or reporting rows), saved in ``head.pt`` with the
    weights, and applied verbatim by the container, which rebuilds this module
    from the spec (``input_norm``, ``norm_dim``) and loads the buffers strictly.

    ``transform(feats [B, n_leads, emb_dim]) -> [B, n_leads, norm_dim]``:
    ``(x - mu) / scale`` (``mu``/``scale`` of shape ``[emb_dim]``, or
    ``[n_leads, emb_dim]`` per lead), or ``(x - mu) @ W`` for PCA whitening
    (``W = U_k / sqrt(lambda_k + eps)``). ``forward = core(transform(feats))``.
    """

    def __init__(self, core: nn.Module, mode: str, n_leads: int, in_dim: int) -> None:
        super().__init__()
        if mode not in INPUT_NORMS or mode == "none":
            raise ValueError(f"InputNorm mode must be one of {INPUT_NORMS[1:]}, got {mode!r}")
        self.core, self.mode = core, mode
        self.kind = getattr(core, "kind", "non_linear")
        self.n_leads, self.emb_dim = int(n_leads), int(in_dim)
        self.norm_dim = int(core.emb_dim)
        if mode != "pca_whiten" and self.norm_dim != self.emb_dim:
            raise ValueError("standardisation keeps the width; core.emb_dim must equal in_dim")
        shape = (self.n_leads, self.emb_dim) if mode == "standardise_per_lead" else (self.emb_dim,)
        self.register_buffer("mu", torch.zeros(shape))
        if mode == "pca_whiten":
            self.register_buffer("W", torch.eye(self.emb_dim)[:, : self.norm_dim].clone())
        else:
            self.register_buffer("scale", torch.ones(shape))
        #: Set by the fit; a head that never saw its statistics must not score.
        self.register_buffer("fitted", torch.tensor(False))

    def spec(self) -> dict:
        out = {**head_spec(self.core), "emb_dim": self.emb_dim, "input_norm": self.mode}
        if self.mode == "pca_whiten":
            out["norm_dim"] = self.norm_dim
        return out

    def transform(self, feats: torch.Tensor) -> torch.Tensor:
        _check_shape(feats, self.n_leads, self.emb_dim)
        if not bool(self.fitted):
            raise RuntimeError("InputNorm statistics were never fitted (src.train.head.fit_input_norm)")
        x = feats - self.mu.to(feats.dtype)
        if self.mode == "pca_whiten":
            return x @ self.W.to(feats.dtype)
        return x / self.scale.to(feats.dtype)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.core(self.transform(feats))

    def per_lead_logits(self, feats: torch.Tensor) -> torch.Tensor:
        return self.core.per_lead_logits(self.transform(feats))

    def lead_weights_now(self) -> Optional[torch.Tensor]:
        f = getattr(self.core, "lead_weights_now", None)
        return f() if f is not None else None

    def penalty(self) -> torch.Tensor:
        pen = getattr(self.core, "penalty", None)
        return pen() if pen is not None else torch.zeros((), device=self.mu.device)


# ---------------------------------------------------------------------------
# Linear concept erasure of recording source (S4C_IMPROVE.md H3, D-073)
# ---------------------------------------------------------------------------

#: ``none`` (every recipe so far) or ``leace_source``: LEACE (Belrose et al.
#: 2023, "LEACE: Perfect linear concept erasure in closed form") of the
#: recording source (CODE-15% / SaMi-Trop / PTB-XL), per lead.
ERASURES = ("none", "leace_source")


class ConceptErasure(nn.Module):
    """A fixed per-lead affine eraser in front of any head.

    ``transform(x)[:, l] = x[:, l] - (x[:, l] - mu[l]) @ proj[l]``, where
    ``proj[l]`` is the transpose of lead ``l``'s LEACE matrix
    ``P = W^+ P_{W S_xz} W`` (``W`` the whitening of the lead's feature
    covariance, ``P_A`` the orthogonal projector onto ``col(A)``, ``S_xz`` the
    feature-source cross-covariance). Afterwards no linear function of the
    features covaries with the source on the rows it was fitted on, with the
    least change in mean squared distance. The buffers are fitted once on the
    TRAINING rows (``src.train.head.fit_leace``), saved in ``head.pt`` and
    applied verbatim by the container (rebuilt from the spec, loaded strictly).
    """

    def __init__(self, core: nn.Module, mode: str, n_leads: int, emb_dim: int) -> None:
        super().__init__()
        if mode not in ERASURES or mode == "none":
            raise ValueError(f"ConceptErasure mode must be one of {ERASURES[1:]}, got {mode!r}")
        self.core, self.mode = core, mode
        self.kind = getattr(core, "kind", "non_linear")
        self.n_leads, self.emb_dim = int(n_leads), int(emb_dim)
        self.register_buffer("mu", torch.zeros(self.n_leads, self.emb_dim))
        self.register_buffer("proj", torch.zeros(self.n_leads, self.emb_dim, self.emb_dim))
        self.register_buffer("fitted", torch.tensor(False))

    @property
    def base(self) -> nn.Module:
        """The wrapped head (what :func:`find_input_norm` walks through), not a second registration."""
        return self.core

    def spec(self) -> dict:
        return {**head_spec(self.core), "emb_dim": self.emb_dim, "erase": self.mode}

    def transform(self, feats: torch.Tensor) -> torch.Tensor:
        _check_shape(feats, self.n_leads, self.emb_dim)
        if not bool(self.fitted):
            raise RuntimeError("ConceptErasure was never fitted (src.train.head.fit_leace)")
        x = feats - self.mu.to(feats.dtype)
        return feats - torch.einsum("bld,lde->ble", x, self.proj.to(feats.dtype))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.core(self.transform(feats))

    def per_lead_logits(self, feats: torch.Tensor) -> torch.Tensor:
        return self.core.per_lead_logits(self.transform(feats))

    def lead_weights_now(self) -> Optional[torch.Tensor]:
        f = getattr(self.core, "lead_weights_now", None)
        return f() if f is not None else None

    def penalty(self) -> torch.Tensor:
        pen = getattr(self.core, "penalty", None)
        return pen() if pen is not None else torch.zeros((), device=self.mu.device)


def find_erasure(head: nn.Module) -> Optional["ConceptErasure"]:
    """The :class:`ConceptErasure` inside ``head`` (through the aux / demographics wrappers), or ``None``."""
    m = head
    for _ in range(6):
        if isinstance(m, ConceptErasure):
            return m
        m = getattr(m, "main", None) if hasattr(m, "chagas_head") else getattr(m, "base", None)
        if m is None:
            return None
    return None


def find_input_norm(head: nn.Module) -> Optional["InputNorm"]:
    """The :class:`InputNorm` inside ``head`` (through the aux / demographics wrappers), or ``None``."""
    m = head
    for _ in range(6):
        if isinstance(m, InputNorm):
            return m
        m = getattr(m, "main", None) if hasattr(m, "chagas_head") else getattr(m, "base", None)
        if m is None:
            return None
    return None


# ---------------------------------------------------------------------------
# Demographics by concatenation into the final linear layer (D-026 A, D-028)
# ---------------------------------------------------------------------------

#: Age is clipped to this range before standardising. PTB-XL anonymises ages
#: above 89 as 300 (a value no other corpus has, i.e. a source flag); clipping
#: at 90 maps it onto the oldest CODE/SaMi-Trop patients.
AGE_CLIP = (0.0, 90.0)
N_DEMO = 4  # [age_z, age_missing, sex (+1 male / -1 female / 0), sex_missing]


def demographic_matrix(age, sex, age_mean: float, age_std: float) -> np.ndarray:
    """Raw age/sex -> ``[N, 4]`` float32: ``[age_z, age_missing, sex_pm1, sex_missing]``.

    ``age``: numbers, ``None``/NaN or unparseable strings (missing). ``sex``:
    ``Male``/``Female`` (case-insensitive, also ``M``/``F``); anything else is
    missing. Missing entries are 0 with their flag set, so a record without a
    header field is scored rather than dropped (the container's contract).
    """
    age = np.asarray([_to_float(a) for a in np.atleast_1d(np.asarray(age, dtype=object))], dtype=np.float64)
    sex = np.atleast_1d(np.asarray(sex, dtype=object))
    if len(age) != len(sex):
        raise ValueError(f"age ({len(age)}) and sex ({len(sex)}) disagree on N")
    a_miss = ~np.isfinite(age)
    a = np.clip(np.where(a_miss, age_mean, age), *AGE_CLIP)
    z = np.where(a_miss, 0.0, (a - age_mean) / max(float(age_std), 1e-6))
    s = np.array([_sex_pm1(v) for v in sex], dtype=np.float64)
    s_miss = s == 0.0
    return np.stack([z, a_miss, s, s_miss], axis=1).astype(np.float32)


def age_stats(age) -> tuple[float, float]:
    """Mean and std of the clipped, finite ages (the standardisation a head stores)."""
    a = np.asarray([_to_float(v) for v in np.atleast_1d(np.asarray(age, dtype=object))], dtype=np.float64)
    a = np.clip(a[np.isfinite(a)], *AGE_CLIP)
    if a.size == 0:
        return 50.0, 20.0
    return float(a.mean()), float(max(a.std(), 1e-6))


def _to_float(v) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return f if np.isfinite(f) else float("nan")


def _sex_pm1(v) -> float:
    t = str(v).strip().lower() if v is not None else ""
    if t in ("male", "m"):
        return 1.0
    if t in ("female", "f"):
        return -1.0
    return 0.0


class WithDemographics(nn.Module):
    """``base(feats) + w . demo``: demographics concatenated into the final layer.

    Every head in this file ends in a linear map to one logit (the ensemble's
    is the mean of twelve), and ``Linear([z; d]) = Linear(z) + w . d``, so an
    additive linear term in the demographic vector *is* concatenation into the
    final layer, for any base head, without changing the base's parameters.
    Deliberately not FiLM (author's scope, 2026-09-27). The standardisation
    (training-pool ``age_mean``/``age_std``) is stored as buffers, so it
    travels in ``state_dict`` to the container.
    """

    def __init__(self, base: nn.Module, n_demo: int = N_DEMO) -> None:
        super().__init__()
        self.base = base
        self.kind = getattr(base, "kind", "non_linear")
        self.n_leads, self.emb_dim = int(base.n_leads), int(base.emb_dim)
        self.n_demo = int(n_demo)
        self.demo = nn.Linear(self.n_demo, 1, bias=False)
        nn.init.zeros_(self.demo.weight)
        self.register_buffer("age_mean", torch.tensor(50.0))
        self.register_buffer("age_std", torch.tensor(20.0))

    uses_demographics = True

    def set_age_stats(self, age) -> None:
        m, s = age_stats(age)
        self.age_mean.fill_(m)
        self.age_std.fill_(s)

    def demo_matrix(self, age, sex) -> torch.Tensor:
        return torch.from_numpy(demographic_matrix(age, sex, float(self.age_mean), float(self.age_std)))

    def spec(self) -> dict:
        return {**head_spec(self.base), "demographics": True}

    def penalty(self) -> torch.Tensor:
        pen = getattr(self.base, "penalty", None)
        return pen() if pen is not None else torch.zeros((), device=self.demo.weight.device)

    def forward(self, feats: torch.Tensor, demo: Optional[torch.Tensor] = None) -> torch.Tensor:
        if demo is None:
            raise ValueError("this head was fitted with demographics; pass demo [B, 4]")
        return self.base(feats) + self.demo(demo.to(feats.dtype)).squeeze(1)


# ---------------------------------------------------------------------------
# S4.3 auxiliary physiology supervision (training-time only; never exported)
# ---------------------------------------------------------------------------

#: The six conduction labels (ST before AF, D-006), then age and sex.
CONDUCTION_LABELS = ("1dAVb", "RBBB", "LBBB", "SB", "ST", "AF")
AUX_TARGETS = CONDUCTION_LABELS + ("age", "sex")
N_AUX = len(AUX_TARGETS)


def aux_target_matrix(six, age, sex) -> np.ndarray:
    """``[N, 8]`` float32 raw auxiliary targets, NaN = missing (masked in the loss).

    ``six``: ``[N, 6]`` conduction labels in :data:`CONDUCTION_LABELS` order
    (SaMi-Trop rows are NaN). ``age``: raw years, clipped to :data:`AGE_CLIP`
    (so PTB-XL's anonymised 300 becomes 90); ``fit_head`` standardises it with
    the training-pool mean/std. ``sex``: 1 male, 0 female, NaN otherwise.
    Provenance differs by source (CODE-15% text-mined from the report,
    PTB-XL cardiologist SCP codes, ``label_provenance``); the targets are used
    as they are, which is why they are only an auxiliary signal.
    """
    six = np.asarray(six, dtype=np.float64).reshape(-1, len(CONDUCTION_LABELS))
    a = np.asarray([_to_float(v) for v in np.atleast_1d(np.asarray(age, dtype=object))], dtype=np.float64)
    a = np.where(np.isfinite(a), np.clip(a, *AGE_CLIP), np.nan)
    s = np.array([_sex_pm1(v) for v in np.atleast_1d(np.asarray(sex, dtype=object))], dtype=np.float64)
    s = np.where(s == 0.0, np.nan, (s + 1.0) / 2.0)
    if not (len(six) == len(a) == len(s)):
        raise ValueError(f"six ({len(six)}), age ({len(a)}) and sex ({len(s)}) disagree on N")
    return np.concatenate([six, a[:, None], s[:, None]], axis=1).astype(np.float32)


def unwrap_head(head: nn.Module) -> tuple[Optional[nn.Module], Optional[nn.Module], Optional["InputNorm"], nn.Module]:
    """``(with_rhythm, with_demographics, input_norm, core)`` of a (possibly wrapped) Chagas head.

    The wrapper order :func:`build_head` produces is
    ``WithRhythm(WithDemographics(InputNorm(core)))``, each layer optional.
    """
    rhy = head if isinstance(head, WithRhythm) else None
    m = head.base if rhy is not None else head
    dem = m if isinstance(m, WithDemographics) else None
    m = m.base if dem is not None else m
    norm = m if isinstance(m, InputNorm) else None
    core = m.core if norm is not None else m
    return rhy, dem, norm, core


def call_head(head: nn.Module, feats: torch.Tensor, demo: Optional[torch.Tensor] = None,
              rhythm: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Call any head with the side inputs it declares (``uses_demographics`` / ``uses_rhythm``)."""
    if getattr(head, "uses_rhythm", False):
        return head(feats, demo, rhythm)
    if getattr(head, "uses_demographics", False):
        return head(feats, demo)
    return head(feats)


class AuxTaskHead(nn.Module):
    """A Chagas head plus a linear auxiliary read-out of its per-lead hidden layer.

    ``main`` is any per-lead-MLP head (:class:`EnsembleHead` ``non_linear``
    or :class:`PooledEnsembleHead`), optionally wrapped in :class:`InputNorm`,
    :class:`WithDemographics` and :class:`WithRhythm`. The auxiliary layer
    reads the concatenated post-ReLU hidden activations of the twelve lead
    MLPs (``[B, 12 * hidden]``, lead identity kept by position) and predicts
    :data:`AUX_TARGETS`, so the auxiliary losses shape the same hidden units
    the Chagas logit is computed from. :meth:`forward` is exactly
    ``main.forward`` -- the Chagas logit, the model's only output;
    :meth:`chagas_head` returns ``main``, which is what ``fit_head`` returns and
    what gets exported (the auxiliary layer never reaches ``head.pt`` or the
    container).
    """

    def __init__(self, main: nn.Module, n_aux: int = N_AUX) -> None:
        super().__init__()
        core = unwrap_head(main)[3]
        leads = getattr(core, "leads", None)
        if leads is None or not all(isinstance(m, nn.Sequential) and len(m) == 4 for m in leads):
            raise ValueError("auxiliary supervision needs a per-lead MLP head (non_linear or pooled_ensemble)")
        self.main = main
        self.kind = getattr(main, "kind", "non_linear")
        self.n_leads, self.emb_dim = int(main.n_leads), int(main.emb_dim)
        hidden = int(leads[0][0].out_features)
        self.aux = nn.Linear(self.n_leads * hidden, int(n_aux))

    @property
    def uses_demographics(self) -> bool:
        return bool(getattr(self.main, "uses_demographics", False))

    @property
    def uses_rhythm(self) -> bool:
        return bool(getattr(self.main, "uses_rhythm", False))

    def chagas_head(self) -> nn.Module:
        return self.main

    def spec(self) -> dict:
        return head_spec(self.main)

    def set_age_stats(self, age) -> None:
        self.main.set_age_stats(age)

    def demo_matrix(self, age, sex) -> torch.Tensor:
        return self.main.demo_matrix(age, sex)

    def fit_rhythm_stats(self, raw) -> dict:
        return self.main.fit_rhythm_stats(raw)

    def rhythm_matrix(self, raw) -> torch.Tensor:
        return self.main.rhythm_matrix(raw)

    def penalty(self) -> torch.Tensor:
        pen = getattr(self.main, "penalty", None)
        return pen() if pen is not None else torch.zeros((), device=self.aux.weight.device)

    def forward(self, feats: torch.Tensor, demo: Optional[torch.Tensor] = None,
                rhythm: Optional[torch.Tensor] = None) -> torch.Tensor:
        return call_head(self.main, feats, demo, rhythm)

    def forward_with_aux(self, feats: torch.Tensor, demo: Optional[torch.Tensor] = None,
                         rhythm: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """``(chagas_logit [B], aux_logits [B, n_aux])``; the logit equals :meth:`forward`'s."""
        rhy, dem, norm, base = unwrap_head(self.main)
        x = norm.transform(feats) if norm is not None else feats
        _check_shape(x, base.n_leads, base.emb_dim)
        if getattr(base, "lead_contrast", False):
            others = (x.sum(dim=1, keepdim=True) - x) / max(base.n_leads - 1, 1)
            x = torch.cat([x, x - others], dim=-1)
        hs, ls = [], []
        for i, m in enumerate(base.leads):
            h = m[1](m[0](x[:, i, :]))
            hs.append(h)
            ls.append(m[3](m[2](h)))
        l = torch.cat(ls, dim=1)
        logit = base.pool_logits(l) if hasattr(base, "pool_logits") else l.mean(dim=1)
        if dem is not None:
            if demo is None:
                raise ValueError("this head was fitted with demographics; pass demo [B, 4]")
            logit = logit + dem.demo(demo.to(feats.dtype)).squeeze(1)
        if rhy is not None:
            logit = logit + rhy.term(rhythm, feats.dtype)
        return logit, self.aux(torch.cat(hs, dim=1))


# ---------------------------------------------------------------------------
# Hybrid rhythm view (exploratory): signal-derived descriptors at the head
# ---------------------------------------------------------------------------

#: The ten S3.3 rhythm / interval descriptors (``src/baselines/features_classical.py``,
#: lead II): heart rate, RR variability and the dwt width proxies (not clinical
#: durations). Label-free functions of the preprocessed signal.
RHYTHM_VIEW_NAMES = ("hr_mean", "hr_std", "rr_sdnn_ms", "rr_rmssd_ms", "rr_pnn50", "rr_cv",
                     "qrs_width_dwt_ms", "pr_dwt_ms", "qt_dwt_ms", "qtc_dwt_ms")
N_RHYTHM = len(RHYTHM_VIEW_NAMES)
#: Standardised descriptors are clipped to +-this many training SDs (outlier guard).
RHYTHM_Z_CLIP = 5.0


class WithRhythm(nn.Module):
    """``base(feats[, demo]) + w . [z(r); missing(r)]``: rhythm descriptors concatenated into the final layer.

    Exactly the construction of :class:`WithDemographics` (an additive linear
    term = concatenation into the last linear layer), for the
    :data:`RHYTHM_VIEW_NAMES` descriptors of the record. Missing values (a
    failed extraction, fewer than three beats, a failed delineation) are
    imputed with the training-pool median and flagged by one indicator per
    descriptor; the imputed values are z-scored with the training-pool mean
    and SD and clipped at :data:`RHYTHM_Z_CLIP`. Median, mean and SD are
    buffers, fitted on the training rows only (:meth:`fit_rhythm_stats`) and
    shipped in ``head.pt``; the container recomputes the raw descriptors from
    the preprocessed record with the same extractor (``src/data/rhythm_view.py``).
    """

    uses_rhythm = True

    def __init__(self, base: nn.Module, n_desc: int = N_RHYTHM) -> None:
        super().__init__()
        self.base = base
        self.kind = getattr(base, "kind", "non_linear")
        self.n_leads, self.emb_dim = int(base.n_leads), int(base.emb_dim)
        self.n_desc = int(n_desc)
        self.rhythm = nn.Linear(2 * self.n_desc, 1, bias=False)
        nn.init.zeros_(self.rhythm.weight)
        self.register_buffer("r_median", torch.zeros(self.n_desc))
        self.register_buffer("r_mean", torch.zeros(self.n_desc))
        self.register_buffer("r_std", torch.ones(self.n_desc))
        self.register_buffer("r_fitted", torch.tensor(False))

    @property
    def uses_demographics(self) -> bool:
        return bool(getattr(self.base, "uses_demographics", False))

    def set_age_stats(self, age) -> None:
        self.base.set_age_stats(age)

    def demo_matrix(self, age, sex) -> torch.Tensor:
        return self.base.demo_matrix(age, sex)

    def fit_rhythm_stats(self, raw) -> dict:
        """Median / mean / SD per descriptor from the TRAINING rows' raw ``[N, n_desc]`` (NaN = missing)."""
        r = np.asarray(raw, dtype=np.float64).reshape(-1, self.n_desc)
        fin = np.isfinite(r)
        med = np.array([np.median(r[fin[:, j], j]) if fin[:, j].any() else 0.0 for j in range(self.n_desc)])
        imp = np.where(fin, r, med)
        mean, sd = imp.mean(axis=0), imp.std(axis=0)
        sd = np.where(sd > 1e-9, sd, 1.0)
        self.r_median.copy_(torch.from_numpy(med))
        self.r_mean.copy_(torch.from_numpy(mean))
        self.r_std.copy_(torch.from_numpy(sd))
        self.r_fitted.fill_(True)
        return {"n_rows": int(len(r)), "missing_frac": [float(1.0 - f) for f in fin.mean(axis=0)],
                "median": med.tolist()}

    def rhythm_matrix(self, raw) -> torch.Tensor:
        """Raw descriptors ``[N, n_desc]`` (NaN = missing) -> ``[N, 2 n_desc]`` = ``[z, missing]``."""
        if not bool(self.r_fitted):
            raise RuntimeError("rhythm statistics were never fitted (WithRhythm.fit_rhythm_stats)")
        r = np.asarray(raw, dtype=np.float64).reshape(-1, self.n_desc)
        miss = ~np.isfinite(r)
        med = self.r_median.detach().cpu().double().numpy()
        z = (np.where(miss, med, r) - self.r_mean.detach().cpu().double().numpy()) \
            / self.r_std.detach().cpu().double().numpy()
        z = np.clip(z, -RHYTHM_Z_CLIP, RHYTHM_Z_CLIP)
        return torch.from_numpy(np.concatenate([z, miss.astype(np.float64)], axis=1).astype(np.float32))

    def term(self, rhythm: Optional[torch.Tensor], dtype: torch.dtype) -> torch.Tensor:
        if rhythm is None:
            raise ValueError("this head was fitted with the rhythm view; pass rhythm [B, 2 n_desc]")
        return self.rhythm(rhythm.to(dtype)).squeeze(1)

    def spec(self) -> dict:
        return {**head_spec(self.base), "rhythm": True}

    def penalty(self) -> torch.Tensor:
        pen = getattr(self.base, "penalty", None)
        return pen() if pen is not None else torch.zeros((), device=self.rhythm.weight.device)

    def lead_weights_now(self) -> Optional[torch.Tensor]:
        f = getattr(unwrap_head(self)[3], "lead_weights_now", None)
        return f() if f is not None else None

    def forward(self, feats: torch.Tensor, demo: Optional[torch.Tensor] = None,
                rhythm: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.base(feats, demo) if self.uses_demographics else self.base(feats)
        return out + self.term(rhythm, feats.dtype)


class LinearProbe(nn.Module):
    """A single linear layer ``in_dim -> n_classes`` for P4-style probing.

    Returns raw logits; ``n_classes == 1`` gives a binary logit ``[B]``,
    otherwise ``[B, n_classes]``.
    """

    def __init__(self, in_dim: int, n_classes: int) -> None:
        super().__init__()
        if n_classes < 1:
            raise ValueError("n_classes must be >= 1")
        self.in_dim = int(in_dim)
        self.n_classes = int(n_classes)
        self.linear = nn.Linear(self.in_dim, self.n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.shape[1] != self.in_dim:
            raise ValueError(f"expected [B, {self.in_dim}], got {tuple(x.shape)}")
        out = self.linear(x)
        return out.squeeze(1) if self.n_classes == 1 else out


__all__ = ["WithRhythm", "RHYTHM_VIEW_NAMES", "N_RHYTHM", "call_head", "unwrap_head", "InputNorm", "INPUT_NORMS", "DEFAULT_PCA_DIM", "find_input_norm", "AuxTaskHead", "aux_target_matrix", "AUX_TARGETS", "CONDUCTION_LABELS", "N_AUX", "EnsembleHead", "LeadAttentionHead", "ConcatMLPHead", "PooledEnsembleHead", "POOLS", "LinearProbe", "HEAD_KINDS",
           "build_head", "head_spec", "WithDemographics", "demographic_matrix", "age_stats", "AGE_CLIP", "N_DEMO", "ANCHOR_N_LEADS", "ANCHOR_EMB_DIM", "ANCHOR_DROPOUT"]
