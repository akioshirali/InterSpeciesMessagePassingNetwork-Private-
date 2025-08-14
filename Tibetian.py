from __future__ import annotations
import os, math, random, json, argparse, time
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import csv
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
from torch.utils.data import Dataset

# Optional deps — handled gracefully if missing
try:
    import faiss                   # for kNN metrics
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

try:
    import ot                      # POT: Python Optimal Transport
    HAS_POT = True
except Exception:
    HAS_POT = False

# Encoders
try:
    from sentence_transformers import SentenceTransformer
    HAS_SBERT = True
except Exception:
    HAS_SBERT = False



# ----------------------------
# Utility and configuration
# ----------------------------

def set_seed(seed: int = 13):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def l2norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(p=2, dim=-1, keepdim=True) + eps)


def whitening(x: torch.Tensor) -> torch.Tensor:
    # Mean-center + normalize covariance diagonals (cheap whitening)
    mu = x.mean(dim=0, keepdim=True)
    xc = x - mu
    std = xc.pow(2).mean(dim=0, keepdim=True).sqrt() + 1e-8
    return xc / std


def orth_penalty(mat: torch.Tensor) -> torch.Tensor:
    # Encourage near-orthogonality of linear map (Frobenius of (W^T W - I))
    w = mat
    m = w.T @ w
    i = torch.eye(m.shape[0], device=w.device)
    return F.mse_loss(m, i)

import re

def strip_punctuation(text: str) -> str:
    # Remove all Unicode punctuation
    return re.sub(r'[\p{P}\p{S}]', '', text)

class PairedTibEngDataset(Dataset):
    def __init__(self, path: str, use_transliteration: bool = False):
        super().__init__()
        import json, regex  # use 'regex' module for \p{} Unicode categories
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.samples = []
        for entry in data:
            if use_transliteration and entry.get("transliteration"):
                tib_str = entry["transliteration"]
            else:
                tib_str = entry["tibetan"]
            eng_str = entry["english"]
            # strip punctuation from both
            tib_str = regex.sub(r'[\p{P}\p{S}]+', '', tib_str)
            eng_str = regex.sub(r'[\p{P}\p{S}]+', '', eng_str)
            self.samples.append((tib_str.strip(), eng_str.strip()))

class PairedBoEnCSVDataset(Dataset):
    """
    Reads a CSV with two columns: 'bo' (Tibetan transliteration) and 'en' (English translation).
    """
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)
        if "bo" not in df.columns or "en" not in df.columns:
            raise ValueError("CSV must have 'bo' and 'en' columns.")
        self.samples = list(zip(df["bo"].astype(str).tolist(), df["en"].astype(str).tolist()))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class PairedBoEnCollator:
    """
    Collates CSV pairs into two lists for SBERT encoders.
    """
    def __call__(self, batch):
        bo_texts = [a for a, _ in batch]
        en_texts = [b for _, b in batch]
        return bo_texts, en_texts


# ----------------------------
# Data: Tibetan–English JSON array
# ----------------------------
class PairedTibEngDataset(Dataset):
    """
    Reads translation_dataset.json with structure:
    [
      {
        "tibetan": "<tibetan text>",
        "transliteration": "<optional transliteration>",
        "english": "<english text>"
      },
      ...
    ]
    """
    def __init__(self, path: str, use_transliteration: bool = False):
        super().__init__()
        import json
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.samples = []
        for entry in data:
            if use_transliteration and entry.get("transliteration"):
                tib_str = entry["transliteration"]
            else:
                tib_str = entry["tibetan"]
            eng_str = entry["english"]
            self.samples.append((tib_str, eng_str))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class PairedTibEngCollator:
    """
    Collates Tibetan–English text pairs into two lists for SBERT encoders.
    """
    def __call__(self, batch):
        tib_texts = [a for a, _ in batch]
        eng_texts = [b for _, b in batch]
        return tib_texts, eng_texts


# ----------------------------
# Modules: Adapters, Attn, IMPN
# ----------------------------
class MLPAdapter(nn.Module):
    def __init__(self, d_in: int, d_out: int, hidden: int = 512, low_rank: Optional[int] = 0):
        super().__init__()
        self.ln = nn.LayerNorm(d_in)
        self.fc1 = nn.Linear(d_in, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, d_out)
        self.res_proj = nn.Linear(d_in, d_out) if d_in!=d_out else nn.Identity()
        # optional low-rank adapter (like LoRA-style residual)
        if low_rank and low_rank>0:
            self.A = nn.Linear(d_in, low_rank, bias=False)
            self.B = nn.Linear(low_rank, d_out, bias=False)
        else:
            self.A = None
            self.B = None

    def forward(self, x):
        z = self.fc2(self.act(self.fc1(self.ln(x))))
        z = z + self.res_proj(x)
        if self.A is not None:
            z = z + self.B(self.A(x))
        return z


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_head = d_model // n_heads
        self.n_heads = n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask: Optional[torch.Tensor]=None, temp: float=1.0):
        B, Nq, D = q.shape
        Nk = k.shape[1]
        qh = self.q(q).view(B, Nq, self.n_heads, self.d_head).transpose(1,2)
        kh = self.k(k).view(B, Nk, self.n_heads, self.d_head).transpose(1,2)
        vh = self.v(v).view(B, Nk, self.n_heads, self.d_head).transpose(1,2)
        attn = (qh @ kh.transpose(-2,-1)) / math.sqrt(self.d_head)
        attn = attn / max(1e-6, temp)
        if mask is not None:
            attn = attn.masked_fill(mask==0, float('-inf'))
        w = attn.softmax(dim=-1)
        out = (w @ vh).transpose(1,2).contiguous().view(B, Nq, D)
        return self.o(out), w


class IMPNLayer(nn.Module):
    """One block of intra-space self-attn + cross-space attention with edge MLP.
    Input: A_nodes [B, Na, Da], B_nodes [B, Nb, Db] projected to common d_msg inside.
    """
    def __init__(self, d_a: int, d_b: int, d_msg: int, self_heads: int=4, cross_heads: int=4):
        super().__init__()
        self.proj_a = nn.Linear(d_a, d_msg)
        self.proj_b = nn.Linear(d_b, d_msg)
        self.self_a = MultiHeadAttention(d_msg, self_heads)
        self.self_b = MultiHeadAttention(d_msg, self_heads)
        self.cross_a2b = MultiHeadAttention(d_msg, cross_heads)
        self.cross_b2a = MultiHeadAttention(d_msg, cross_heads)
        self.ff_a = nn.Sequential(nn.LayerNorm(d_msg), nn.Linear(d_msg, 4*d_msg), nn.GELU(), nn.Linear(4*d_msg, d_msg))
        self.ff_b = nn.Sequential(nn.LayerNorm(d_msg), nn.Linear(d_msg, 4*d_msg), nn.GELU(), nn.Linear(4*d_msg, d_msg))
        # edge function
        self.edge_mlp = nn.Sequential(
            nn.Linear(2*d_msg+1, d_msg), nn.GELU(), nn.Linear(d_msg, 1)
        )

    def forward(self, A, B, cross_mask_ab=None, cross_mask_ba=None,
                self_temp=0.5, cross_temp=0.2, topk=None):
        # Project to shared message space
        a = self.proj_a(A)
        b = self.proj_b(B)
        # Intra-space self-attn
        a_sa, wa = self.self_a(a, a, a, temp=self_temp)
        b_sb, wb = self.self_b(b, b, b, temp=self_temp)
        a = a + a_sa
        b = b + b_sb
        a = a + self.ff_a(a)
        b = b + self.ff_b(b)
        # Cross-space attention with learned edge gating
        with torch.no_grad():
            a_n = l2norm(a)
            b_n = l2norm(b)
            aff = torch.einsum('bid,bjd->bij', a_n, b_n)
        Bi, Na, D = a.shape
        Nb = b.shape[1]
        ai = a.unsqueeze(2).expand(Bi, Na, Nb, D)
        bj = b.unsqueeze(1).expand(Bi, Na, Nb, D)
        edge_feat = torch.cat([ai, bj, aff.unsqueeze(-1)], dim=-1)
        edge_gate = torch.sigmoid(self.edge_mlp(edge_feat)).squeeze(-1)
        if cross_mask_ab is not None:
            ab_mask = cross_mask_ab
        else:
            ab_mask = torch.ones_like(edge_gate, dtype=torch.bool)
        if cross_mask_ba is not None:
            ba_mask = cross_mask_ba
        else:
            ba_mask = torch.ones(Bi, Nb, Na, device=a.device, dtype=torch.bool)
            ba_mask = ba_mask.transpose(1, 2)
        a2b_out, wab = self.cross_a2b(a, b, b, mask=ab_mask.unsqueeze(1), temp=cross_temp)
        b2a_out, wba = self.cross_b2a(b, a, a, mask=ba_mask.unsqueeze(1), temp=cross_temp)
        wab_m = wab.mean(dim=1) * edge_gate
        wba_m = wba.mean(dim=1) * edge_gate.transpose(1, 2)  # <-- fixed transpose bug
        # Apply top-k masking if requested
        if topk is not None and topk > 0:
            def topk_mask(w, k):
                # Ensure k is not larger than last dimension
                k_safe = min(k, w.shape[-1])
                if k_safe <= 0:
                    return torch.zeros_like(w)
                vals, idx = torch.topk(w, k_safe, dim=-1)
                mask = torch.zeros_like(w).scatter_(-1, idx, 1.0)
                return w * mask
            wab_m = topk_mask(wab_m, topk)
            wba_m = topk_mask(wba_m, topk)
        a = a + torch.einsum('bij,bjd->bid', wab_m, b)
        b = b + torch.einsum('bji,bid->bjd', wba_m, a)
        a = a + self.ff_a(a)
        b = b + self.ff_b(b)
        return a, b, wab_m.detach(), wba_m.detach()

class IMPNBridge(nn.Module):
    def __init__(self, d_a: int, d_b: int, d_msg: int = 256,
                 n_layers: int = 3, self_heads: int = 4, cross_heads: int = 4):
        """
        Bridge network made of stacked IMPNLayer blocks.

        Args:
            d_a: Dimension of encoder A embeddings.
            d_b: Dimension of encoder B embeddings.
            d_msg: Shared message dimension.
            n_layers: Number of IMPNLayer blocks to stack.
            self_heads: Number of heads for self-attention.
            cross_heads: Number of heads for cross-attention.
        """
        super().__init__()
        self.layers = nn.ModuleList([
            IMPNLayer(
                d_a if i == 0 else d_msg,
                d_b if i == 0 else d_msg,
                d_msg,
                self_heads,
                cross_heads
            )
            for i in range(n_layers)
        ])
        self.readout_a = nn.Linear(d_msg, d_msg)
        self.readout_b = nn.Linear(d_msg, d_msg)

    def forward(self, A, B, cross_mask_ab=None, cross_mask_ba=None,
                self_temp=0.5, cross_temp=0.2, topk=None):
        """
        Run A and B through stacked IMPNLayers and return pooled representations.

        Args:
            A: Tensor [B, Na, Da] (A-side nodes)
            B: Tensor [B, Nb, Db] (B-side nodes)
            cross_mask_ab: Optional cross attention mask A->B
            cross_mask_ba: Optional cross attention mask B->A
            self_temp: Temperature for self-attention
            cross_temp: Temperature for cross-attention
            topk: Keep only top-k cross edges (if not None)
        """
        wab_logs, wba_logs = [], []
        for layer in self.layers:
            A, B, wab, wba = layer(
                A, B,
                cross_mask_ab=cross_mask_ab,
                cross_mask_ba=cross_mask_ba,
                self_temp=self_temp,
                cross_temp=cross_temp,
                topk=topk
            )
            wab_logs.append(wab)
            wba_logs.append(wba)

        # Pool over nodes to get fixed-size vectors
        Ra = self.readout_a(A).mean(dim=1)
        Rb = self.readout_b(B).mean(dim=1)
        return Ra, Rb, wab_logs, wba_logs, A, B




# ----------------------------
# VAE heads (optional)
# ----------------------------
class TinyVAE(nn.Module):
    def __init__(self, d_in: int, d_lat: int=64):
        super().__init__()
        self.enc_mu = nn.Linear(d_in, d_lat)
        self.enc_logvar = nn.Linear(d_in, d_lat)
        self.dec = nn.Sequential(nn.Linear(d_lat, d_in))

    def forward(self, x):
        mu = self.enc_mu(x)
        logv = self.enc_logvar(x)
        std = (0.5*logv).exp()
        z = mu + std * torch.randn_like(std)
        xrec = self.dec(z)
        kld = -0.5 * torch.mean(1 + logv - mu.pow(2) - logv.exp())
        rec = F.mse_loss(xrec, x)
        return xrec, rec, kld


# ----------------------------
# Full model wrapper
# ----------------------------

# ===== PARTIAL UNFREEZING LOGIC =====
class AgenticIMPNSystem(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        assert HAS_SBERT, "sentence-transformers not installed"

        # Load encoders
        self.encA = SentenceTransformer(cfg.model_a)
        self.encB = SentenceTransformer(cfg.model_b)

        # Get embedding dimensions
        self.d_a = self.encA.get_sentence_embedding_dimension()
        self.d_b = self.encB.get_sentence_embedding_dimension()

        # Adapters
        self.F = MLPAdapter(self.d_a, self.d_b)
        self.G = MLPAdapter(self.d_b, self.d_a)

        # Bridge network
        self.bridge = IMPNBridge(self.d_a, self.d_b, cfg.d_msg, cfg.n_layers, cfg.self_heads, cfg.cross_heads)

        # Optional VAE heads
        self.vaeA = TinyVAE(self.d_a) if cfg.use_vae else None
        self.vaeB = TinyVAE(self.d_b) if cfg.use_vae else None

        self.cfg = cfg

        if cfg.freeze_encoders:
            # Freeze everything
            for p in self.encA.parameters():
                p.requires_grad = False
            for p in self.encB.parameters():
                p.requires_grad = False
        else:
            # Dynamically unfreeze last N layers
            def unfreeze_last_n_layers(model, n_layers: int):
                # Collect encoder block IDs (works for BERT, MiniLM, etc.)
                block_names = sorted(
                    {name.split(".")[2] for name, _ in model.named_parameters() if "encoder.layer" in name}
                )
                last_blocks = set(block_names[-n_layers:])
                for name, p in model.named_parameters():
                    if any(f"layer.{blk}" in name for blk in last_blocks):
                        p.requires_grad = True
                    else:
                        p.requires_grad = False

            unfreeze_last_n_layers(self.encA, n_layers=2)
            unfreeze_last_n_layers(self.encB, n_layers=2)


    @torch.no_grad()
    def encode_texts(self, texts_a: List[str], texts_b: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get numpy embeddings from SBERT and immediately move to correct device/dtype
        Za = torch.from_numpy(
            self.encA.encode(texts_a, convert_to_numpy=True, normalize_embeddings=False)
        ).to(dtype=torch.float32, device=self.cfg.device)
        Zb = torch.from_numpy(
            self.encB.encode(texts_b, convert_to_numpy=True, normalize_embeddings=False)
        ).to(dtype=torch.float32, device=self.cfg.device)
        return Za, Zb


    def forward(self, Za, Zb, agent_cfg: Dict) -> Dict:
        Za_n = l2norm(Za)
        Zb_n = l2norm(Zb)
        Fa = self.F(Za_n)
        Gb = self.G(Zb_n)
        A_nodes = Za_n.unsqueeze(1)
        B_nodes = Zb_n.unsqueeze(1)
        Ra, Rb, wab_logs, wba_logs, A_lat, B_lat = self.bridge(
            A_nodes, B_nodes,
            self_temp=agent_cfg.get('self_temp', self.cfg.self_temp),
            cross_temp=agent_cfg.get('cross_temp', self.cfg.cross_temp),
            topk=agent_cfg.get('topk', self.cfg.topk_cross)
        )
        out = {
            'Za': Za_n, 'Zb': Zb_n,
            'Fa': Fa, 'Gb': Gb,
            'Ra': Ra, 'Rb': Rb,
            'wab': wab_logs, 'wba': wba_logs,
            'A_lat': A_lat, 'B_lat': B_lat,
        }
        if self.vaeA is not None:
            xrecA, recA, kldA = self.vaeA(Za_n)
            out.update({'vaeA_rec': recA, 'vaeA_kld': kldA})
        if self.vaeB is not None:
            xrecB, recB, kldB = self.vaeB(Zb_n)
            out.update({'vaeB_rec': recB, 'vaeB_kld': kldB})
        return out


# ----------------------------
# Losses
# ----------------------------
class LossComputer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.temp = 0.05

    def info_nce(self, Qa: torch.Tensor, Kb: torch.Tensor) -> torch.Tensor:
        # symmetric contrastive: cross-softmax over batch
        Qa = l2norm(Qa)
        Kb = l2norm(Kb)
        logits = Qa @ Kb.t() / self.temp
        labels = torch.arange(Qa.size(0), device=Qa.device)
        loss1 = F.cross_entropy(logits, labels)
        loss2 = F.cross_entropy(logits.t(), labels)
        return 0.5*(loss1+loss2)

    def cycle(self, Za, Zb, Fa, Gb, Fmod: MLPAdapter, Gmod: MLPAdapter) -> torch.Tensor:
        Za_hat = Gmod(Fa)
        Zb_hat = Fmod(Gb)
        return F.mse_loss(Za_hat, Za) + F.mse_loss(Zb_hat, Zb)

    def knn_preserve(self, Z: torch.Tensor, Zp: torch.Tensor, k:int=5) -> torch.Tensor:
        # Encourage neighbors to be preserved after mapping Z->Zp
        Z = l2norm(Z)
        Zp = l2norm(Zp)
        simZ = Z @ Z.t()
        simZp = Zp @ Zp.t()
        topkZ = simZ.topk(k+1, dim=-1).indices[:,1:]
        topkZp = simZp.topk(k+1, dim=-1).indices[:,1:]
        # Jaccard distance between neighbor sets
        inter = torch.stack([(topkZ[i].unsqueeze(0)==topkZp[i].unsqueeze(1)).any(dim=1).float().sum() for i in range(Z.size(0))])
        jacc = 1 - inter / k
        return jacc.mean()

    def ot_align(self, Za, Zb) -> torch.Tensor:
        if not HAS_POT or self.cfg.w_ot<=0:
            return torch.tensor(0.0, device=Za.device)
        a = torch.full((Za.size(0),), 1.0/Za.size(0), device=Za.device)
        b = torch.full((Zb.size(0),), 1.0/Zb.size(0), device=Zb.device)
        C = (1 - l2norm(Za) @ l2norm(Zb).t()).clamp(min=0)
        P = ot.sinkhorn(a.detach().cpu().numpy(), b.detach().cpu().numpy(), C.detach().cpu().numpy(), reg=0.1)
        P = torch.tensor(P, device=Za.device, dtype=Za.dtype)
        return (P * C).sum()

    def orth(self, Fmod: MLPAdapter, Gmod: MLPAdapter) -> torch.Tensor:
        """
        Apply orthogonality penalty to the main projection matrices in F and G.
        This encourages them to be close to orthonormal transforms.
        """
        pen = 0.0
        # Apply directly to the final projection weights of each adapter
        pen += orth_penalty(Fmod.fc2.weight)
        pen += orth_penalty(Gmod.fc2.weight)
        return pen

    def __call__(self, model: AgenticIMPNSystem, batch_out: Dict) -> Tuple[torch.Tensor, Dict[str,float]]:
        Za, Zb = batch_out['Za'], batch_out['Zb']
        Fa, Gb = batch_out['Fa'], batch_out['Gb']
        Ra, Rb = batch_out['Ra'], batch_out['Rb']
        losses = {}
        Lc = self.info_nce(Fa, Zb) + self.info_nce(Gb, Za) + self.info_nce(Ra, Rb)
        losses['contrast'] = Lc
        Lcyc = self.cycle(Za, Zb, Fa, Gb, model.F, model.G)
        losses['cycle'] = Lcyc
        Lknn = self.knn_preserve(Za, Fa) + self.knn_preserve(Zb, Gb)
        losses['knn'] = Lknn
        Lorth = self.orth(model.F, model.G)
        losses['orth'] = Lorth
        Lot = self.ot_align(Fa, Zb) if self.cfg.w_ot>0 else torch.tensor(0.0, device=Za.device)
        losses['ot'] = Lot
        # optional VAE
        if model.vaeA is not None:
            losses['vaeA'] = batch_out['vaeA_rec'] + 0.1*batch_out['vaeA_kld']
        if model.vaeB is not None:
            losses['vaeB'] = batch_out['vaeB_rec'] + 0.1*batch_out['vaeB_kld']
        # total
        total = (self.cfg.w_contrast*losses['contrast'] +
                 self.cfg.w_cycle*losses['cycle'] +
                 self.cfg.w_knn*losses['knn'] +
                 self.cfg.w_orth*losses['orth'] +
                 self.cfg.w_ot*losses['ot'] +
                 losses.get('vaeA', 0.0) + losses.get('vaeB', 0.0))
        return total, {k: v.item() if torch.is_tensor(v) else float(v) for k,v in losses.items()}


# ----------------------------
# Agentic loop
# ----------------------------
class Planner:
    def propose(self, step:int, stats:Dict[str,float], cfg:Config) -> Dict:
        # Simple heuristic: if contrast high, reduce temp; if cycle high, increase top-k
        agent_cfg = {}
        agent_cfg['self_temp'] = max(0.2, cfg.self_temp * (0.95 if stats.get('contrast',0)>1.0 else 1.0))
        agent_cfg['cross_temp'] = max(0.1, cfg.cross_temp * (0.95 if stats.get('cycle',0)>0.3 else 1.0))
        agent_cfg['topk'] = cfg.topk_cross
        return agent_cfg

class Actor:
    def run(self, model: AgenticIMPNSystem, Za, Zb, agent_cfg: Dict) -> Dict:
        return model(Za, Zb, agent_cfg)

class Critic:
    def score(self, losses: Dict[str,float]) -> float:
        # Lower losses -> higher score
        return - (losses.get('contrast',0) + 0.5*losses.get('cycle',0))


# ----------------------------
# Training and evaluation
# ----------------------------

def evaluate_retrieval(model: AgenticIMPNSystem, texts_a: List[str], texts_b: List[str], k: int = 5) -> Dict[str, float]:
    """
    Evaluate retrieval accuracy by mapping both sides into a common space (F(a) vs Zb).
    Returns R@1, R@k, and MRR.
    """
    model.eval()
    with torch.no_grad():
        Za, Zb = model.encode_texts(texts_a, texts_b)
        out = model(Za, Zb, {
            'self_temp': model.cfg.self_temp,
            'cross_temp': model.cfg.cross_temp,
            'topk': model.cfg.topk_cross
        })
        # Use F(a) vs original Zb embeddings for retrieval
        Qa = l2norm(out['Fa'])
        Kb = l2norm(out['Zb'])  # Could swap for Gb if you want symmetrical space

        sims = (Qa @ Kb.t()).cpu()
        ranks = []
        for i in range(sims.size(0)):
            order = torch.argsort(sims[i], descending=True)
            rank = (order == i).nonzero(as_tuple=False)
            r = int(rank[0, 0].item()) + 1 if rank.numel() > 0 else sims.size(1)
            ranks.append(r)

        r1 = sum(1 for r in ranks if r <= 1) / len(ranks)
        rk = sum(1 for r in ranks if r <= k) / len(ranks)
        mrr = float(sum(1.0 / r for r in ranks) / len(ranks))

    return {"R@1": r1, f"R@{k}": rk, "MRR": mrr}

def save_loss_history(loss_history: dict, path: str):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(loss_history.keys())
        min_len = min(len(v) for v in loss_history.values())
        for i in range(min_len):
            writer.writerow([loss_history[k][i] for k in loss_history.keys()])
    print(f"Loss history saved to {path}")

def plot_loss_history(loss_history: dict, output_path: str):
    steps = loss_history["step"]
    loss_keys = [k for k in loss_history if k != "step" and not k.startswith("eval_")]
    metric_keys = [k for k in loss_history if k.startswith("eval_")]

    # Plot loss curves
    n_losses = len(loss_keys)
    fig, axes = plt.subplots(n_losses, 1, figsize=(10, 3 * n_losses), sharex=True)
    if n_losses == 1:
        axes = [axes]
    for ax, key in zip(axes, loss_keys):
        ax.plot(steps[:len(loss_history[key])], loss_history[key], label=key)
        ax.set_ylabel(key)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
    axes[-1].set_xlabel("Step")
    fig.suptitle("Training Loss Curves (Separate Scales)", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path.replace(".png", "_losses.png"))
    plt.close(fig)

    # Metric curves
    plt.figure(figsize=(10, 6))
    for key in metric_keys:
        plt.plot(
            steps[:len(loss_history[key])],
            loss_history[key],
            label=key
        )
    plt.xlabel("Step")
    plt.ylabel("Score")
    plt.title("Retrieval Metrics Over Training")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.savefig(output_path.replace(".png", "_metrics.png"))
    plt.close()



# ===== TRAIN DEMO CHANGES =====

def train_demo(cfg: Config, data_path: str, output_root: str) -> Tuple[nn.Module, str]:
    import datetime, uuid
    os.makedirs(output_root, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(output_root, ts)
    if os.path.exists(run_dir):
        run_dir = f"{run_dir}_{uuid.uuid4().hex[:6]}"
    os.makedirs(run_dir, exist_ok=False)

    def _save_ckpt(path: str, model_obj: AgenticIMPNSystem, cfg_obj: Config):
        torch.save({
            "F": model_obj.F.state_dict(),
            "G": model_obj.G.state_dict(),
            "bridge": model_obj.bridge.state_dict(),
            "cfg": asdict(cfg_obj),
        }, path)
        print(f"[Checkpoint] Saved to {path}")

    # Safe CSV writer
    def _save_loss_history_safe(loss_history: dict, path: str):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(loss_history.keys())
            min_len = min(len(v) for v in loss_history.values())
            for i in range(min_len):
                writer.writerow([loss_history[k][i] for k in loss_history.keys()])
        print(f"[Loss History] Saved to {path}")

    set_seed(123)
    device = cfg.device
    ds = PairedBoEnCSVDataset(data_path)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=PairedBoEnCollator())

    model = AgenticIMPNSystem(cfg).to(device)
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optim = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=cfg.lr)
    losscomp = LossComputer(cfg)
    planner, actor, critic = Planner(), Actor(), Critic()

    texts_all_a = [a for a, _ in ds.samples]
    texts_all_b = [b for _, b in ds.samples]

    csv_path = os.path.join(run_dir, "loss_history.csv")
    plot_path = os.path.join(run_dir, "loss_plot.png")
    latest_ckpt_path = os.path.join(run_dir, "latest.pt")

    loss_history: Dict[str, List[float]] = {
        "step": [],
        "total": [],
        "eval_R@1": [],
        "eval_R@5": [],
        "eval_MRR": [],
    }

    global_step = 0
    t0 = time.time()
    last_stats: Dict[str, float] = {}

    print("[Training] Starting infinite loop. Stop manually with Ctrl+C.")

    try:
        while True:
            for A, B in dl:
                model.train()
                Za, Zb = model.encode_texts(A, B)
                Za, Zb = Za.to(device), Zb.to(device)

                agent_cfg = planner.propose(global_step, last_stats, cfg)
                out = actor.run(model, Za, Zb, agent_cfg)

                total, losses = losscomp(model, out)
                last_stats = losses

                optim.zero_grad()
                total.backward()
                torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
                optim.step()

                loss_history["step"].append(global_step)
                loss_history["total"].append(total.item())
                for k, v in losses.items():
                    if k not in loss_history:
                        loss_history[k] = []
                    loss_history[k].append(v)

                # Log and evaluate at specified intervals
                if global_step % cfg.log_every == 0:
                    et = time.time() - t0
                    retr = evaluate_retrieval(model, texts_all_a[:128], texts_all_b[:128])

                    loss_history["eval_R@1"].append(retr["R@1"])
                    loss_history["eval_R@5"].append(retr["R@5"])
                    loss_history["eval_MRR"].append(retr["MRR"])

                    msg = {
                        "step": global_step,
                        "time_s": round(et, 2),
                        "loss": round(float(total.item()), 4),
                        **{k: round(v, 4) for k, v in losses.items()},
                        **{f"eval_{k}": round(v, 4) for k, v in retr.items()},
                    }
                    print(json.dumps(msg))

                # Save checkpoint and plots every 100 steps
                if global_step % 100 == 0 and global_step > 0:
                    _save_ckpt(latest_ckpt_path, model, cfg)
                    plot_loss_history(loss_history, plot_path)
                    _save_loss_history_safe(loss_history, csv_path)

                global_step += 1

    except KeyboardInterrupt:
        print("\n[Training] Stopped by user.")
        _save_ckpt(latest_ckpt_path, model, cfg)
        plot_loss_history(loss_history, plot_path)
        _save_loss_history_safe(loss_history, csv_path)
        print(f"[Artifacts] Saved in: {run_dir}")

    return model, run_dir



# ----------------------------
# CLI
# ----------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--demo', action='store_true', help='Run demo train loop on dataset')
    p.add_argument('--steps', type=int, default=400)
    p.add_argument('--data_path', type=str, default='eng_fra_pairs.jsonl',
                   help='Path to JSONL file with {"a": english, "b": french}')
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out_dir', type=str, default='runs',
                   help='Root output folder; each run creates a timestamped subfolder here')
    args = p.parse_args()

    cfg = Config()
    cfg.max_steps = int(args.steps)
    cfg.device = args.device

    print("Config:", json.dumps(asdict(cfg), indent=2))
    run_dir = make_run_dir(args.out_dir)
    print(f"[IMPN] Writing artifacts to: {run_dir}")
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    if args.demo:
        model, run_dir = train_demo(cfg, args.data_path, run_dir)  # <-- unpack tuple
        ckpt_path = os.path.join(run_dir, 'agentic_impnn_demo.pt')
        torch.save({
            'F': model.F.state_dict(),
            'G': model.G.state_dict(),
            'bridge': model.bridge.state_dict(),
            'cfg': asdict(cfg),
        }, ckpt_path)
        print(f"Saved checkpoint -> {ckpt_path}")


if __name__ == '__main__':
    main()
