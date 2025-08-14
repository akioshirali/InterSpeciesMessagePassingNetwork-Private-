 
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
import numpy as np


ENCODERS = {
    "Image:ResNet50":       {"encoder": "ResNet50 (ImageNet)",                  "latent_dim": 2048},
    "Image:CLIP-ViT-B32":   {"encoder": "CLIP ViT-B/32",                        "latent_dim": 512},
    "Image:EfficientNetB4": {"encoder": "EfficientNet-B4",                      "latent_dim": 1792},

    "Text:BERT":            {"encoder": "BERT-base-uncased",                    "latent_dim": 768},
    "Text:SentenceVec":     {"encoder": "sentence-transformers/all-MiniLM-L6-v2","latent_dim": 384},

    "Video:TimeSformer":    {"encoder": "TimeSformer",                          "latent_dim": 1024},

    "Voice:Wav2Vec2":       {"encoder": "Wav2Vec2-base",                        "latent_dim": 768},
    "Voice:Whisper":        {"encoder": "OpenAI Whisper-small",                 "latent_dim": 1024},
    "Voice:ECAPA-TDNN":     {"encoder": "ECAPA-TDNN",                           "latent_dim": 192}
}



os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppresses most TF/XLA/CUDA warnings

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

# ========= MANIFEST-based precomputed latent dataset =========
class ManifestLatentDataset(Dataset):
    """
    Loads two modalities (A,B) from a MANIFEST.json produced by your generator.
    Pairs are index-aligned up to min length by default.

    Robust path resolution:
      - Accepts absolute paths as-is when they exist.
      - If not found, resolves relative to the MANIFEST's directory.
      - If still not found, tries just the basename inside the MANIFEST's directory.
    """
    def __init__(self,
                 manifest_path: str,
                 enc_key_a: str,
                 enc_key_b: str,
                 pairing: str = "index",        # "index" | "min" | "repeat_shorter"
                 mmap: bool = True):
        super().__init__()

        # Normalize to MANIFEST.json file path
        if os.path.isdir(manifest_path):
            manifest_path = os.path.join(manifest_path, "MANIFEST.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"MANIFEST not found: {manifest_path}")
        self._manifest_dir = os.path.dirname(os.path.abspath(manifest_path))

        with open(manifest_path, "r", encoding="utf-8") as f:
            mani = json.load(f)

        # Index items by info_type and pick the two we need
        items_by_key = {it["info_type"]: it for it in mani.get("items", [])}
        if enc_key_a not in items_by_key:
            raise KeyError(f"{enc_key_a} not in MANIFEST items: {list(items_by_key.keys())}")
        if enc_key_b not in items_by_key:
            raise KeyError(f"{enc_key_b} not in MANIFEST items: {list(items_by_key.keys())}")

        self.a_info = items_by_key[enc_key_a]
        self.b_info = items_by_key[enc_key_b]

        # Resolve file paths robustly
        a_path = self._resolve_path(self.a_info.get("path", ""))
        b_path = self._resolve_path(self.b_info.get("path", ""))

        # Load (or memmap) arrays
        self.A = np.load(a_path, mmap_mode=("r" if mmap else None), allow_pickle=False)
        self.B = np.load(b_path, mmap_mode=("r" if mmap else None), allow_pickle=False)

        if self.A.ndim != 2 or self.B.ndim != 2:
            raise ValueError(f"Expect 2D [N, D] arrays, got A={self.A.shape}, B={self.B.shape}")

        self.na, self.da = int(self.A.shape[0]), int(self.A.shape[1])
        self.nb, self.db = int(self.B.shape[0]), int(self.B.shape[1])

        self.pairing = pairing
        if pairing in ("index", "min"):
            self.N = int(min(self.na, self.nb))
        elif pairing == "repeat_shorter":
            self.N = int(max(self.na, self.nb))
        else:
            raise ValueError(f"Unknown pairing strategy: {pairing}")

        # Expose dims for model construction overrides
        self.dim_a = int(self.da)
        self.dim_b = int(self.db)

    def _resolve_path(self, raw_path: str) -> str:
        """
        Try a few strategies to locate the .npy path referenced in the manifest.
        """
        # 1) As-is (absolute or current working dir)
        if raw_path and os.path.isabs(raw_path) and os.path.exists(raw_path):
            return raw_path
        if raw_path and os.path.exists(raw_path):
            return raw_path

        # 2) Relative to MANIFEST directory
        if raw_path:
            cand = os.path.join(self._manifest_dir, raw_path)
            if os.path.exists(cand):
                return cand

        # 3) Basename inside MANIFEST directory
        base = os.path.basename(raw_path) if raw_path else ""
        if base:
            cand2 = os.path.join(self._manifest_dir, base)
            if os.path.exists(cand2):
                return cand2

        raise FileNotFoundError(
            f"Could not resolve data file '{raw_path}'. "
            f"Tried: as-is, relative to MANIFEST dir, and basename in MANIFEST dir ({self._manifest_dir})."
        )

    def __len__(self) -> int:
        # Must return a real Python int for PyTorch samplers
        return int(self.N)

    def __getitem__(self, idx: int):
        if self.pairing in ("index", "min"):
            ia = idx
            ib = idx
        elif self.pairing == "repeat_shorter":
            ia = idx % self.na
            ib = idx % self.nb
        else:
            raise RuntimeError("Unreachable pairing mode.")

        # Ensure float32 arrays (torch will tensorize them in the collator)
        a_vec = np.asarray(self.A[ia], dtype=np.float32)
        b_vec = np.asarray(self.B[ib], dtype=np.float32)
        return {"a_vec": a_vec, "b_vec": b_vec}


class ManifestLatentCollator:
    """
    Collates a batch of manifest-loaded latent pairs into the vector mode
    expected by AgenticIMPNSystem.encode_pairs (i.e., lists of float32 rows).
    """
    def __call__(self, batch):
        # batch is a list of {"a_vec": np.ndarray[d_a], "b_vec": np.ndarray[d_b]}
        A_vec = [np.asarray(item["a_vec"], dtype=np.float32) for item in batch]
        B_vec = [np.asarray(item["b_vec"], dtype=np.float32) for item in batch]
        return {
            "mode": "vec",
            "A_vec": A_vec,
            "B_vec": B_vec,
        }


    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx):
        if self.pairing in ("index", "min"):
            ia = idx
            ib = idx
        elif self.pairing == "repeat_shorter":
            ia = idx % self.na
            ib = idx % self.nb
        else:
            raise RuntimeError("Unreachable")

        # Ensure float32 tensors downstream
        a_vec = np.asarray(self.A[ia], dtype=np.float32)
        b_vec = np.asarray(self.B[ib], dtype=np.float32)
        return {"a_vec": a_vec, "b_vec": b_vec}



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

@dataclass
class Config:
    # Keys from ENCODERS dict for each side
    enc_key_a: str = "Text:SentenceVec"
    enc_key_b: str = "Text:SentenceVec"

    freeze_encoders: bool = False
    use_vae: bool = False

    d_msg: int = 256
    n_layers: int = 3
    self_heads: int = 4
    cross_heads: int = 4

    topk_cross: int = 8
    cross_temp: float = 0.2
    self_temp: float = 0.5

    w_contrast: float = 1.0
    w_cycle: float = 0.5
    w_knn: float = 0.2
    w_orth: float = 1e-3
    w_ot: float = 0.0

    lr: float = 1e-4
    batch_size: int = 16
    max_steps: int = 5000
    log_every: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    agent_iters: int = 1


def make_run_dir(output_root: str) -> str:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(output_root, ts)
    # ensure uniqueness if run within the same second
    i = 1
    base = run_dir
    while os.path.exists(run_dir):
        run_dir = f"{base}_{i}"
        i += 1
    os.makedirs(run_dir, exist_ok=False)
    return run_dir




# ----------------------------
# Data: Paied Generics Eng–Fra JSONL
# ----------------------------
class PairedGenericDataset(Dataset):
    """
    JSONL lines can be:
      {"a": {"mod": "Text", "data": "hello"}, "b": {"mod": "Image", "data": "path.jpg"}}
    or precomputed:
      {"a_vec": [...], "b_vec": [...]}
    """
    def __init__(self, path: str):
        self.samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class PairedGenericCollator:
    def __call__(self, batch):
        if "a_vec" in batch[0]:
            return {
                "mode": "vec",
                "A_vec": [item["a_vec"] for item in batch],
                "B_vec": [item["b_vec"] for item in batch]
            }
        else:
            return {
                "mode": "raw",
                "A_mod": batch[0]["a"]["mod"],
                "B_mod": batch[0]["b"]["mod"],
                "A_data": [item["a"]["data"] for item in batch],
                "B_data": [item["b"]["data"] for item in batch]
            }


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
class MultiEncoder(nn.Module):
    def __init__(self, enc_key: str, device: str):
        super().__init__()
        assert enc_key in ENCODERS, f"Unknown encoder key: {enc_key}"
        self.key = enc_key
        self.device = device
        self.latent_dim = ENCODERS[enc_key]["latent_dim"]
        self.modality = enc_key.split(":")[0]
        self._load_model()

    def _load_model(self):
        mod = self.modality
        name = ENCODERS[self.key]["encoder"]

        if mod == "Text":
            if "SentenceTransformer" in name:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(name)
                self.encode_fn = self._encode_sbert
            else:
                from transformers import AutoTokenizer, AutoModel
                self.tokenizer = AutoTokenizer.from_pretrained(name)
                self.model = AutoModel.from_pretrained(name).to(self.device)
                self.encode_fn = self._encode_hf_text

        elif mod == "Image":
            import torchvision.transforms as T
            from PIL import Image
            self.pre = T.Compose([
                T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            if "ResNet50" in name:
                import torchvision.models as models
                self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2).to(self.device)
            elif "EfficientNet-B4" in name:
                import timm
                self.model = timm.create_model("efficientnet_b4.ra2_in1k", pretrained=True).to(self.device)
            self.encode_fn = self._encode_image

        elif mod == "Voice":
            import torchaudio
            self.torchaudio = torchaudio
            from transformers import Wav2Vec2Processor, Wav2Vec2Model
            self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(self.device)
            self.encode_fn = self._encode_audio

        elif mod == "Video":
            raise NotImplementedError("Video encoder loading here")

    @torch.no_grad()
    def encode(self, batch_inputs):
        return self.encode_fn(batch_inputs)

    def _encode_sbert(self, texts):
        arr = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=False)
        return torch.from_numpy(arr).to(self.device, dtype=torch.float32)

    def _encode_hf_text(self, texts):
        toks = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        out = self.model(**toks).last_hidden_state.mean(dim=1)
        return out

    def _encode_image(self, paths):
        from PIL import Image
        imgs = [self.pre(Image.open(p).convert("RGB")) for p in paths]
        x = torch.stack(imgs).to(self.device)
        feats = self.model.forward_features(x) if hasattr(self.model, "forward_features") else self.model(x)
        if feats.ndim == 4:
            feats = feats.mean(dim=[2, 3])
        return feats

    def _encode_audio(self, paths):
        outs = []
        for p in paths:
            wav, sr = self.torchaudio.load(p)
            wav = self.torchaudio.functional.resample(wav, sr, 16000).mean(dim=0)
            inputs = self.processor(wav.numpy(), sampling_rate=16000, return_tensors="pt").to(self.device)
            out = self.model(**inputs).last_hidden_state.mean(dim=1).squeeze(0)
            outs.append(out)
        return torch.stack(outs).to(self.device)
        

class AgenticIMPNSystem(nn.Module):
    def __init__(self,
                 cfg: Config,
                 enc_key_a: Optional[str] = None,
                 enc_key_b: Optional[str] = None,
                 d_a_override: Optional[int] = None,
                 d_b_override: Optional[int] = None,
                 disable_encoders: bool = False):
        super().__init__()
        self.cfg = cfg

        # Resolve encoder keys (used only if encoders are enabled)
        self.enc_key_a = enc_key_a or getattr(cfg, "enc_key_a", "Text:SentenceVec")
        self.enc_key_b = enc_key_b or getattr(cfg, "enc_key_b", "Text:SentenceVec")

        # If we're training from precomputed latents, we don't need to load encoders at all.
        self.use_encoders = not disable_encoders

        if self.use_encoders:
            # Uses your MultiEncoder wrapper (from earlier patch)
            self.encA = MultiEncoder(self.enc_key_a, cfg.device)
            self.encB = MultiEncoder(self.enc_key_b, cfg.device)
            self.d_a = d_a_override or self.encA.latent_dim
            self.d_b = d_b_override or self.encB.latent_dim
        else:
            # Must specify latent dims via overrides
            assert d_a_override is not None and d_b_override is not None, \
                "When disable_encoders=True, you must pass d_a_override and d_b_override"
            self.encA = None
            self.encB = None
            self.d_a = int(d_a_override)
            self.d_b = int(d_b_override)

        # Adapters and bridge
        self.F = MLPAdapter(self.d_a, self.d_b)
        self.G = MLPAdapter(self.d_b, self.d_a)
        self.bridge = IMPNBridge(self.d_a, self.d_b,
                                 cfg.d_msg, cfg.n_layers, cfg.self_heads, cfg.cross_heads)

        # Optional VAE heads
        self.vaeA = TinyVAE(self.d_a) if cfg.use_vae else None
        self.vaeB = TinyVAE(self.d_b) if cfg.use_vae else None

    @torch.no_grad()
    def encode_pairs(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        - 'vec' mode: batch contains precomputed arrays (we just stack/return).
        - 'raw' mode: use encoders to produce latents from inputs.
        """
        if batch["mode"] == "vec":
            Za = torch.tensor(batch["A_vec"], dtype=torch.float32, device=self.cfg.device)
            Zb = torch.tensor(batch["B_vec"], dtype=torch.float32, device=self.cfg.device)
            return Za, Zb

        # raw path (requires encoders)
        assert self.use_encoders and self.encA is not None and self.encB is not None, \
            "Raw mode requested but encoders are disabled"
        Za = self.encA.encode(batch["A_data"]).to(self.cfg.device, dtype=torch.float32)
        Zb = self.encB.encode(batch["B_data"]).to(self.cfg.device, dtype=torch.float32)
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

def evaluate_retrieval_generic(model: nn.Module,
                               eval_path: str,
                               batch_size: int = 64,
                               k: int = 5,
                               device: Optional[str] = None,
                               max_eval: Optional[int] = 2048) -> Dict[str, float]:
    """
    Modality-agnostic retrieval evaluation:
      - Loads pairs (raw or precomputed) from a JSONL at `eval_path`
      - Encodes with model.encode_pairs
      - Computes retrieval with F(a) (queries) vs Zb (keys)

    Returns: {"R@1": ..., f"R@{k}": ..., "MRR": ...}
    """
    device = device or getattr(model, "device", "cuda" if torch.cuda.is_available() else "cpu")
    ds = PairedGenericDataset(eval_path)
    if max_eval is not None and len(ds) > max_eval:
        # slice a deterministic prefix for speed
        ds.samples = ds.samples[:max_eval]
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=PairedGenericCollator())

    model.eval()
    Qa_list, Kb_list = [], []

    with torch.no_grad():
        for batch in dl:
            Za, Zb = model.encode_pairs(batch)     # [B, d_a], [B, d_b]
            Za = Za.to(device); Zb = Zb.to(device)
            # forward once to get Fa and Zb (original keys)
            out = model(Za, Zb, {
                'self_temp': model.cfg.self_temp,
                'cross_temp': model.cfg.cross_temp,
                'topk': model.cfg.topk_cross
            })
            Qa = l2norm(out['Fa'])   # queries from side A after adapter F
            Kb = l2norm(out['Zb'])   # keys from side B (pre-adapter) to match training objective
            Qa_list.append(Qa.detach().cpu())
            Kb_list.append(Kb.detach().cpu())

    Qa_all = torch.cat(Qa_list, dim=0)  # [N, d']
    Kb_all = torch.cat(Kb_list, dim=0)  # [N, d_b]
    sims = Qa_all @ Kb_all.t()          # [N, N]
    ranks = []
    for i in range(sims.size(0)):
        order = torch.argsort(sims[i], descending=True)
        where = (order == i).nonzero(as_tuple=False)
        r = int(where[0, 0].item()) + 1 if where.numel() > 0 else sims.size(1)
        ranks.append(r)

    r1 = sum(1 for r in ranks if r <= 1) / len(ranks)
    rk = sum(1 for r in ranks if r <= k) / len(ranks)
    mrr = float(sum(1.0 / r for r in ranks) / len(ranks))
    return {"R@1": r1, f"R@{k}": rk, "MRR": mrr}




def save_loss_history(loss_history: dict, path: str):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(loss_history.keys())
        for i in range(len(loss_history["step"])):
            writer.writerow([loss_history[k][i] for k in loss_history.keys()])
    print(f"Loss history saved to {path}")



def plot_loss_history(loss_history: dict, output_path: str):
    steps = loss_history["step"]
    loss_keys = [k for k in loss_history if k != "step" and not k.startswith("eval_")]
    metric_keys = [k for k in loss_history if k.startswith("eval_")]

    # Loss curves
    n_losses = len(loss_keys)
    fig, axes = plt.subplots(n_losses, 1, figsize=(10, 3 * n_losses), sharex=True)
    if n_losses == 1:
        axes = [axes]
    for ax, key in zip(axes, loss_keys):
        ax.plot(steps, loss_history[key], label=key)
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
        plt.plot(steps, loss_history[key], label=key)
    plt.xlabel("Step")
    plt.ylabel("Score")
    plt.title("Retrieval Metrics Over Training")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.savefig(output_path.replace(".png", "_metrics.png"))
    plt.show()



# ===== TRAIN DEMO CHANGES =====

def _looks_like_manifest(path: str) -> bool:
    if os.path.isdir(path):
        return os.path.exists(os.path.join(path, "MANIFEST.json"))
    base = os.path.basename(path).lower()
    return base == "manifest.json" or "manifest" in base

def train_demo(cfg: Config, data_path: str, output_root: str) -> Tuple[nn.Module, str]:
    import datetime, uuid, csv
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

    set_seed(123)
    device = cfg.device

    use_manifest = _looks_like_manifest(data_path)
    if use_manifest:
        # MANIFEST-based dataset (precomputed latents)
        train_ds = ManifestLatentDataset(
            manifest_path=data_path,
            enc_key_a=cfg.enc_key_a,
            enc_key_b=cfg.enc_key_b,
            pairing="index"  # or "repeat_shorter" if lengths differ a lot
        )
        dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=ManifestLatentCollator())
        # Build a model with encoder disabled and dims taken from dataset
        model = AgenticIMPNSystem(cfg,
                                  enc_key_a=cfg.enc_key_a,
                                  enc_key_b=cfg.enc_key_b,
                                  d_a_override=train_ds.dim_a,
                                  d_b_override=train_ds.dim_b,
                                  disable_encoders=True).to(device)

        # Build a tiny eval JSONL from a slice of MANIFEST pairs to reuse the generic evaluator
        eval_jsonl = os.path.join(run_dir, "eval_from_manifest.jsonl")
        n_eval = max(200, min(len(train_ds), int(0.1 * len(train_ds))))
        with open(eval_jsonl, "w", encoding="utf-8") as f:
            for i in range(n_eval):
                rec = train_ds[i]
                f.write(json.dumps({"a_vec": rec["a_vec"].tolist(), "b_vec": rec["b_vec"].tolist()}) + "\n")
        eval_path = eval_jsonl

    else:
        # JSONL dataset (raw or precomputed vecs), from earlier patches
        train_ds = PairedGenericDataset(data_path)
        dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=PairedGenericCollator())
        # Use encoders (or pass-through if 'vec' mode per batch)
        model = AgenticIMPNSystem(cfg).to(device)

        # Simple holdout file for eval
        eval_path = os.path.join(run_dir, "eval_holdout.jsonl")
        with open(eval_path, "w", encoding="utf-8") as f:
            take = max(200, min(len(train_ds), int(0.1 * len(train_ds))))
            for obj in train_ds.samples[:take]:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optim = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=cfg.lr)
    losscomp = LossComputer(cfg)
    planner, actor, critic = Planner(), Actor(), Critic()

    # Artifacts
    csv_path = os.path.join(run_dir, "loss_history.csv")
    plot_path = os.path.join(run_dir, "loss_plot.png")
    latest_ckpt_path = os.path.join(run_dir, "latest.pt")
    final_ckpt_path = os.path.join(run_dir, "final.pt")

    # History dict
    loss_history: Dict[str, List[float]] = {
        "step": [],
        "total": [],
        "eval_R@1": [],
        "eval_R@5": [],
        "eval_MRR": [],
    }

    def save_loss_history(loss_history: dict, path: str):
        keys = list(loss_history.keys())
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(keys)
            rows = zip(*[loss_history[k] for k in keys])
            for row in rows:
                writer.writerow(row)

    global_step = 0
    t0 = time.time()
    last_stats: Dict[str, float] = {}

    while True:
        for batch in dl:
            model.train()
            Za, Zb = model.encode_pairs(batch)
            Za, Zb = Za.to(device), Zb.to(device)

            agent_cfg = planner.propose(global_step, last_stats, cfg)
            out = actor.run(model, Za, Zb, agent_cfg)

            total, losses = losscomp(model, out)
            last_stats = losses

            optim.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
            optim.step()

            # Log per-step
            loss_history["step"].append(global_step)
            loss_history["total"].append(float(total.item()))
            for k, v in losses.items():
                if k not in loss_history:
                    loss_history[k] = []
                loss_history[k].append(float(v))

            # Periodic eval + checkpoint
            if (global_step % cfg.log_every) == 0:
                et = time.time() - t0
                retr = evaluate_retrieval_generic(
                    model, eval_path,
                    batch_size=min(128, cfg.batch_size*2),
                    k=5, device=device
                )
                loss_history["eval_R@1"].append(float(retr["R@1"]))
                loss_history["eval_R@5"].append(float(retr["R@5"]))
                loss_history["eval_MRR"].append(float(retr["MRR"]))

                msg = {
                    "step": global_step,
                    "time_s": round(et, 2),
                    "loss": round(float(total.item()), 4),
                    **{k: round(float(v), 4) for k, v in losses.items()},
                    **{f"eval_{k}": round(float(v), 4) for k, v in retr.items()},
                }
                print(json.dumps(msg))
                save_loss_history(loss_history, csv_path)
                _save_ckpt(latest_ckpt_path, model, cfg)

            global_step += 1
            if global_step >= cfg.max_steps:
                save_loss_history(loss_history, csv_path)
                try:
                    plot_loss_history(loss_history, plot_path)
                except Exception as e:
                    print(f"Plotting failed (ok): {e}")
                _save_ckpt(final_ckpt_path, model, cfg)
                print(f"Artifacts saved in: {run_dir}")
                return model, run_dir


# ----------------------------
# CLI
# ----------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--demo', action='store_true', help='Run demo train loop on dataset')
    p.add_argument('--steps', type=int, default=1000, help='Max training steps')
    p.add_argument('--data_path', type=str, default='pairs.jsonl',
                   help='Either a JSONL file (raw/vec) or a MANIFEST.json (or its directory)')
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out_dir', type=str, default='runs',
                   help='Root output folder; each run creates a timestamped subfolder here')
    p.add_argument('--enc_a', type=str, default='Text:SentenceVec',
                   help='Encoder key for side A (must be a key in ENCODERS)')
    p.add_argument('--enc_b', type=str, default='Text:SentenceVec',
                   help='Encoder key for side B (must be a key in ENCODERS)')
    args = p.parse_args()

    cfg = Config(
        enc_key_a=args.enc_a,
        enc_key_b=args.enc_b,
        device=args.device,
    )
    cfg.max_steps = int(args.steps)

    print("Config:", json.dumps(asdict(cfg), indent=2))
    run_dir = make_run_dir(args.out_dir)
    print(f"[IMPN] Writing artifacts to: {run_dir}")
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    if args.demo:
        model, run_dir = train_demo(cfg, args.data_path, run_dir)
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
