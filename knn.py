from __future__ import annotations
import torch

# Optional FAISS
try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False

def torch_knn_cosine_dist(X: torch.Tensor, Y: torch.Tensor, chunk: int = 32768) -> torch.Tensor:
    """1 - max cosine similarity for each row of X against database Y. Both must be normalized."""
    Nx = X.shape[0]
    out = []
    with torch.no_grad():
        for i in range(0, Nx, chunk):
            Xi = X[i:i+chunk]
            sims = Xi @ Y.T
            m, _ = sims.max(dim=1)
            out.append(1.0 - m)
    return torch.cat(out, dim=0)

def faiss_knn_cosine_dist(X: torch.Tensor, Y: torch.Tensor, use_gpu: bool = True) -> torch.Tensor:
    """FAISS 1-NN with cosine distance (inner product on normalized vectors)."""
    if not _HAS_FAISS:
        raise RuntimeError("FAISS not installed.")
    Xc = X.detach().float().cpu().numpy()
    Yc = Y.detach().float().cpu().numpy()
    d = Yc.shape[1]
    if use_gpu and faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatIP(res, d)
    else:
        index = faiss.IndexFlatIP(d)
    index.add(Yc)
    sims, _ = index.search(Xc, 1)
    dist = 1.0 - sims[:, 0]
    return torch.from_numpy(dist)

def has_faiss() -> bool:
    return _HAS_FAISS