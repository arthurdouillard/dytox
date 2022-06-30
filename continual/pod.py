import torch
from torch.nn import functional as F


def pod_loss(feats, old_feats, scales=[1], normalize=True):
    loss = 0.

    assert len(feats) == len(old_feats)

    for feat, old_feat in zip(feats, old_feats):
        emb = _local_pod(feat, scales)
        old_emb = _local_pod(old_feat, scales)

        if normalize:
            emb = F.normalize(emb, p=2, dim=-1)
            old_emb = F.normalize(old_emb, p=2, dim=-1)

        loss += torch.frobenius_norm(emb - old_emb, dim=-1)

    return loss.mean() / len(feats)


def _local_pod(x, spp_scales=[1, 2, 4]):
    b = x.shape[0]
    w = x.shape[-1]
    emb = []

    for scale in spp_scales:
        k = w // scale
        for i in range(scale):
            for j in range(scale):
                tensor = x[..., i * k:(i + 1) * k, j * k:(j + 1) * k]

                horizontal_pool = tensor.mean(dim=3).view(b, -1)
                vertical_pool = tensor.mean(dim=2).view(b, -1)
                emb.append(horizontal_pool)
                emb.append(vertical_pool)

    return torch.cat(emb, dim=1)
