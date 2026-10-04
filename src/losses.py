import torch, torch.nn as nn, torch.nn.functional as F


def lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    inter = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jac = 1.0 - inter / union
    if p > 1:
        jac[1:p] = jac[1:p] - jac[0:-1]
    return jac


def lovasz_softmax(logits, target, max_px=600_000):
    """mIoU(Jaccard)를 직접 최적화하는 surrogate.
    픽셀이 많으면 클래스별 정렬 비용이 커지므로 랜덤 서브샘플한다."""
    probas = logits.softmax(1)
    C = probas.shape[1]
    probas = probas.permute(0, 2, 3, 1).reshape(-1, C)
    labels = target.reshape(-1)
    if labels.numel() > max_px:
        idx = torch.randperm(labels.numel(), device=labels.device)[:max_px]
        probas, labels = probas[idx], labels[idx]
    losses = []
    for c in range(C):
        fg = (labels == c).float()
        if fg.sum() == 0:          # 배치에 없는 클래스는 건너뛴다
            continue
        err = (fg - probas[:, c]).abs()
        err_sorted, perm = torch.sort(err, 0, descending=True)
        losses.append(torch.dot(err_sorted, lovasz_grad(fg[perm])))
    return torch.stack(losses).mean()


def dice_loss(logits, target, eps=1.0):
    C = logits.shape[1]
    p = logits.softmax(1)
    t = F.one_hot(target, C).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = (p * t).sum(dims)
    denom = p.sum(dims) + t.sum(dims)
    return (1 - (2 * inter + eps) / (denom + eps)).mean()


def build_loss(name, weight, device, wd=(0.5, 0.5)):
    w = torch.tensor(weight, device=device)
    ce_plain = nn.CrossEntropyLoss()
    ce_w = nn.CrossEntropyLoss(weight=w)

    def fn(logits, target):
        if name == 'ce':
            return ce_plain(logits, target)
        if name == 'wce':
            return ce_w(logits, target)
        if name == 'wce_dice':
            return wd[0] * ce_w(logits, target) + wd[1] * dice_loss(logits, target)
        if name == 'ce_lovasz':
            return wd[0] * ce_plain(logits, target) + wd[1] * lovasz_softmax(logits, target)
        raise ValueError(f'알 수 없는 loss: {name}')
    return fn
