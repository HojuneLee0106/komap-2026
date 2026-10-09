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


def boundary_mask(target):
    """3x3 이웃에 다른 클래스가 있는 픽셀(= 1px 경계띠). GT 90장에서 전체의 31.1%."""
    t = target[:, None].float()
    mx = F.max_pool2d(t, 3, 1, 1)
    mn = -F.max_pool2d(-t, 3, 1, 1)
    return ((mx != t) | (mn != t))[:, 0]


def weighted_ce(logits, target, w, bmask, bw):
    """경계띠 픽셀에 (1+bw)배 가중치를 준 교차엔트로피.

    근거: GT 90장 전수 측정에서 1px 경계띠는 전체 픽셀의 31.1%인데,
    경계띠 정확도와 mIoU가 67%->0.777 / 80%->0.855 / 88%->0.900 으로
    거의 선형이다. 현재 모델의 경계띠 정확도가 66.6%이고 실측 mIoU가
    0.78~0.80이므로 남은 오차는 사실상 전부 이 띠에 있다. 평범한 CE는
    이 띠를 전체의 31%만큼만 신경 쓰므로 명시적으로 끌어올린다.
    """
    ls = F.cross_entropy(logits, target, weight=w, reduction='none')
    pw = 1.0 + bw * bmask.float()
    return (ls * pw).sum() / pw.sum()


def build_loss(name, weight, device, wd=(0.5, 0.5), bw=0.0):
    w = torch.tensor(weight, device=device)
    ce_plain = nn.CrossEntropyLoss()
    ce_w = nn.CrossEntropyLoss(weight=w)

    def fn(logits, target):
        bm = boundary_mask(target) if bw > 0 else None
        ce_wf = (lambda l, t: weighted_ce(l, t, w, bm, bw)) if bw > 0 else ce_w
        ce_pf = (lambda l, t: weighted_ce(l, t, None, bm, bw)) if bw > 0 else ce_plain
        if name == 'ce':
            return ce_pf(logits, target)
        if name == 'wce':
            return ce_wf(logits, target)
        if name == 'wce_dice':
            return wd[0] * ce_wf(logits, target) + wd[1] * dice_loss(logits, target)
        if name == 'ce_lovasz':
            return wd[0] * ce_pf(logits, target) + wd[1] * lovasz_softmax(logits, target)
        raise ValueError(f'알 수 없는 loss: {name}')
    return fn
