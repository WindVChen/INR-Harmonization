import torch


def loss_generator(ignore: list = None):
    loss_fn = {'mse': mse,
               'lut_mse': lut_mse,
               'masked_mse': masked_mse,
               'sample_weighted_mse': sample_weighted_mse,
               'regularize_LUT': regularize_LUT,
               'MaskWeightedMSE': MaskWeightedMSE}

    if ignore:
        for fn in ignore:
            ignore.pop(fn)

    return loss_fn


def mse(pred, gt):
    return torch.mean((pred - gt) ** 2)


def masked_mse(pred, gt, mask):
    delimin = torch.clamp_min(torch.sum(mask, dim=([x for x in range(1, len(mask.shape))])), 100).cuda()
    # total = torch.sum(torch.ones_like(mask), dim=([x for x in range(1, len(mask.shape))]))
    out = torch.sum((mask > 100 / 255.) * (pred - gt) ** 2, dim=([x for x in range(1, len(mask.shape))]))
    out = out / delimin
    return torch.mean(out)


def sample_weighted_mse(pred, gt, mask):
    multi_factor = torch.clamp_min(torch.sum(mask, dim=([x for x in range(1, len(mask.shape))])), 100).cuda()
    multi_factor = multi_factor / (multi_factor.sum())
    # total = torch.sum(torch.ones_like(mask), dim=([x for x in range(1, len(mask.shape))]))
    out = torch.mean((pred - gt) ** 2, dim=([x for x in range(1, len(mask.shape))]))
    out = out * multi_factor
    return torch.sum(out)


def regularize_LUT(lut):
    st = lut[lut < 0.]
    reg_st = (st ** 2).mean() if min(st.shape) != 0 else 0

    lt = lut[lut > 1.]
    reg_lt = ((lt - 1.) ** 2).mean() if min(lt.shape) != 0 else 0

    return reg_lt + reg_st


def lut_mse(feat, lut_batch):
    loss = 0
    for id in range(feat.shape[0] // lut_batch):
        for i in feat[id * lut_batch: id * lut_batch + lut_batch]:
            for j in feat[id * lut_batch: id * lut_batch + lut_batch]:
                loss += mse(i, j)

    return loss / lut_batch


def MaskWeightedMSE(pred, label, mask):
    label = label.view(pred.size())
    reduce_dims = get_dims_with_exclusion(label.dim(), 0)

    loss = (pred - label) ** 2
    delimeter = pred.size(1) * torch.clamp_min(torch.sum(mask, dim=reduce_dims), 100)
    loss = torch.sum(loss, dim=reduce_dims) / delimeter

    return torch.mean(loss)


def get_dims_with_exclusion(dim, exclude=None):
    dims = list(range(dim))
    if exclude is not None:
        dims.remove(exclude)

    return dims