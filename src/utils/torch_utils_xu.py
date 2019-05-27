from utils.torch_utils import *

# bins = (8.0, 15.0)
bins = (4.5, 8.0, 15.0)
# bins = tuple(np.arange(4.5, 15.5, 0.5).tolist())
t_bins = torch.tensor((0,) + bins, dtype=torch.float, device=device)
weight_bins = (8.0, 15.0)
weights = (20.5, 5.4, 1.0)


def quantize_distance_matrix(dmat, bins=bins, use_ordinal=False):
    b, m, n = dmat.size()
    assert m == n
    qdmat = torch.zeros((b, len(bins) + int(not use_ordinal), m, n), dtype=torch.float, device=device)
    qdmat[:, 0, :, :] = (dmat < bins[0])
    for i in range(1, len(bins)):
        if not use_ordinal:
            qdmat[:, i, :, :] = (dmat >= bins[i - 1]) * (dmat < bins[i])
        else:
            qdmat[:, i, :, :] = (dmat < bins[i])
    if not use_ordinal:
        qdmat[:, len(bins), :, :] = (dmat >= bins[-1])
    return qdmat.float()


def digitize_distance_matrix(dmat, bins=bins):
    ddmat = torch.zeros(dmat.size(), dtype=torch.float, device=device)
    for i in range(1, len(bins)):
        ddmat[(dmat >= bins[i - 1]) * (dmat < bins[i])] = i
    ddmat[(dmat >= bins[-1])] = len(bins)
    return ddmat.float()


class OuterProduct(nn.Module):

    def __init__(self, include_mid=True):
        super(OuterProduct, self).__init__()
        self.include_middle = include_mid

    def forward(self, seq1, seq2):
        m = seq1.size(2)
        v_i = (seq1.transpose(1, 2).unsqueeze(2).repeat(1, 1, m, 1))
        v_j = v_i.transpose(1, 2)
        if self.include_middle:
            v = torch.cat([v_i, v_j, (v_i + v_j).div(2)], 3)
        else:
            v = torch.cat([v_i, v_j], 3)
        return v.permute(0, 3, 1, 2)


def mask_distance_matrix(dmat, weight_bins=weight_bins):
    """
    Answer: yep, a larger weight is assigned to a pair of residues forming a contact.
    I assigned 20.5, 5.4, 1 to the distance 0-8, 8-15, and >15, respectively, for residue pairs (i, j) where |i-j| >=24.
    These numbers were derived from simple statistics of an old training set.
    However, you don't have to be very accurate here.
    When |i-j| is small, you can reduce 20.5 and 5.4 to smaller values.
    :param dmat: A distance matrix
    :param bins: The quantized distance matrix
    :return: The quantized distance matrix
    """
    b, m, n = dmat.size()
    imj = b * [[[abs(i-j) >= 24 for j in range(n)] for i in range(m)]]
    t_imj = torch.tensor(imj, dtype=torch.float, device=device)
    masks = quantize_distance_matrix(dmat, weight_bins, False)
    return masks, t_imj


def compute_weighted_loss_w_masks(losses, dmat, weights=weights):
    b, m, n = dmat.size()
    assert m == n
    masks, imj = mask_distance_matrix(dmat)
    masks, imj = masks.contiguous().view(-1, b * m * n), imj.view(b * m * n)
    w_losses = torch.zeros(losses.size(), dtype=torch.float, device=device)
    for i, w in enumerate(weights):
        msk = masks[i]
        w_losses.add_(losses * msk * imj * w)
        w_losses.add_(losses * msk * (1.0 - imj) * max(1.0, w / 2.0))
    return w_losses


def get_cross_entropy_loss(cmap_hat, dmat):
    b, c, m, n = cmap_hat.size()
    assert m == n
    cmap = quantize_distance_matrix(dmat, use_ordinal=False)
    v_cmap = cmap.view(c, b*m*n).transpose(0, 1).unsqueeze(1)
    v_cmap_hat = cmap_hat.view(c, b*m*n).transpose(0, 1).unsqueeze(2)
    ce = -v_cmap.bmm(F.log_softmax(v_cmap_hat, 1)).view(-1)
    ce = compute_weighted_loss_w_masks(ce, dmat)
    return ce.mean()


def ce_cmap_to_dmat(cmap):
    b, _, m, n = cmap.size()
    dmat = torch.gather(t_bins, 0, cmap.argmax(1).view(-1)).view(b, m, n)
    return dmat


def ordinal_cmap_to_dmat(cmap):
    b, m, n, _, _ = cmap.size()
    dmat = torch.gather(t_bins, 0, cmap.argmax(4).sum(3).view(-1)).view(b, m, n)
    return dmat


def get_ordinal_log_loss(cmap_hat, dmat):
    b, m, n, c, _ = cmap_hat.size()
    assert m == n
    cmap = quantize_distance_matrix(dmat, use_ordinal=True)
    v_cmap = cmap.view(c, b * m * n).transpose(0, 1).unsqueeze(1)
    v_cmap = torch.cat([v_cmap, 1.0 - v_cmap], 1).transpose(1, 2).contiguous()
    v_cmap_hat = torch.log_softmax(cmap_hat.view(b*m*n, c, -1), 2).contiguous()
    ce = -v_cmap.view(-1, 2).unsqueeze(1).bmm(v_cmap_hat.view(-1, 2).unsqueeze(2)).view(b*m*n, c).sum(1)
    ce = compute_weighted_loss_w_masks(ce, dmat)
    return ce.mean()
