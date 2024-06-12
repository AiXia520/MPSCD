import torch
import torch.nn as nn
import torch.nn.functional as F

def single_stage_at_loss(f_s, f_t, p):
    def _at(feat, p):
        return F.normalize(feat.pow(p).mean(1).reshape(feat.size(0), -1))

    s_H, t_H = f_s.shape[2], f_t.shape[2]
    if s_H > t_H:
        f_s = F.adaptive_avg_pool2d(f_s, (t_H, t_H))
    elif s_H < t_H:
        f_t = F.adaptive_avg_pool2d(f_t, (s_H, s_H))
    return (_at(f_s, p) - _at(f_t, p)).pow(2).mean()


def at_loss(g_s, g_t, p):
    return sum([single_stage_at_loss(f_s, f_t, p) for f_s, f_t in zip(g_s, g_t)])



class AT(nn.Module):
    """
    Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer
    src code: https://github.com/szagoruyko/attention-transfer
    """

    def __init__(self, p=2):
        super(AT, self).__init__()
        self.p = p

    def forward(self, f_s, f_t, targets, logits_student=None, logits_teacher=None):
        return at_loss(f_s, f_t, self.p)


