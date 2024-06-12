import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RkdAngle(nn.Module):
    @staticmethod
    def forward(student, teacher):
        with torch.no_grad():
            td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss = F.smooth_l1_loss(s_angle, t_angle, reduction='mean')
        return loss


def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res


class RkdDistance(nn.Module):
    @staticmethod
    def forward(student, teacher):
        N, C = student.shape

        with torch.no_grad():
            t_d = pdist(teacher, squared=False)
            mean_td = t_d[t_d > 0].mean()
            t_d = t_d / mean_td

        d = pdist(student, squared=False)
        mean_d = d[d > 0].mean()
        d = d / mean_d

        loss = F.smooth_l1_loss(d, t_d, reduction="sum") / N
        return loss


class RKD(nn.Module):
    def __init__(self, x=10, y=10):
        super().__init__()
        self.alpha_1 = x
        self.alpha_2 = y
        self.rkd_dist = RkdDistance()
        self.rkd_angle = RkdAngle()
        self.eps = 1e-8

    def _calculate_isd_sim(self, features):
        sim_q = torch.mm(features, features.T)
        logits_mask = torch.scatter(
            torch.ones_like(sim_q),
            1,
            torch.arange(sim_q.size(0)).view(-1, 1).cuda(),
            0
        )
        row_size = sim_q.size(0)
        sim_q = sim_q[logits_mask.bool()].view(row_size, -1)
        return sim_q / 0.02

    def forward(self, f_s, f_t, targets, logits_student=None, logits_teacher=None):
        # f_s, f_t = f_s[-1], f_t[-1]

        f_s = F.normalize(f_s, dim=1)
        f_t = F.normalize(f_t, dim=1)

        # rkd_dist_loss = self.rkd_dist(f_s, f_t)
        # rkd_angle_loss = self.rkd_angle(f_s, f_t)
        # loss1 = rkd_dist_loss * self.alpha_1 + rkd_angle_loss * self.alpha_2

        f = torch.matmul(f_t, f_s.t())
        alpha = 1.0
        n = f.size(0)
        first_term = f.diag().mean()
        aij = torch.ones(n, n).cuda() * \
              np.log((n - alpha) / (n * n * (n - 1)))
        aii = torch.log(torch.tensor((alpha / (n * n),))).cuda()
        mij = (torch.ones(n, n) - torch.ones(n).diag()).cuda()
        mii = torch.ones(n).diag().cuda()
        weighted_f = f + mii * aii + mij * aij
        second_term = weighted_f.logsumexp(dim=1).logsumexp(dim=0)
        loss = first_term - second_term
        return loss
