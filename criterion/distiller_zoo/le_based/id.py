import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

from criterion.distiller_zoo.le_based.huber_dist import HuberDist


class ID(nn.Module):  # instance-aware label-wise embedding distillation
    def __init__(self):
        super().__init__()
        self.le_distill_criterion = HuberDist()
        self.eps = 1e-8

    def _contrasive_forward(self, f_s, f_t):


        f_s = F.normalize(f_s, dim=1)
        f_t = F.normalize(f_t, dim=1)

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




    def forward(self, le_student, le_teacher, targets):
        N, C, le_length = le_student.shape
        le_mask = targets.unsqueeze(2).repeat(1, 1, le_length)
        le_student_pos = le_student * le_mask
        le_teacher_pos = le_teacher * le_mask
        n_pos_per_instance = targets.sum(dim=1)
        loss = 0.0

        for i in range(N):
            if n_pos_per_instance[i] > 1:
                le_s_i = le_student_pos[i, :, :]
                le_t_i = le_teacher_pos[i, :, :]
                le_s_pos_i = le_s_i[~(le_s_i == 0).all(1)]
                le_t_pos_i = le_t_i[~(le_t_i == 0).all(1)]
                delta_loss = self.le_distill_criterion(le_s_pos_i, le_t_pos_i)

                loss += delta_loss

        return loss


    def forward1(self, le_student, le_teacher, targets):
        N, C, le_length = le_student.shape
        le_mask = targets.unsqueeze(2).repeat(1, 1, le_length)
        # le_student_pos = le_student * le_mask
        # le_teacher_pos = le_teacher * le_mask
        n_pos_per_instance = targets.sum(dim=1)
        loss1 = 0.0
        loss2 = 0.0
        loss = 0.0
        for i in range(N):
            if n_pos_per_instance[i] > 0:
                neg_s = torch.masked_select(le_student[i, :, :], le_mask[i, :, :] == 0)
                pos_s = torch.masked_select(le_student[i, :, :], le_mask[i, :, :] == 1)
                neg_t = torch.masked_select(le_teacher[i, :, :], le_mask[i, :, :] == 0)
                pos_t = torch.masked_select(le_teacher[i, :, :], le_mask[i, :, :] == 1)

                # p_s = torch.sigmoid(pos_s)
                # p_t = torch.sigmoid(pos_t)
                # p_s = torch.clamp(p_s, min=self.eps, max=1 - self.eps)
                # p_t = torch.clamp(p_t, min=self.eps, max=1 - self.eps)
                #
                # np_s = torch.sigmoid(neg_s)
                # np_t = torch.sigmoid(neg_t)
                # np_s = torch.clamp(np_s, min=self.eps, max=1 - self.eps)
                # np_t = torch.clamp(np_t, min=self.eps, max=1 - self.eps)

                p_s = torch.softmax(pos_s, dim=0)
                p_t = torch.softmax(pos_t, dim=0)

                np_s = torch.softmax(neg_s, dim=0)
                np_t = torch.softmax(neg_t, dim=0)

                loss = nn.KLDivLoss(reduction="batchmean")(torch.log(p_s), p_t) + \
                       nn.KLDivLoss(reduction="batchmean")(torch.log(np_s), np_t)

                # loss1 += nn.KLDivLoss(reduction="batchmean")(torch.log(p_s), p_t)
                # loss2 += nn.KLDivLoss(reduction="batchmean")(torch.log(np_s), np_t)
                #
                # loss = loss1 + loss2

        return loss


