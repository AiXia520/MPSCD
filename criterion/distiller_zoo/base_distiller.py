from torch import nn

from criterion.distiller_zoo import ft_based, le_based, logits_based
from criterion.distiller_zoo.none import NONE


class BaseDistiller(nn.Module):
    def __init__(self, lambda_ft=1.0, ft_dis=None, lambda_le=1.0, le_dis=None, lambda_logits=1.0, logits_dis=None):
        super().__init__()

        self.lambda_ft = lambda_ft

        self.ft_based_distiller = ft_based.__dict__[ft_dis["name"]](**ft_dis["para"])

        self.lambda_le = lambda_le
        self.le_based_distiller = le_based.__dict__[le_dis["name"]](**le_dis["para"])

        self.lambda_logits = lambda_logits
        if lambda_logits == 0 or logits_dis is None:
            self.logits_based_distiller = NONE()
        else:
            self.logits_based_distiller = logits_based.__dict__[logits_dis["name"]](**logits_dis["para"])

    def forward(self, f_s, f_t, le_s, le_t, logits_s, logits_t, targets, semantic_prototype=None):
        loss_ft = self.ft_based_distiller(f_s, f_t, targets, logits_student=logits_s, logits_teacher=logits_t)
        loss_le = self.le_based_distiller(le_s, le_t, targets, semantic_prototype=semantic_prototype)
        loss_logits = self.logits_based_distiller(logits_s, logits_t, targets)
        loss = self.lambda_ft * loss_ft + self.lambda_le * loss_le + self.lambda_logits * loss_logits

        return loss
