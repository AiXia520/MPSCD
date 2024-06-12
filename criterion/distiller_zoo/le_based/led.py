from torch import nn
import torch
from criterion.distiller_zoo.le_based.cd import CD
from criterion.distiller_zoo.le_based.id import ID
import torch.nn.functional as F
from criterion.distiller_zoo.le_based.huber_dist import HuberDist
class MultiSupConLoss(nn.Module):
    """
    Supervised Multi-Label Contrastive Learning.
    Author: Leilei Ma
    Date: Nov 4, 2022"""

    def __init__(self, temperature=0.2, contrast_mode='all',
                 base_temperature=0.2):
        super(MultiSupConLoss, self).__init__()

        self.contrast_mode = contrast_mode
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.eps = 1e-5

    def forward(self, features, labels=None, mask=None, multi=True):
        '''
        P(i, j) = \left\{ Z_{k j} \in A(i,j) \mid y_{k j} = y_{i j} = 1 \right\}

        \sum_{z_{i,j} \in  I}
        \mathcal{L}_{L L C L}^{i j}=\frac{-1}{|P(i, j)|} \sum_{z_p \in P(i, j)} \log \frac{\exp \left(z_{i j} \cdot z_p / \tau\right)}{\sum_{z_a \in A(i, j)} \exp \left(z_{i j} \cdot z_a / \tau\right)}
        '''
        # labels shape: [bs, n_view]

        device = features.device
        features = F.normalize(features, dim=-1)  # [bs, n_view, nc, dim]

        batch_size = features.shape[0]
        num_class = features.shape[2]

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # [n_view*bs, nc, dim]

        if self.contrast_mode == 'all':
            anchor_feature = contrast_feature  # [256, 80, 2432]  [n_view*bs, nc, dim]
            anchor_count = contrast_count  # [256, 80, 2432]  [n_view*bs, nc, dim]
        # intra-class and inter-class 类内和类间

        # 同类不同图片之间的对比
        anchor_dot_contrast_intra = torch.div(
            torch.matmul(anchor_feature.transpose(1, 0), contrast_feature.permute(1, 2, 0)),
            self.temperature)  # [80, 256, 256]   [nc, n_view*bs, n_view*bs]

        # 所有特征都相互计算相似度
        # contrast_feature                                                   # [n_view*bs, nc, dim]
        all_features = contrast_feature.view(-1, contrast_feature.shape[-1])  # [n_view*bs*nc, dim]
        all_anchor_dot_contrast = torch.div(
            torch.matmul(all_features, all_features.transpose(1, 0)),
            self.temperature)  # [20480, 20480]    [n_view*bs*nc, dim, n_view*bs*nc, dim]

        # 生成同类不同图片的mask
        mask_intra = labels.transpose(1, 0).unsqueeze(-1).matmul(labels.transpose(1, 0).unsqueeze(1))  # [nc, bs, bs]
        mask_intra = mask_intra.repeat(1, anchor_count, contrast_count)  # [nc, n_view*bs, n_view*bs]

        # 所有的特征的mask
        all_mask = torch.matmul(labels.contiguous().view(-1, 1), labels.contiguous().view(1, -1))
        all_mask = all_mask.repeat(anchor_count, contrast_count)  # [n_view*bs*nc, n_view*bs*nc]

        # 同类不同图片之间，去掉自身的mask
        logits_mask_intra = torch.scatter(
            torch.ones(batch_size * anchor_count, batch_size * anchor_count).to(device),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0)  # [80, 256, 256]  [n_view*bs, n_view*bs]
        logits_mask_intra = logits_mask_intra.repeat(num_class, 1, 1)
        mask_intra = mask_intra * logits_mask_intra  # [n_view*bs*nc, n_view*bs*nc]
        logits_intra = mask_intra * anchor_dot_contrast_intra  # [80, 256, 256]   [nc, n_view*bs, n_view*bs]

        # 同类不同图片之间, 为了数值稳定
        logits_max, _ = torch.max(logits_intra.permute(1, 2, 0), dim=-1, keepdim=True)  # [n_view*bs, n_view*bs, 1]
        logits_intra = (logits_intra.permute(1, 2, 0) - logits_max.detach()).permute(2, 0, 1)

        # import ipdb; ipdb.set_trace()
        # 所有的类，去掉自身
        self_mask = torch.eye(batch_size * num_class * contrast_count).to(device)
        self_mask = torch.ones_like(self_mask) - self_mask
        all_mask = all_mask * self_mask

        all_anchor_dot_contrast = all_anchor_dot_contrast * all_mask

        # 所有特征, 为了数值稳定
        # all_anchor_dot_contrast 已经被mask
        logits_max_all, _ = torch.max(all_anchor_dot_contrast, dim=-1, keepdim=True)
        logits_all = all_anchor_dot_contrast - logits_max_all.detach()

        # label mask
        labels_mask = labels.unsqueeze(1).repeat(1, contrast_count, 1).contiguous().view(-1, 1)  # [n_view*bs*nc, 1]

        # 分母 [n_view*bs*nc, n_view*bs*nc]
        logits_all = all_mask * torch.exp(logits_all)  # [n_view*bs*nc, n_view*bs*nc]
        logits_all = logits_all.sum(-1, keepdim=True)  # [n_view*bs*nc, 1]
        logits_all[logits_all <= 0] = 1
        logits_all = labels_mask * torch.log(logits_all + self.eps)  # [n_view*bs*nc, 1]

        # 分子 [n_view*bs*nc, n_view*bs]   [nc, n_view*bs, n_view*bs] => [n_view*bs*nc, n_view*bs]
        logits_intra = (logits_intra).permute(1, 0, 2).reshape(contrast_count * batch_size * num_class,
                                                               contrast_count * batch_size)
        mask_intra = (mask_intra).permute(1, 0, 2).reshape(contrast_count * batch_size * num_class,
                                                           contrast_count * batch_size)

        # 计算对数似然除以正的均值
        log_prob = logits_intra - logits_all  # [n_view*bs*nc, n_view*bs] - [n_view*bs*nc, 1] => [n_view*bs*nc, n_view*bs]

        # mask_intra 对应论文中的 P(i, j) => [n_view*bs*nc, n_view*bs]
        log_prob = (mask_intra * log_prob).sum(-1)  # [n_view*bs*nc, 1]
        mask_intra = mask_intra.sum(-1)
        mask_intra[mask_intra == 0] = 1
        mean_log_prob_pos = log_prob / mask_intra

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos  # [n_view*bs*nc]

        # 所有存在的示例计算平均，计算 （\mathcal{L}_{L L C L}^{i j}).sum(i, j) / |count(i,j)|
        labels_mask = labels_mask.view(-1)
        if labels_mask.sum() == 0.0:
            loss = (loss * labels_mask).sum()
        else:
            loss = (loss * labels_mask).sum() / labels_mask.sum()

        return loss


class Semantic2PrototypeConLoss(nn.Module):
    def __init__(self, metric='cos', reduction='mean', temperature=0.07):
        super(Semantic2PrototypeConLoss, self).__init__()
        self.temperature = temperature
        self.metric = metric
        self.reduction = reduction
        self.eps = 1e-5

    def forward(self, semantic_embed, proto_embed, target):
        """
        :param semantic_embed: [bs, n, dim]
        :param proto_embed:    [n, dim] or [1, n, dim]
        :param target:         [bs, num]
        :return:
        """
        if proto_embed.dim() != 3:
            proto_embed = proto_embed.unsqueeze(0)

        bs, num_c, dim = semantic_embed.shape
        semantic_embed = semantic_embed.permute(1, 0, 2)  # [n, bs, dim]
        proto_embed = proto_embed.repeat(bs, 1, 1).permute(1, 0, 2)  # [n, bs, dim]

        target = target.permute(1, 0)  # [n, bs]
        pos_mask = target  # [n, bs]
        neg_mask = 1 - target  # [n, bs]

        # target =  target.unsqueeze(-1)                                                      # [n, bs, 1]
        cos_sim = F.cosine_similarity(semantic_embed, proto_embed, dim=-1) / self.temperature  # [n, bs]

        # 为了数值稳定
        cos_sim_max, _ = torch.max(cos_sim, dim=-1, keepdim=True)  # [n, bs]
        cos_sim = cos_sim - cos_sim_max.detach()  # [n, bs]

        pos2proto = cos_sim * pos_mask  # 正类与原型对比  # [n, bs]
        # neg2proto = cos_sim * neg_mask                                       # 负类与原型对比  # [n, bs]
        neg2proto = cos_sim

        # 分母
        neg2proto = torch.exp(neg2proto)  # [n, bs]
        # neg2proto = neg2proto * neg_mask                                                      # [n, bs] # 保留负类与原型对比，mask掉正类

        neg2proto = neg2proto.sum(dim=-1, keepdim=True)  # [n, 1]
        neg2proto[neg2proto <= 0] = 1.0  # [n, 1]  # 优于被mask部分为0，消除求和后为0的部分
        neg2proto = torch.log(neg2proto)  # [n, 1]

        # 分子
        pos2proto = torch.exp(pos2proto)  # [n, bs]
        pos2proto = pos2proto * pos_mask  # [n, bs] # 保留负类与原型对比，mask掉正类

        pos2proto = pos2proto.sum(dim=-1, keepdim=True)  # [n, 1]
        pos2proto[pos2proto <= 0] = 1.0  # [n, 1]  # 优于被mask部分为0，消除求和后为0的部分
        pos2proto = torch.log(pos2proto)  # [n, 1]

        # 分子减去分母
        log_prob = (pos2proto - neg2proto).sum(dim=-1, keepdim=True)

        # 计算对数似然除以正的均值
        if pos_mask.sum() != 0:
            loss = - log_prob.sum() / pos_mask.sum()
        else:
            loss = - log_prob.meam()

        return loss



class LED(nn.Module):
    def __init__(self, lambda_cd=100.0, lambda_id=1000.0):
        super().__init__()
        self.cd_distiller = CD()
        self.id_distiller = ID()
        self.lambda_cd = lambda_cd
        self.lambda_id = lambda_id

        self.lambda_proto = 100
        self.sup_contrastive = MultiSupConLoss()

        self.proto_contrastive = Semantic2PrototypeConLoss()

        self.le_distill_criterion = HuberDist()

    def forward(self, le_student, le_teacher, targets, semantic_prototype = None):
        # loss_cd = self.cd_distiller(le_student, le_teacher, targets)
        # loss_id = self.id_distiller(le_student, le_teacher, targets)

        loss_supcontrastive = self.sup_contrastive(le_student.unsqueeze(1), targets)

        log_teacher = self.proto_contrastive(le_teacher, semantic_prototype, targets)

        log_student = self.proto_contrastive(le_student, semantic_prototype, targets)

        loss_prototype = F.mse_loss(log_student, log_teacher)

        # loss = self.lambda_cd * loss_cd + self.lambda_id * loss_id  + loss_supcontrastive + self.lambda_proto * loss_prototype

        loss = self.lambda_cd * loss_supcontrastive + self.lambda_id * loss_prototype

        return loss
