import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn import Parameter
import os
import math
import numpy as np
import torch.nn.functional as F

def gen_A(num_classes, t, adj_file):
    import pickle
    result = pickle.load(open(adj_file, 'rb'))
    _adj = result['adj']  #(20, 20)
    _nums = result['nums']  #(20)
    _nums = _nums[:, np.newaxis]  #(20,1)
    _adj = _adj / _nums
    _adj[_adj < t] = 0
    _adj[_adj >= t] = 1
    #上述是等式7
    # p = 0.2
    _adj = _adj * 0.25 / (_adj.sum(0, keepdims=True) + 1e-6)
    _adj = _adj + np.diag(np.ones(num_classes) - 0.0)
    #上述是等式8
    return _adj


def gen_adj(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj

class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-5, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                with torch.no_grad():
                    if self.disable_torch_grad_focal_loss:
                        torch.set_grad_enabled(False)
                    self.xs_pos = self.xs_pos * self.targets
                    self.xs_neg = self.xs_neg * self.anti_targets
                    self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                                self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
                    if self.disable_torch_grad_focal_loss:
                        torch.set_grad_enabled(True)
                self.loss *= self.asymmetric_w
            else:
                self.xs_pos = self.xs_pos * self.targets
                self.xs_neg = self.xs_neg * self.anti_targets
                self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                            self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
                self.loss *= self.asymmetric_w
        _loss = - self.loss.sum() / x.size(0)
        _loss = _loss / y.size(1) * 1000

        return _loss


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.relu = nn.ReLU()

        # self.init_weights()

    def init_weights(self, init_linear='normal'):
        init_weights(self, init_linear)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class Learner(nn.Module):
    def __init__(self, model, criterion, optimizer, scheduler):
        super(Learner, self).__init__()
        self.model = model
        self.scaler = GradScaler()
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epoch = 0

    def forward(self, x):
        return self.model.forward(x)

    def forward_with_criterion(self, inputs, targets):
        with autocast():
            out = self.forward(inputs).float()
        return self.criterion(out, targets), out

    def learn(self, inputs, targets):
        loss, out = self.forward_with_criterion(inputs, targets)
        self.model.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        return loss, out


class Learner_KD(nn.Module):
    def __init__(self, cfg, model_t, model_s, criterion_s, criterion_t2s, optimizer, scheduler):
        super(Learner_KD, self).__init__()
        self.cfg = cfg
        self.model_t = model_t
        self.model_s = model_s
        self.scaler = GradScaler()
        self.criterion_s = criterion_s
        self.criterion_t2s = criterion_t2s
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epoch = 0

        if self.cfg.dataset =="voc":
            adj_file = os.path.join('data', 'voc', 'voc_adj.pkl')
        elif self.cfg.dataset =="coco":
            adj_file = os.path.join('data', 'coco', 'coco_adj.pkl')
        elif self.cfg.dataset == "nuswide":
            adj_file = os.path.join('data', 'nuswide', 'nuswide_adj.pkl')

        self.in_channel = 300
        self.t = 0.4
        # GCN module
        self.gc1 = GraphConvolution(self.in_channel, 512)
        self.gc2 = GraphConvolution(512, 768)

        self.relu = nn.LeakyReLU(0.2)
        _adj = gen_A(self.cfg.num_classes, self.t, adj_file)
        self.A = Parameter(torch.from_numpy(_adj).float())

        if self.cfg.model_t == "resnet101" and self.cfg.model_s == "resnet34":
            self.t_emb = nn.Linear(2048, 768).cuda()
            self.s_emb = nn.Linear(512, 768).cuda()

        if self.cfg.model_t == "resnet101" and self.cfg.model_s == "resnet50":
            self.t_emb = nn.Linear(2048, 768).cuda()
            self.s_emb = nn.Linear(2048, 768).cuda()

        if self.cfg.model_t == "resnet34" and self.cfg.model_s == "resnet101":
            self.t_emb = nn.Linear(512, 768).cuda()
            self.s_emb = nn.Linear(2048, 768).cuda()

        if self.cfg.model_t == "resnet50" and self.cfg.model_s == "resnet18":
            self.t_emb = nn.Linear(2048, 768).cuda()
            self.s_emb = nn.Linear(512, 768).cuda()

        if self.cfg.model_t == "repvgg_a2" and self.cfg.model_s == "repvgg_a0":
            self.t_emb = nn.Linear(1408, 768).cuda()
            self.s_emb = nn.Linear(1280, 768).cuda()

        if self.cfg.model_t == "wrn101" and self.cfg.model_s == "wrn50":
            self.t_emb = nn.Linear(2048, 768).cuda()
            self.s_emb = nn.Linear(2048, 768).cuda()

        if self.cfg.model_t == "swin_s" and self.cfg.model_s == "swin_t":
            self.t_emb = nn.Linear(768, 768).cuda()
            self.s_emb = nn.Linear(768, 768).cuda()

        if self.cfg.model_t == "resnet101" and self.cfg.model_s == "repvgg_a0":
            self.t_emb = nn.Linear(2048, 768).cuda()
            self.s_emb = nn.Linear(1280, 768).cuda()

        if self.cfg.model_t == "resnet101" and self.cfg.model_s == "mobilenet_v2":
            self.t_emb = nn.Linear(2048, 768).cuda()
            self.s_emb = nn.Linear(1280, 768).cuda()

        if self.cfg.model_t == "swin_t" and self.cfg.model_s == "mobilenet_v2":
            self.t_emb = nn.Linear(768, 768).cuda()
            self.s_emb = nn.Linear(1280, 768).cuda()

        if self.cfg.model_t == "swin_t" and self.cfg.model_s == "resnet34":
            self.t_emb = nn.Linear(768, 768).cuda()
            self.s_emb = nn.Linear(512, 768).cuda()

        if self.cfg.model_t == "resnet50" and self.cfg.model_s == "repvgg_a0":
            self.t_emb = nn.Linear(2048, 768).cuda()
            self.s_emb = nn.Linear(1280, 768).cuda()

        if self.cfg.model_t == "resnet50" and self.cfg.model_s == "mobilenet_v2":
            self.t_emb = nn.Linear(2048, 768).cuda()
            self.s_emb = nn.Linear(1280, 768).cuda()

        if self.cfg.model_t == "swin_t" and self.cfg.model_s == "resnet18":
            self.t_emb = nn.Linear(768, 768).cuda()
            self.s_emb = nn.Linear(512, 768).cuda()


        # self.gc3 = GraphConvolution(self.in_channel, 512)
        # self.gc4 = GraphConvolution(512, 1024)

        self.proj_head = MLP(input_dim=2048, hidden_dim=768, output_dim=768, num_layers=2)
        self.select_prototype_path = self.select_Prototype()
        self.prototype = self.load_Prototype()

        # self.asyloss = AsymmetricLossOptimized()
    def select_Prototype(self):
        if self.cfg.dataset == 'coco':
            self.prototype_path = os.path.join('data', 'coco', 'vis_prototype_embed_file_select_resnet101_448_coco_085.npy')
        elif self.cfg.dataset == 'nuswide':
            self.prototype_path = os.path.join('data', 'nuswide', 'vis_prototype_embed_file_select_resnet101_448_nuswide_085.npy')
        elif self.cfg.dataset == 'vg500':
            self.prototype_path = os.path.join('data', 'vg500', 'vis_prototype_embed_file_select_resnet101_576_vg500_085.npy')
        elif self.cfg.dataset == 'voc':
            self.prototype_path = os.path.join('data', 'voc', 'vis_prototype_embed_file_select_resnet101_448_voc_085.npy')
        elif self.cfg.dataset == 'voc2012':
            self.prototype_path = os.path.join('data', 'voc', 'vis_prototype_embed_file_select_resnet101_448_voc_085.npy')
        else:
            raise NotImplementedError
        return self.prototype_path

    def load_Prototype(self):
        prototype = torch.from_numpy(np.load(self.select_prototype_path)).unsqueeze(0)
        prototype = nn.Parameter(prototype, requires_grad=True)
        # prototype = nn.Parameter(prototype, requires_grad=False)
        return prototype


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

    def _off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def logits_to_signal(self, logits):
        probs = torch.softmax(logits, dim=-1)
        confidence = torch.max(probs, dim=-1)[0]
        return confidence.sum()  # gradients of different samples can backprop simutaneously

    def diff_loss(self, f_s, f_t):
        def _at(feat, p):
            return F.normalize(feat.pow(p).mean(1).reshape(feat.size(0), -1))
        s_H, t_H = f_s.shape[2], f_t.shape[2]
        if s_H > t_H:
            f_s = F.adaptive_avg_pool2d(f_s, (t_H, t_H))
        elif s_H < t_H:
            f_t = F.adaptive_avg_pool2d(f_t, (s_H, s_H))

        return (_at(f_s, 2) - _at(f_t, 2)).pow(2).mean()

    def forward_with_criterion(self, inputs, targets, inp = None):

        if inp is not None:

            inp = inp[0]  # (c, k)
            adj = gen_adj(self.A).detach()
            x = self.gc1(inp, adj)
            x = self.relu(x)
            semantic_prototype = self.gc2(x, adj)  # (c, d')
            semantic_prototype = semantic_prototype.cuda()


        with autocast():
            f_s, le_s, logits_s, feats = self.model_s.forward(inputs, le=True, ft=True)
            with torch.no_grad():
                f_t, le_t, logits_t, feats_t = self.model_t.forward(inputs, le=True, ft=True)


        f_t, le_t, logits_t = f_t.float().detach(), le_t.float().detach(), logits_t.float().detach()
        f_s, le_s, logits_s = f_s.float(), le_s.float(), logits_s.float()
        targets = targets.to(torch.float)

       

        feature_prototype = self.proj_head(self.prototype).cuda()
        feature_prototype = feature_prototype.squeeze(0)

        weight = le_s.mean(dim=0) - feature_prototype
        weight = F.softmax(- torch.norm(weight, 2, dim=1))
        feature_prototype = weight.unsqueeze(1) * feature_prototype

        semantic_prototype = feature_prototype * semantic_prototype

        loss_hard = self.criterion_s(logits_s, targets)
        loss_hard_t = self.criterion_s(logits_t, targets)


        loss_soft = self.criterion_t2s(torch.mm(self.s_emb(f_s),semantic_prototype.transpose(0,1)),
                                           torch.mm(self.t_emb(f_t),semantic_prototype.transpose(0,1)), le_s, le_t, logits_s, logits_t, targets,
                                           semantic_prototype=semantic_prototype)

        loss = loss_hard + self.cfg.gamma * loss_soft

        return loss, logits_s

    def learn(self, inputs, targets, inp = None):
        loss, out = self.forward_with_criterion(inputs, targets, inp = inp)
        self.model_s.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        return loss, out

    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            module.train(mode)
        self.model_t.eval()
        return self





