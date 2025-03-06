# Obtained from:
#  https://github.com/BIT-DA/SePiCo
# https://github.com/NazaninMn/GenGMM

# ---------------------------------------------------------------
# Copyright (c) 2022 BIT-DA. All rights reserved.
# Licensed under the Apache License, Version 2.0
# -------------------------------------------------

import torch
import torch.nn as nn  # TODO: Modifications
import torch.utils.data
import torch.distributed
import torch.backends.cudnn
from collections import deque
from core.models.GMMSeg import distributed_sinkhorn_wograd, shifted_var, init_weights, l2_normalize, momentum_update, rnd_sample  # TODO: Modifications

from timm.models.layers import trunc_normal_  # TODO: Modifications
from einops import rearrange, repeat # TODO: Modifications
from core.models.distributions import MultivariateNormalDiag  # TODO: Modifications


class ProtoEstimator:
    def __init__(self, dim, class_num, memory_length=100, resume=""):
        super(ProtoEstimator, self).__init__()
        self.dim = dim
        self.class_num = class_num
        self.embedding_dim=dim # TODO: Modifications
        self.num_components = 15 # TODO: Modifications
        self.feat_norm = nn.LayerNorm(self.embedding_dim).cuda()   # TODO: Modifications
        self.mask_norm = nn.LayerNorm(self.class_num).cuda()   # TODO: Modifications
        self.means = nn.Parameter(torch.zeros(self.class_num, self.num_components, self.embedding_dim),
                                  requires_grad=False) # TODO: Modifications
        trunc_normal_(self.means, std=0.02)  # TODO: Modifications
        self.num_prob_n = self.num_components  # TODO: Modifications
        self.diagonal = nn.Parameter(torch.ones(self.class_num, self.num_components, self.embedding_dim),
                                     requires_grad=False) # Modifications
        self.eye_matrix = nn.Parameter(torch.ones(self.embedding_dim), requires_grad=False) # Modifications
        self.factors = [1, 1, 1]    # TODO: Modifications it was [2, 1, 1]
        self.update_GMM_interval = 1  # TODO: Modifications
        self.max_sample_size = 20  # TODO: Modifications
        self.K = 50 # TODO: Modifications
        self.queue = torch.randn(self.class_num * self.num_components, self.embedding_dim, self.K) # TODO: Modifications
        self.queue = nn.functional.normalize(self.queue, dim=-2) # TODO: Modifications
        self.queue_ptr = torch.zeros(self.class_num * self.num_components, dtype=torch.long) # TODO: Modifications
        
        self.K_target = 160  # TODO: Modifications
        self.queue_target = torch.randn(self.class_num , self.embedding_dim,
                                 self.K_target)  # TODO: Modifications
        self.queue_target = nn.functional.normalize(self.queue_target, dim=-2)  # TODO: Modifications
        self.queue_ptr_target = torch.zeros(self.class_num, dtype=torch.long)  # TODO: Modifications
        self.Ks_target = torch.tensor([self.K_target for _c in range(self.class_num)], dtype=torch.long)  # TODO: Modifications
        self.means_target = nn.Parameter(torch.zeros(self.class_num, self.embedding_dim),
                                  requires_grad=False)  # TODO: Modifications


        self.Ks = torch.tensor([self.K for _c in range(self.class_num * self.num_components)], dtype=torch.long) # TODO: Modifications
        self.gamma_mean = 0.999 # TODO: Modifications
        self.gamma_cov = 0 # TODO: Modifications

    # TODO: Modifications, update GMM's componnents per category
    def update_proto(self, features, labels, epoch):
        """Update variance and mean

        Args:
            features (Tensor): feature map, shape [B, A, H, W]  N = B*H*W
            labels (Tensor): shape [B, 1, H, W]
        """
        with torch.no_grad():# TODO: Modifications
#             features = self.feat_norm(features.detach())  # * n, d  # TODO: Modifications
            labels = labels.reshape(-1)
            features = l2_normalize(features)  # TODO: Modifications
            self.means.data.copy_(l2_normalize(self.means))  # TODO: Modifications
            _log_prob = self.compute_log_prob(features.cpu()) # TODO: Modifications
            final_probs = _log_prob.contiguous().view(-1, self.class_num, self.num_prob_n) # TODO: Modifications
            _m_prob = torch.amax(final_probs.cuda(), dim=-1) # TODO: Modifications
            out_seg = self.mask_norm(_m_prob) # TODO: Modifications
            # out_seg = rearrange(out_seg, "(b h w) k -> b k h w", b=base_feature.shape[0], h=base_feature.shape[2]) # TODO: Modifications

            contrast_logits, contrast_target, qs = self.online_contrast(labels, final_probs.cuda(), features, out_seg)  # TODO: Modifications

            _c_mem = features  # TODO: Modifications
            _gt_seg_mem = labels  # TODO: Modifications
            _qs = qs   # TODO: Modifications

            unique_c_list = _gt_seg_mem.unique().int() # TODO: Modifications
            for k in unique_c_list: # TODO: Modifications
                if k == 255: continue # TODO: Modifications
                self._dequeue_and_enqueue_k(k.item(), _c_mem, _qs.bool(), (_gt_seg_mem == k.item()))  # TODO: Modifications

            # * EM
            if epoch % self.update_GMM_interval == 0:  # TODO: Modifications
                self.update_GMM(unique_c_list)   # TODO: Modifications

        return out_seg, contrast_logits, contrast_target




    @torch.no_grad()
    def update_target_bank(self,features, labels, num = 10):  # TODO: Modifications
        features = self.feat_norm(features)  # * n, d  # TODO: Modifications
        features = l2_normalize(features)   # TODO: Modifications
        unique_c_list = labels.unique().int()  # TODO: Modifications
        for k in unique_c_list:  ## TODO: Modifications
            if k == 255: continue  # TODO: Modifications
            features_selected = features[labels == k] # TODO: Modifications
            # labels = labels[labels == k]# TODO: Modifications
            mean_f = features_selected.mean(0)# TODO: Modifications
            mean_f = l2_normalize(mean_f.unsqueeze(0)).T# TODO: Modifications
            similarity = features_selected.mm(mean_f) #  dot product# TODO: Modifications
            if similarity.shape[0]<num: continue
            values, indices = torch.topk(similarity, k=num, dim=0)
            selected = features_selected[indices.squeeze(1)]

            ptr = int(self.queue_ptr_target[k])# TODO: Modifications
            if ptr + num >= self.Ks_target[k]:# TODO: Modifications
                _fir = self.Ks_target[k] - ptr# TODO: Modifications
                _sec = num - self.Ks_target[k] + ptr# TODO: Modifications
                self.queue_target[k, :, ptr:self.Ks_target[k]] = selected[:_fir].T# TODO: Modifications
                self.queue_target[k, :, :_sec] = selected[_fir:].T# TODO: Modifications
            else:# TODO: Modifications
                self.queue_target[k, :, ptr:ptr + num] = selected.T# TODO: Modifications
            ptr = (ptr + num) % self.Ks_target[k].item()  # move pointer # TODO: Modifications
            self.queue_ptr_target[k] = ptr# TODO: Modifications


        
    @torch.no_grad()
    def _dequeue_and_enqueue_k(self, _c, _c_embs, _c_cluster_q, _c_mask):   # TODO: Modifications

        if _c_mask is None: _c_mask = torch.ones(_c_embs.shape[0]).detach_()

        _k_max_sample_size = self.max_sample_size
        _embs = _c_embs[_c_mask > 0]   # embedding that their ground truth labels are k (_c_mask > 0)
        _cluster = _c_cluster_q[_c_mask > 0]  # _c_cluster_q already contains the correct qu's and know it considers those that are correct and their labels are equal to k
        for q_index in range(self.num_components):
            _q_ptr = _c * self.num_components + q_index  # it is used to convert to 1 to n*componnent labels
            ptr = int(self.queue_ptr[_q_ptr])

            if torch.sum(_cluster[:, q_index]) == 0: continue
            assert _cluster[:, q_index].shape[0] == _embs.shape[0]
            _q_embs = _embs[_cluster[:, q_index]]  # find embedding which are correct

            _q_sample_size = _q_embs.shape[0]
            assert _q_sample_size == torch.sum(_cluster[:, q_index])

            if self.max_sample_size != -1 and _q_sample_size > _k_max_sample_size:
                _rnd_sample = rnd_sample(_q_sample_size, _k_max_sample_size, _uniform=True, _device=_c_embs.device)
                _q_embs = _q_embs[_rnd_sample, ...]
                _q_sample_size = _k_max_sample_size

            # replace the embs at ptr (dequeue and enqueue)
            if ptr + _q_sample_size >= self.Ks[_q_ptr]:
                _fir = self.Ks[_q_ptr] - ptr
                _sec = _q_sample_size - self.Ks[_q_ptr] + ptr
                self.queue[_q_ptr, :, ptr:self.Ks[_q_ptr]] = _q_embs[:_fir].T
                self.queue[_q_ptr, :, :_sec] = _q_embs[_fir:].T
            else:
                self.queue[_q_ptr, :, ptr:ptr + _q_sample_size] = _q_embs.T

            ptr = (ptr + _q_sample_size) % self.Ks[_q_ptr].item()  # move pointer
            self.queue_ptr[_q_ptr] = ptr

    @torch.no_grad()
    def update_GMM(self, unique_c_list):   # TODO: Modifications
        components = self.means.data.clone()
        covs = self.diagonal.data.clone()

        for _c in unique_c_list:
            if _c == 255: continue
            _c = _c if isinstance(_c, int) else _c.item()

            for _p in range(self.num_components):
                _p_ptr = _c * self.num_components + _p
                _mem_fea_q = self.queue[_p_ptr, :, :self.Ks[_c]].transpose(-1, -2)  # n,d

                f = l2_normalize(torch.sum(_mem_fea_q, dim=0))  # d,

                new_value = momentum_update(old_value=components[_c, _p, ...], new_value=f, momentum=self.gamma_mean,
                                            debug=False)
                components[_c, _p, ...] = new_value

                _shift_fea = _mem_fea_q - f[None, ...]  # * n, d

                _cov = shifted_var(_shift_fea, rowvar=False)
                _cov = _cov + 1e-2 * self.eye_matrix
                _cov = _cov.sqrt()

                new_covariance = momentum_update(old_value=covs[_c, _p, ...], new_value=_cov, momentum=self.gamma_cov,
                                                 debug=False)
                covs[_c, _p, ...] = new_covariance

        self.means = nn.Parameter(components, requires_grad=False)
        self.diagonal = nn.Parameter(covs, requires_grad=False)
        # * NOTE: need not to sync across gpus. memory is shared across all gpus


    def online_contrast(self, gt_seg, simi_logits, _c, out_seg): # TODO: Modifications
        # find pixels that are correctly classified
        pred_seg = torch.max(out_seg, 1)[1]
        mask = (gt_seg == pred_seg.view(-1))  #find the correct predictions

        # compute logits
        contrast_logits = simi_logits.flatten(1) # * n, c*p
        contrast_target = gt_seg.clone().float()

        return_qs = torch.zeros(size=(simi_logits.shape[0], self.num_components), device=gt_seg.device)
        # clustering for each class
        for k in gt_seg.unique().long():
            if k == 255: continue
            # get initial assignments for the k-th class
            init_q = simi_logits[:, k, :] #[n in patch, n class, components]=>[n in patch, components]
            init_q = init_q[gt_seg == k, ...] # n,p
            init_q = init_q[:,:self.num_components]
            init_q = init_q / torch.abs(init_q).max()

            # * init_q: [gt_n, p]
            # clustering q.shape = n x self.num_components
            q, indexs = distributed_sinkhorn_wograd(init_q)
            try:
                assert torch.isnan(q).int().sum() <= 0
            except:
                # * process nan
                q[torch.isnan(q)] = 0
                indexs[torch.isnan(q).int().sum(dim=1)>0] = 255 - (self.num_prob_n * k)

            # binary mask for pixels of the k-th class
            m_k = mask[gt_seg == k]

            m_k_tile = repeat(m_k, 'n -> n tile', tile=self.num_components)
            # mask the incorrect q with zero
            q = q * m_k_tile  # n x self.num_prob_n

            contrast_target[gt_seg == k] = indexs.float() + (self.num_prob_n * k)

            return_qs[gt_seg == k] = q

        return contrast_logits, contrast_target, return_qs
    def compute_log_prob(self, _fea):  # TODO: Modifications
        covariances = self.diagonal.detach_() # * c,p,d,d

        _prob_n = []
        
        _n_group = _fea.shape[0] // self.factors[0]
        if _n_group==0:
            _n_group=1
        _c_group = self.class_num // self.factors[1]
        for _c in range(0,self.class_num,_c_group):
            _prob_c = []
            _c_means = self.means[_c:_c+_c_group]
            _c_covariances = covariances[_c:_c+_c_group]

            _c_gauss = MultivariateNormalDiag(_c_means.view(-1, self.embedding_dim), scale_diag=_c_covariances.view(-1,self.embedding_dim)) # * c*p multivariate gaussian
            for _n in range(0,_fea.shape[0],_n_group):
                _prob_c.append(_c_gauss.log_prob(_fea[_n:_n+_n_group,None,...]))
            _c_probs = torch.cat(_prob_c, dim=0) # n, cp
            _c_probs = _c_probs.contiguous().view(_c_probs.shape[0], -1, self.num_prob_n)
            _prob_n.append(_c_probs)
        probs = torch.cat(_prob_n, dim=1)

        return probs.contiguous().view(probs.shape[0],-1)
    def save_proto(self, path):
        torch.save({'CoVariance': self.CoVariance.cpu(),
                    'Ave': self.Ave.cpu(),
                    'Amount': self.Amount.cpu()
                    }, path)
