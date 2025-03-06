# --> General imports
import numpy as np
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
import torch
import random
import pickle
import torch.nn as nn
from core.models.distributions import MultivariateNormalDiag  # TODO: Modifications
from core.models.GMMSeg import distributed_sinkhorn_wograd, shifted_var, init_weights, l2_normalize, momentum_update, rnd_sample  # TODO: Modifications

# --> Torch imports 
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

def save_pkl(filename, save_object):
	writer = open(filename,'wb')
	pickle.dump(save_object, writer)
	writer.close()


def load_pkl(filename):
	loader = open(filename,'rb')
	file = pickle.load(loader)
	loader.close()
	return file


def set_seed(SEED, disable_cudnn=False):
    torch.manual_seed(SEED)  # Seed the RNG for all devices (both CPU and CUDA).
    random.seed(SEED)        # Set python seed for custom operators.
    rs = RandomState(MT19937(SeedSequence(SEED)))  # If any of the libraries or code rely on NumPy seed the global NumPy RNG.
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # If you are using multi-GPU. In case of one GPU, you can use # torch.cuda.manual_seed(SEED).

    if not disable_cudnn:
        torch.backends.cudnn.benchmark = False 
        torch.backends.cudnn.deterministic = True  
    else:
        torch.backends.cudnn.enabled = False 


def collate_MoSARe(batch):
    feats, rna_data, proto_emb, label,proto_text_local,proto_text_global,mask_img,mask_rna,mask_text = zip(*batch)  #TODO: Nazanin
    return torch.stack(feats, 0), torch.stack(rna_data, 0), torch.stack(proto_emb, 0),torch.tensor(label), torch.stack(proto_text_local,0),torch.stack(proto_text_global,0),torch.tensor(mask_img),torch.tensor(mask_rna),torch.tensor(mask_text)  #TODO: Nazanin


def collate_slide(batch):
    """
    Args:
        batch (List[Tuple[torch.Tensor, int]]): List of individual data points from the dataset.
    Returns:
        features_batch (torch.Tensor): Batch of feature tensors.
        labels_batch (torch.Tensor): Batch of labels.
    """
    features_list, ids_list = zip(*batch)
    features_batch = torch.stack(features_list, dim=0)
    return features_batch, ids_list


def smooth_rank_measure(embedding_matrix, eps=1e-7):
    """
    Compute the smooth rank measure of a matrix of embeddings.
    
    Args:
        embedding_matrix (torch.Tensor): Matrix of embeddings (n x m). n: number of patch embeddings, m: embedding dimension
        alpha (float): Smoothing parameter to avoid division by zero.

    Returns:
        float: Smooth rank measure.
    """
    
    # Perform SVD on the embedding matrix
    _, S, _ = torch.svd(embedding_matrix)
    
    # Compute the smooth rank measure
    p = S / torch.norm(S, p=1) + eps
    p = p[:embedding_matrix.shape[1]]
    smooth_rank = torch.exp(-torch.sum(p * torch.log(p)))
    smooth_rank = round(smooth_rank.item(), 2)
    
    return smooth_rank

def ProtoGMM_loss(feat,
             mask,
             mean=None,
             covariance=None,
             ratio=1.0,
             index=-1,
             contrast_temp=1.,
             use_avg_pool=True,
             scale_min_ratio=0.75,
             num_classes=6,
             weight=None,
             class_weight=None,
             reduction='mean',
             avg_factor=None,
             reg_weight=0,
             ignore_index=255,
             # source=True, # TODO: Modifications
             **kwargs):
    if index >= 0:
        assert isinstance(feat, list), f'feat list expected for index={index}'
        assert isinstance(mean, (list, dict)), f'mean list expected for index={index}'
        assert isinstance(covariance, (list, dict)), f'covariance list expected for index={index}'
        feat = feat[index]
        mean = mean[index]
        covariance = covariance[index]
    
    # feat, mask = contrast_preparations(feat, mask, use_avg_pool, scale_min_ratio, num_classes, ignore_index,remove_ignore=False) # TODO: Nazanin semi general UDA
    assert mean is not None, 'Parameter `mean` required'
    assert covariance is not None, 'Parameter `covariance` required'
    assert not mean.requires_grad
    assert not covariance.requires_grad
    assert feat.requires_grad
    assert not mask.requires_grad
    criterion = nn.CrossEntropyLoss(reduction='none')

    # GMM
    # Compute log probability p(component/feature)
    _c_gauss = MultivariateNormalDiag(mean.view(-1, mean.shape[-1]), scale_diag=covariance.view(-1,covariance.shape[-1]))  # * c*p multivariate gaussian   # TODO: Modifications
    feat = l2_normalize(feat)  # TODO: Modifications
    log_prob = _c_gauss.log_prob(feat.detach()[:,None,:].cpu()).cuda()   # TODO: Modifications
    log_prob = log_prob.view(-1, mean.shape[0], mean.shape[1]) # TODO: Modifications
    value = F.softmax(log_prob.reshape(log_prob.shape[0], -1), dim=1).reshape(log_prob.shape[0], log_prob.shape[1],
                                                                log_prob.shape[2]) # TODO: Modifications

    ######mask for d^2/zigma^2   # TODO: Gaussian weighting private labeles
    msk = mask.clone()  # TODO: Gaussian weighting private labeles
    mask = torch.argmax(log_prob.view(-1,15*num_classes), -1)  # TODO: Gaussian weighting private labeles
    means = _c_gauss.loc.cuda()[mask]  # TODO: Gaussian weighting private labeles
    # Initialize an empty array to store the covariance matrices # TODO: Gaussian weighting private labeles
#             covariance_matrices = [] # TODO: Gaussian weighting private labeles

#             # Loop through each row (variance values) and create the corresponding covariance matrix  # TODO: Gaussian weighting private labeles
#             for variances_row in _c_gauss.scale_diag: # TODO: Gaussian weighting private labeles
#                 # Create a diagonal matrix with variances_row on the diagonal and zeros elsewhere # TODO: Gaussian weighting private labeles
#                 covariance_matrix = torch.diag(variances_row)   # TODO: Gaussian weighting private labeles

#                 # Append the covariance matrix to the list   # TODO: Gaussian weighting private labeles
#                 covariance_matrices.append(covariance_matrix)   # TODO: Gaussian weighting private labeles

#             # Convert the list of covariance matrices to a 3D NumPy array   # TODO: Gaussian weighting private labeles
#             covariance_matrices = torch.stack(covariance_matrices)  # TODO: Gaussian weighting private labeles
    cov2 = torch.diag_embed(_c_gauss.scale_diag.cuda())[mask].cuda()
    # The final output will be a 3D array with shape (95, 64, 64)  # TODO: Gaussian weighting private labeles

#             cov2 = covariance_matrices[mask] # TODO: Gaussian weighting private labeles
    diff = (feat.detach()-means.cuda()).unsqueeze(1).cuda() # TODO: Gaussian weighting private labeles
    # Compute the differences between each data point and the mean vectors # TODO: Gaussian weighting private labeles
#             diff = diff.unsqueeze(1).cuda()  # Shape: [51200, 1, 64] # TODO: Gaussian weighting private labeles
    # Transpose cov2 for proper multiplication # TODO: Gaussian weighting private labeles
    cov2_transposed = 1/(2*cov2.transpose(1, 2))  # Shape: [51200, 64, 64] # TODO: Gaussian weighting private labeles
    # Check for inf values and create a mask # TODO: Gaussian weighting private labeles
    inf_mask = torch.isinf(cov2_transposed) # TODO: Gaussian weighting private labeles

    # Replace inf values with zero using the mask  # TODO: Gaussian weighting private labeles
    cov2_transposed[inf_mask] = 0.0   # TODO: Gaussian weighting private labeles
    # Perform matrix multiplication to calculate the Mahalanobis distances # TODO: Gaussian weighting private labeles
    # Here, we're using torch.matmul to perform the multiplications # TODO: Gaussian weighting private labeles
    # The result will have shape [51200, 1, 1]  # TODO: Gaussian weighting private labeles
    weight_gaussian = torch.exp(-torch.matmul(torch.matmul(diff, cov2_transposed), diff.transpose(1, 2))).squeeze() # TODO: Gaussian weighting private labeles
    # print("weight_gaussian", weight_gaussian)
    # Take the square root and squeeze to get the final Mahalanobis distances  # TODO: Gaussian weighting private labeles
    # mahalanobis_distances = torch.sqrt(mahalanobis_distances).squeeze()   # TODO: Gaussian weighting private labeles
#             weight_gaussian = torch.exp(-mahalanobis_distances).squeeze()  # TODO: Gaussian weighting private labeles
    # weight_gaussian_reshape = weight_gaussian.reshape(2, 160, 160)  # TODO: Gaussian weighting private labeles
#             print(weight_gaussian_reshape.sum())





    ########
    # msk = mask.clone()  # TODO: Nazanin semi general    # TODO: Gaussian weighting private labeles

    # mask = softmax(value.sum(-1))  # TODO: Nazanin semi general UDA  # TODO: Gaussian weighting private labeles
    # mask_kl = mask.reshape(2, 160, 160, -1).permute(0, 3, 1, 2)  # TODO: Gaussian weighting private labeles
    # classifier = kwargs['ema_unlabeled_source_softmax']   # TODO: Gaussian weighting private labeles
    # weight_gaussian_reshape = resize(  # TODO: Nazanin semi general UDA
    #     input=weight_gaussian_reshape.unsqueeze(1),  # TODO: Nazanin semi general UDA
    #     size=(640,640),  # TODO: Nazanin semi general UDA
    #     mode='bilinear',  # TODO: Nazanin semi general UDA
    #     align_corners=True)  # TODO: Nazanin semi general UDA
#             print('weight_gaussian_reshape',weight_gaussian_reshape.shape)

    # # combine mask_kl with classifier
    # final_value = mask_kl*0.5 + 0.5*classifier
    # final_value = torch.argmax(final_value, 1)
    # loss_pick_1 = F.cross_entropy(mask_kl, final_value, reduce=False)
    # loss_pick_2 = F.cross_entropy(classifier, final_value, reduce=False)
    # kl_loss_1 = F.kl_div(F.log_softmax(mask_kl, dim=1), F.softmax(classifier, dim=1), reduction='none').sum(1)
    # kl_loss_2 = F.kl_div(F.log_softmax(classifier, dim=1), F.softmax(mask_kl, dim=1), reduction='none').sum(1)
    # loss_pick_640_640 = loss_pick_1+loss_pick_2+0.1*(kl_loss_1+kl_loss_2)
    # loss_pick_160_160 = F.avg_pool2d(loss_pick_640_640.float(), kernel_size=4)
    # loss_pick_160_160 = loss_pick_160_160.reshape(-1)
    # ind_sorted = torch.argsort(loss_pick_160_160)
    # remember_rate = 1 - 0.2
    # num_remember = int(remember_rate * ind_sorted.shape[0])
    # msk[ind_sorted[num_remember:]] = 255   # the last 20% remove from the loss
    mask_ = torch.argmax(value.sum(-1), axis=-1)[
        msk != 255]  # TODO: Modifications# TODO: Nazanin semi general UDA
#             # comput whole logits
#             mean = mean.cuda() 
#             logits_ = feat.mm(
#                 mean.view(-1, mean.shape[-1]).permute(1, 0).contiguous()) / contrast_temp  # TODO: Modifications
#             logits_ = logits_.view(-1, mean.shape[0], mean.shape[1])
#             logits_max_ = logits_.max(-1)[0]
#             mask_one_hot_ = F.one_hot(torch.argmax(value.sum(-1), axis=-1), mean.shape[0])
#             logits_ = mask_one_hot_ * logits_max_ + (1 - mask_one_hot_) * logits_max_
#             # comput whole logits


    if feat.size(0) == 0:
        return torch.tensor(0., requires_grad=True).cuda()
    mean = mean.cuda()  # TODO: Modifications
    # Multi ptototype contrastive learning with hard sampling

    logits = feat.mm(
        mean.cuda().view(-1, mean.shape[-1]).permute(1, 0).contiguous()) / contrast_temp  # TODO: Modifications
    if (msk!=255).sum()==0:
        loss = (logits.sum(1)*0).mean()
    else:
        feat = feat[msk != 255]  # TODO: Nazanin semi general UDA
        logits = logits[msk != 255]
        logits = logits.view(-1, mean.shape[0], mean.shape[1]) # TODO: Modifications
        mask_one_hot = F.one_hot(mask_, mean.shape[0]) #m
#         logits_min = logits.min(-1)[0]
        logits_max = logits.max(-1)[0]
        logits = mask_one_hot * logits_max + (1 - mask_one_hot) * logits_max   #max over both
        #generated the repeated masks=> TODO: Nazanin semi general UDA
        # repeated_msk = msk.unsqueeze(-1).repeat(1,1,logits.shape[-1])  # TODO: Nazanin semi general UDA
        # logits = logits[repeated_msk != 255]  # TODO: Nazanin semi general UDA
        # mask_ = mask[msk != 255]  # TODO: Nazanin semi general UDA


        loss = F.cross_entropy(
            logits,
            mask_,
            weight=class_weight,
            reduction='none',
            ignore_index=ignore_index)

        # loss = loss*weight_gaussian   ######mask for d^2/zigma^2  # TODO: Gaussian weighting private labeles


        # apply weights and do the reduction
        if weight is not None:
            weight = weight.float()
        weight_gaussian=weight_gaussian[msk != 255]
    return loss, weight_gaussian # TODO: Gaussian weighting private labeles # TODO: Nazanin KLD  loss, indices of those pixels with high kld(20%),skld



