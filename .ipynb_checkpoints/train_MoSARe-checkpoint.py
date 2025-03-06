# ---------------------------------------------------------------
# We design Novel Multimodal model callsed MoSARE 
# This implementation is based on:
# https://github.com/mahmoodlab/TANGLE
# https://github.com/MrPetrichor/MECOM
# https://github.com/NazaninMn/GenGMM
# A copy of the licenses are available at resources
# Modifications: New Method and approaches have been designed and included
# ---------------------------------------------------------------

# --> General imports
import os
import numpy as np
from tqdm import tqdm
import json 
import pdb
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import gc
import random 

# --> Torch imports 
import torch
from torch.utils.data import DataLoader
import time
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import LinearLR

# --> internal imports 
from core.models.mosare import MoSARe
from core.dataset.dataset import MoSAReDataset
from core.loss.MoSARe_loss import InfoNCE, SymCL
from core.utils.learning import smooth_rank_measure, collate_MoSARe, set_seed
from core.utils.process_args import process_args
from core.models.proto_estimator import ProtoEstimator
from core.utils.learning import ProtoGMM_loss


os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


# Set device 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def Loss_re1(res,gt,sn):
    return torch.sum(torch.sum(torch.pow(res-gt,2),dim=1) * sn)

class train_loop_class:
    def __init__(self):
        self._best_acc=0
        self._best_f1_score=0
        self._best_precission=0
        self._best_recall=0
        self._best_auc=0
        self._best_epoch=0
        self.alpha = 0.999
        self.mean_s = torch.zeros(args['epochs'], args['num_classes'] * 15, 256)  #TODO: Modifications
        self.cov_s = torch.zeros(args['epochs'], args['num_classes'] * 15, 256)  #TODO: Modifications
        self.feat_distributions_global = ProtoEstimator(dim=1024, class_num = args['num_classes'],
                                                 memory_length=5000) #TODO: Modifications
        self.threshold = 10 #TODO: Modifications

    def train_loop(self, args, loss_fn_rnaRecon, symcl, mosare_model, epoch, dataloader, optimizer, scheduler_warmup, scheduler, dataloader_val):
            
        
        loss_function = nn.CrossEntropyLoss().to(DEVICE)
        ep_loss = 0.
        fb_time = 0.

        for b_idx, (patch_emb, rna_seq, proto_wsi_local,label, proto_text_local,proto_text_global, mask_img, mask_rna, mask_text) in enumerate(dataloader):   #TODO: Modifications
            mosare_model.train()
            mosare_model.to(DEVICE)

            losses = []    
            s_fb = time.time()

            patch_emb = patch_emb.to(DEVICE)
            proto_wsi_local = proto_wsi_local.to(DEVICE)
            rna_seq = rna_seq.to(DEVICE) if rna_seq is not None else rna_seq
            proto_text_local = proto_text_local.to(DEVICE)
            proto_text_global = proto_text_global.to(DEVICE)
            # classification loss and reconstruction loss
            wsi_emb, rna_emb, rna_emb_mlp, rna_reconstruction, logits, image_all, rna_all, text_all, deco_image, deco_rna, deco_text, x, image_all_9, rna_all_9, text_all_9, \
        deco_image_9, deco_rna_9,deco_text_9, x_9,logits_img,logits_rna,final_x,final_x_text,logits_img_9,logits_rna_9,x_text, x_9_text, gate_wsi,gate_rna,gate_text, logits_text, logits_text_9   = mosare_model(patch_emb, rna_seq.unsqueeze(-1),mask_img,mask_rna,mask_text, proto_wsi_local,proto_text_local,proto_text_global) #TODO: Modifications
            loss_cls = loss_function(logits, label.cuda())+loss_function(logits_img, label.cuda())+loss_function(logits_text, label.cuda())+loss_function(logits_rna, label.cuda()) +loss_function(logits_img_9, label.cuda()) +loss_function(logits_rna_9, label.cuda()) +loss_function(logits_text_9, label.cuda()) #TODO: Modifications
            loss_reconstuction = (Loss_re1(deco_image, image_all, mask_img.cuda()) + Loss_re1(deco_rna, rna_all, mask_rna.cuda()) + \
                                    Loss_re1(deco_text, text_all, mask_text.cuda()) + \
                                Loss_re1(deco_image_9, image_all_9, mask_img.cuda()) + Loss_re1(deco_rna_9, rna_all_9,mask_rna.cuda())\
                                    + Loss_re1(deco_text_9, text_all_9,mask_text.cuda())\
                                        ) / 1024 / args['batch_size']  #TODO: Modifications
            
            losscl = loss_cls + loss_reconstuction #TODO: Modifications
            optimizer.zero_grad()  #TODO: Modifications
            losscl.backward()  #TODO: Modifications
            optimizer.step()  #TODO: Modifications
            output_feature = torch.zeros(size=(args['batch_size'], 2048))
            output_feature = output_feature.to(DEVICE)

            # update GMM
            with torch.no_grad():#TODO: Modifications
                wsi_emb, rna_emb, rna_emb_mlp, rna_reconstruction, logits, image_all, rna_all, text_all, deco_image, deco_rna, deco_text, x, image_all_9, rna_all_9, text_all_9, \
            deco_image_9, deco_rna_9,deco_text_9, x_9,logits_img,logits_rna,final_x,final_x_text,logits_img_9,logits_rna_9,x_text, x_9_text, gate_wsi,gate_rna,gate_text , logits_text, logits_text_9  = mosare_model(patch_emb, rna_seq.unsqueeze(-1),mask_img,mask_rna,mask_text,proto_wsi_local,proto_text_local,proto_text_global) #TODO: Modifications
                output_feature = final_x #TODO: Modifications
                   
            out_seg, contrast_logits, contrast_target = self.feat_distributions_global.update_proto(features=output_feature, labels=label.cuda(), epoch=epoch) #TODO: Modifications
            mean_global = self.feat_distributions_global.means  # TODO: Modifications
            covariance_global = self.feat_distributions_global.diagonal # TODO: Modifications
            # Aligning loss
            loss_cont=0 # TODO: Modifications
            loss_protogmm=0 # TODO: Modifications
            if epoch >= self.threshold:
                mosare_model.train() # TODO: Modifications
                # symmetric contrastive loss
                wsi_emb, rna_emb, rna_emb_mlp, rna_reconstruction, logits, image_all, rna_all, text_all, deco_image, deco_rna, deco_text, x, image_all_9, rna_all_9, text_all_9, \
            deco_image_9, deco_rna_9,deco_text_9, x_9,logits_img,logits_rna,final_x,final_x_text,logits_img_9,logits_rna_9,x_text, x_9_text,gate_wsi,gate_rna,gate_text, logits_text, logits_text_9  = mosare_model(patch_emb, rna_seq.unsqueeze(-1),mask_img,mask_rna,mask_text, proto_wsi_local, proto_text_local,proto_text_global) #TODO: Modifications
                # global
                losses1=symcl(query=image_all, positive_key=x, symmetric=True) # TODO: Modifications
                losses2=symcl(query=rna_all, positive_key=x, symmetric=True) # TODO: Modifications
                losses3=symcl(query=text_all, positive_key=x, symmetric=True) # TODO: Modifications
                aligning_loss_global = (losses1+losses2+losses3)/3 # TODO: Modifications

                # Local
                losses1=symcl(query=image_all_9, positive_key=x_9, symmetric=True) # TODO: Modifications
                losses2=symcl(query=rna_all_9, positive_key=x_9, symmetric=True) # TODO: Modifications
                losses3=symcl(query=text_all_9, positive_key=x_9, symmetric=True) # TODO: Modifications
                aligning_loss_local = (losses1+losses2+losses3)/3 # TODO: Modifications

                loss_cont= (aligning_loss_global + aligning_loss_local)*0.5 # TODO: Modifications
                optimizer.zero_grad() # TODO: Modifications
                loss_cont.backward() # TODO: Modifications
                optimizer.step() # TODO: Modifications

                # protoGMM loss
                wsi_emb, rna_emb, rna_emb_mlp, rna_reconstruction, logits, image_all, rna_all, text_all, deco_image, deco_rna, deco_text, x, image_all_9, rna_all_9, text_all_9, \
            deco_image_9, deco_rna_9,deco_text_9, x_9,logits_img,logits_rna,final_x,final_x_text,logits_img_9,logits_rna_9,x_text, x_9_text, gate_wsi,gate_rna,gate_text, logits_text, logits_text_9  = mosare_model(patch_emb, rna_seq.unsqueeze(-1),mask_img,mask_rna,mask_text, proto_wsi_local, proto_text_local,proto_text_global) # TODO: Modifications


                loss_protogmm, weight_proto = ProtoGMM_loss(feat=final_x, mask=label.reshape(-1), mean=mean_global, covariance=covariance_global,
                                            source=False
                                            , class_sourse=None,
                                            class_target=None,
                                            target_memory=self.feat_distributions_global.queue_target, num_classes=args['num_classes']) # TODO: Modifications
                loss_protogmm = (loss_protogmm).mean()*0.5     # TODO: Modifications

                optimizer.zero_grad()   # TODO: Modifications
                loss_protogmm.backward()   # TODO: Modifications
                optimizer.step()   # TODO: Modifications


            
            e_fb = time.time()
            fb_time += e_fb - s_fb

            if epoch <= args["warmup_epochs"]:
                scheduler_warmup.step()
            else:
                scheduler.step()  
                
            if (b_idx % 3) == 0:
                print(f"Loss for batch: {b_idx} = {losscl}, {loss_cont}, {losscl+loss_cont+loss_protogmm}")
                
            loss = losscl+loss_cont+loss_protogmm     # TODO: Modifications
            ep_loss += loss    # TODO: Modifications

        num_samples = 0
        value_acc = 0
        value_f1_score = 0
        value_precision = 0
        value_recall = 0
        value_roc_auc = 0
        value_acc_1=0
        value_acc_9=0
        y_pred = []
        y_true = []
        # Validation evaluation
        for b_idx, (patch_emb_val, rna_seq_val, proto_wsi_local_val,label_val, proto_text_local_val,proto_text_global_val, mask_img_val, mask_rna_val, mask_text_val) in enumerate(dataloader_val):      # TODO: Modifications
            mosare_model.eval()      # TODO: Modifications
            patch_emb_val = patch_emb_val.to(DEVICE)      # TODO: Modifications
            proto_wsi_local_val = proto_wsi_local_val.to(DEVICE)      # TODO: Modifications
            rna_seq_val = rna_seq_val.to(DEVICE) if rna_seq_val is not None else rna_seq_val      # TODO: Modifications
            proto_text_local_val = proto_text_local_val.to(DEVICE)      # TODO: Modifications
            proto_text_global_val = proto_text_global_val.to(DEVICE)      # TODO: Modifications

            with torch.no_grad():      # TODO: Modifications
                        wsi_emb_val, rna_emb_val, rna_emb_mlp_val, rna_reconstruction_val, logits_val, image_all_val, rna_all_val, text_all_val, deco_image_val, deco_rna_val, deco_text_val, x_val, image_all_9_val, rna_all_9_val, text_all_9_val, \
            deco_image_9_val, deco_rna_9_val,deco_text_9_val, x_9_val,logits_img_val,logits_rna_val,final_x_val,final_x_text,logits_img_9,logits_rna_9,x_text, x_9_text, gate_wsi,gate_rna,gate_text, logits_text, logits_text_9 = mosare_model(patch_emb_val.cuda(), rna_seq_val.unsqueeze(-1).cuda(),mask_img_val.cuda(),mask_rna_val.cuda(),mask_text_val.cuda(), proto_wsi_local_val.cuda(),proto_text_local_val,proto_text_global_val)      # TODO: Modifications

                        test_predict = torch.max(logits_val, dim=1)[1]      # TODO: Modifications
                        value_acc += torch.eq(test_predict, label_val.cuda()).sum().item()      # TODO: Modifications
                        num_samples +=  patch_emb_val.shape[0]      # TODO: Modifications
                        y_pred+= [int(t.item()) for t in test_predict.cpu()]      # TODO: Modifications
                        y_true+= [int(t.item()) for t in label_val.cpu()]      # TODO: Modifications
        value_acc = value_acc / num_samples      # TODO: Modifications
        value_f1_score = f1_score(y_true, y_pred)      # TODO: Modifications
        value_precision = precision_score(y_true, y_pred)      # TODO: Modifications
        value_recall = recall_score(y_true, y_pred)      # TODO: Modifications
        value_roc_auc = roc_auc_score(y_true, y_pred)        # TODO: Modifications

        
        if (value_roc_auc > self._best_auc):
            self._best_epoch = epoch
            self._best_auc = value_roc_auc
            print('Better AUC: Saving model')
            torch.save(mosare_model.state_dict(), os.path.join('logs', "model.pt"))
                
        if (value_recall > self._best_recall):
            self._best_recall = value_recall  
        if (value_precision > self._best_precission):
            self._best_precission = value_precision
        if (value_f1_score > self._best_f1_score):
            self._best_f1_score = value_f1_score
        if (value_acc > self._best_acc):
            self._best_acc = value_acc
        
                

        print("best_epoch = {:.0f} ===> epoch = {:.0f} ===> loss = {:.6f} ===> Best_auc = {:.4f} ===> Auc = {:.4f} ===> Recall = {:.4f} ===> Precision = {:.4f} ===> F1-score = {:.4f} ===> Acc = {:.4f}"
                            .format(self._best_epoch + 1, epoch + 1, loss,
                                    self._best_auc, value_roc_auc,value_recall,value_precision,value_f1_score,value_acc))
        gc.collect()  # Free CPU memory    
        return ep_loss 



def write_dict_to_config_file(config_dict, json_file_path):
    """
    Write a dictionary to a configuration file.
    Args:
        config_dict (dict): The dictionary to be written to the config file.
        config_file_path (str): The path to the configuration file.
    Returns:
        None
    """
    with open(json_file_path, 'w') as jsonfile:
        json.dump(config_dict, jsonfile, indent=4)


if __name__ == "__main__":
    
    # setup args and seed
    args = process_args()
    args = vars(args)
    set_seed(args["seed"])


    # Set params for loss computation 
    RNA_RECONSTRUCTION = False 
    INTRA_MODALITY = False 
    N_TOKENS_RNA = 4908 if args["study"]=='nsclc' else 4999

    args["rna_reconstruction"] = RNA_RECONSTRUCTION
    args["intra_modality_wsi"] = INTRA_MODALITY
    args["rna_token_dim"] = N_TOKENS_RNA

    # paths 
    ROOT_SAVE_DIR = "./logs/{}_checkpoints_and_embeddings".format(args["study"])
    EXP_CODE = "{}_{}_lr{}_epochs{}_bs{}_tokensize{}_temperature{}_uni".format(
        args["method"],
        args["study"],
        args["learning_rate"], 
        args["epochs"], 
        args["batch_size"], 
        args["n_tokens"],
        args["temperature"]
    )
    RESULTS_SAVE_PATH = os.path.join(ROOT_SAVE_DIR, EXP_CODE)
    os.makedirs(RESULTS_SAVE_PATH, exist_ok=True)
    write_dict_to_config_file(args, os.path.join(RESULTS_SAVE_PATH, "config.json"))

    print()
    print(f"Running experiment {EXP_CODE}...")
    print()
    
    # Create a SummaryWriter
    log_dir = os.path.join(ROOT_SAVE_DIR, 'logs', EXP_CODE)
    os.makedirs(log_dir, exist_ok=True)
    
    # make MoSARe dataset
    print("* Setup dataset...")

    dataset = MoSAReDataset(
        feats_dir_wsi='./data/NSCLC/feats_pt_wsi', 
        rna_dir='./data/NSCLC/rna_all.csv', 
        sampling_strategy=args["sampling_strategy"], 
        n_tokens=args["n_tokens"],
        local_wsi_dir = './data/NSCLC/proto_save_wsi',     
        meta_dir= './data/NSCLC/train_fold_'+str( args["fold"])+'.csv',  
        text_local= './data/NSCLC/proto_save_text/',  
        text_global= './data/NSCLC/global_text/',  
        missing_percentage=args['mask_percentage']  
    )

    dataset_val = MoSAReDataset(
        feats_dir_wsi='./data/NSCLC/feats_pt_wsi', 
        rna_dir='./data/NSCLC/rna_all.csv', 
        sampling_strategy=args["sampling_strategy"], 
        n_tokens=args["n_tokens"],
        local_wsi_dir = './data/NSCLC/proto_save_wsi',      
        meta_dir= './data/NSCLC/test_fold_'+str( args["fold"])+'.csv',  
        text_local= './data/NSCLC/proto_save_text/', 
        text_global= './data/NSCLC/global_text/',  
        missing_percentage=args['mask_percentage']   
    )
    
    # set up dataloader
    print("* Setup dataloader...")
    dataloader = DataLoader(
        dataset, 
        batch_size=args["batch_size"], 
        shuffle=True, 
        collate_fn=collate_MoSARe
    )

    print("* Setup val dataloader...")
    dataloader_val = DataLoader(
        dataset_val, 
        batch_size=args["batch_size"], 
        shuffle=True, 
        collate_fn=collate_MoSARe
    )

    print("* Setup model...")
    
    mosare_model = MoSARe(config=args, n_tokens_rna=N_TOKENS_RNA,number_classes=args['num_classes']).to(DEVICE)
    
    if len(args["gpu_devices"]) > 1:
        print(f"* Using {torch.cuda.device_count()} GPUs.")
        mosare_model = nn.DataParallel(mosare_model, device_ids=args["gpu_devices"])
    mosare_model.to("cuda:0")
    
    # set up optimizers
    print("* Setup optimizer...")
    optimizer = optim.AdamW(mosare_model.parameters(), lr=args["learning_rate"])
    
    # set up schedulers
    print("* Setup schedulers...")
    T_max = (args["epochs"] - args["warmup_epochs"]) * len(dataloader) if args["warmup"] else args["epochs"] * len(dataloader)
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=T_max,
        eta_min=args["end_learning_rate"]
    )
    
    if args["warmup"]:
        scheduler_warmup = LinearLR(
            optimizer, 
            start_factor=0.00001,
            total_iters=args["warmup_epochs"] * len(dataloader)
    )
    else:
        scheduler_warmup = None
    
    # set up losses
    print("* Setup losses...")
    loss_fn_rnaRecon = nn.MSELoss()
    symcl = SymCL(args) 
    train_loop_c = train_loop_class()
    # main training loop
    best_rank = 0.

    for epoch in range(args["epochs"]):
        
        print()
        print(f"Training for epoch {epoch}...")
        print()
        
        # train
        start = time.time()
        ep_loss = train_loop_c.train_loop(args, loss_fn_rnaRecon, symcl, mosare_model, epoch, dataloader, optimizer, scheduler_warmup, scheduler,dataloader_val)
        end = time.time()

        print()
        print(f"Done with epoch {epoch}")
        print(f"Total loss = {ep_loss}")
        print("Total time = {:.3f} seconds".format(end-start))

    
    print()
    print("Done")
    print()