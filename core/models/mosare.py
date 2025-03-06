
# --> Torch imports 
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
import math
# --> Internal imports 
from .abmil import BatchedABMIL

import pdb


class MoSARe(nn.Module):
    def __init__(self, config, n_tokens_rna, number_classes, drop_pro=0.1):
        super(MoSARe, self).__init__()

        self.config = config
        self.n_tokens_wsi = config['n_tokens'] 
        self.patch_embedding_dim = config['embedding_dim']
        self.n_tokens_rna = n_tokens_rna
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.number_classes=number_classes
        # Define MoE parameters
        num_experts = 5
        expert_dim = 1024  # Keeping feature size the same
        top_k = 2

        # Initialize MoE for each modality inside the model
        self.moE_wsi = MoE_hard(input_dim=1024, num_experts=num_experts, expert_dim=expert_dim, top_k=top_k)
        self.moE_rna = MoE_hard(input_dim=1024, num_experts=num_experts, expert_dim=expert_dim, top_k=top_k)
        self.moE_text = MoE_hard(input_dim=1024, num_experts=num_experts, expert_dim=expert_dim, top_k=top_k)
        
        self.final = MoE_hard_final(input_dim=1024, num_experts=num_experts, expert_dim=expert_dim, top_k=top_k)

        ########## WSI embedder #############
        if self.config["wsi_encoder"] == "abmil":
            assert self.config["n_heads"] == 1, "ABMIL must have only 1 head"
            pre_params = {
                "input_dim": self.patch_embedding_dim,
                "hidden_dim": self.patch_embedding_dim,
            }
            attention_params = {
                "model": "ABMIL",
                "params": {
                    "input_dim": self.patch_embedding_dim,
                    "hidden_dim": self.config["hidden_dim"],
                    "dropout": True,
                    "activation": self.config["activation"],
                    "n_classes": 1,
                },
            }
            self.wsi_embedder = ABMILEmbedder(
                pre_attention_params=pre_params,
                attention_params=attention_params,
            )

        elif self.config["wsi_encoder"] == "abmil_mh":
            assert self.config["n_heads"] > 1, "ABMIL_MH must have more than 1 head"

            pre_params = {
                "input_dim": self.patch_embedding_dim,
                "hidden_dim": self.config["hidden_dim"],
            }
            attention_params = {
                "model": "ABMIL",
                "params": {
                    "input_dim": self.config["hidden_dim"],
                    "hidden_dim": self.config["hidden_dim"],
                    "dropout": True,
                    "activation": self.config["activation"],
                    "n_heads": self.config["n_heads"],
                    "n_classes": 1,
                },
            }

            
            self.wsi_embedder = ABMILEmbedder_MH(
                pre_attention_params=pre_params,
                attention_params=attention_params,
            )


        ########## RNA embedder: Linear or MLP or abmil_mh #############
        pre_params_rna = {      #TODO: Modification
                "input_dim": n_tokens_rna,
                "hidden_dim": self.config["hidden_dim"],
            }
        attention_params_rna = {       #TODO: Modification
                "model": "ABMIL",
                "params": {
                    "input_dim": self.config["hidden_dim"],
                    "hidden_dim": self.config["hidden_dim"],
                    "dropout": True,
                    "activation": self.config["activation"],
                    "n_heads": 16,
                    "n_classes": 1,
                },
            }
        if self.config["rna_encoder"] == "mlp":
                self.rna_embedder = MLP_rna(input_dim=n_tokens_rna, hidden_dim=n_tokens_rna, output_dim=self.config["embedding_dim"])
        elif self.config["rna_encoder"] == "abmil_mh":  #TODO: Modification
                self.rna_embedder = ABMILEmbedder_MH(
                pre_attention_params=pre_params_rna,
                attention_params=attention_params_rna,
            )
        self.rna_embedder_mlp = MLP_rna(input_dim=n_tokens_rna, hidden_dim=n_tokens_rna, output_dim=self.config["embedding_dim"])    
        ########## RNA Reconstruction module: Linear or MLP #############
        if self.config["rna_reconstruction"]:
            if self.config["rna_encoder"] == "linear":
                self.rna_reconstruction = nn.Linear(in_features=self.config["embedding_dim"], out_features=n_tokens_rna)
            else:
                self.rna_reconstruction = MLP(input_dim=self.config["embedding_dim"], hidden_dim=self.config["embedding_dim"], output_dim=n_tokens_rna)
        else:
            self.rna_reconstruction = None
    
        ########## Projection Head #############
        if self.config["embedding_dim"] != self.config["hidden_dim"]:
            self.mean_projector = ProjHead(
                self.config["embedding_dim"],
                self.config["hidden_dim"],
            )
        else:
            self.mean_projector = None
        
        #TODO: Modification
        self.fc_2048_100 = nn.Linear(1024, self.number_classes)  #TODO: Modification 
        self.fc_4096_100 = nn.Linear(1024, self.number_classes)  #TODO: Modification
        self.fc_2048_100_1 = nn.Linear(1024, self.number_classes)  #TODO: Modification
        self.fc_2048_100_9 = nn.Linear(1024, self.number_classes)  #TODO: Modification

        self.instance_image = nn.Sequential(
                nn.Linear(1024, 1),
                nn.ReLU()
        )
        self.instance_rna = nn.Sequential(
                nn.Linear(1024, 1),
                nn.ReLU()
        )

        self.instance_text = nn.Sequential(
                nn.Linear(1024, 1),
                nn.ReLU()
        )

        if 1:
            self.image_rna = GA(1024, drop_pro, 4, 256)  #TODO: Modification
            self.image_text = GA(1024, drop_pro, 4, 256)
            self.rna_image = GA(1024, drop_pro, 4, 256)
            self.rna_text = GA(1024, drop_pro, 4, 256)
            self.text_image = GA(1024, drop_pro, 4, 256)
            self.text_rna = GA(1024, drop_pro, 4, 256)

            self.image_rna_9 = GA_9(1024, drop_pro, 4, 256)
            self.image_text_9 = GA_9(1024, drop_pro, 4, 256)
            self.rna_image_9 = GA_9(1024, drop_pro, 4, 256)
            self.rna_text_9 = GA_9(1024, drop_pro, 4, 256)
            self.text_image_9 = GA_9(1024, drop_pro, 4, 256)
            self.text_rna_9 = GA_9(1024, drop_pro, 4, 256)

        if 1:
            self.de_image_2048 = nn.Sequential(
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
            )

            self.de_rna_2048 = nn.Sequential(
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
            )

            self.de_text_2048 = nn.Sequential(
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
            )

            self.de_image_2048_9 = nn.Sequential(
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
            )

            self.de_rna_2048_9 = nn.Sequential(
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
            )

            self.de_text_2048_9 = nn.Sequential(
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
            )

    def forward(self, wsi_emb, rna_emb=None,mask_img=None,mask_rna=None, mask_text=None, proto_wsi_local=None,proto_text_local=None,proto_text_global=None):  #TODO: Modification
        
        wsi_emb = self.wsi_embedder(wsi_emb) #wsi_emb [32, 2048, 1024]).  wsi_emb[32, 1024]) using attentions


        if self.config["intra_modality_wsi"] or rna_emb is None or self.config['rna_reconstruction']:
            rna_emb = None
        else:
            rna_emb_mlp = self.rna_embedder_mlp(rna_emb.squeeze(-1))  #[32,1024] using multi-head attention
            rna_emb = self.rna_embedder(rna_emb)  #[32,1024,16] using multi-head attention
        
        if self.config["rna_reconstruction"]:
            rna_reconstruction = self.rna_reconstruction(wsi_emb)
        else:
            rna_reconstruction = None
        # masking
        wsi_emb=wsi_emb*mask_img.reshape(mask_img.size()[0], 1).cuda().detach() #TODO: Modification
        rna_emb_mlp=rna_emb_mlp*mask_rna.reshape(mask_rna.size()[0], 1).cuda().detach() #TODO: Modification
        rna_emb=rna_emb*mask_rna.reshape(mask_rna.size()[0], 1,1).cuda().detach() #TODO: Modification
        # WSI global
        image_rna_a,map = self.image_rna(wsi_emb, rna_emb_mlp)  #TODO: Modification
        proto_text_global = proto_text_global.squeeze(1)
        image_text_a,map = self.image_text(wsi_emb, proto_text_global)
        image_a = (image_rna_a + image_text_a) / 2     #TODO: Modification
        image_all = torch.mul(image_a, mask_img.reshape(mask_img.size()[0], 1).cuda().detach())      #TODO: Modification
        
        # RNA global
        rna_image_a,map = self.rna_image(rna_emb_mlp, wsi_emb)
        rna_text_a,map = self.rna_text(rna_emb_mlp, proto_text_global)
        rna_a = (rna_image_a + rna_text_a ) / 2
        rna_all = torch.mul(rna_a, mask_rna.reshape(mask_rna.size()[0], 1).cuda().detach())

        # Report global
        text_image_a,map = self.text_image(proto_text_global, wsi_emb)
        text_rna_a,map = self.text_rna(proto_text_global, rna_emb_mlp)
        text_a = (text_image_a + text_rna_a ) / 2
        text_all = torch.mul(text_a, mask_text.reshape(mask_text.size()[0], 1).cuda().detach())

        x = (image_all + rna_all+ text_all)
        
        
        deco_image = self.de_image_2048(x)
        deco_rna = self.de_rna_2048(x)
        deco_text = self.de_text_2048(x)

        mask_img_re = (mask_img - 1) * (-1) #TODO: Modification
        mask_rna_re = (mask_rna - 1) * (-1)  ##TODO: Modification
        mask_text_re = (mask_text - 1) * (-1)  #TODO: Modification

        full_image = deco_image * mask_img_re.reshape(mask_img_re.size()[0], 1).cuda().detach() + image_all     #TODO: Modification
        full_rna = deco_rna * mask_rna_re.reshape(mask_rna_re.size()[0], 1).cuda().detach() + rna_all #TODO: Modification
        full_text = deco_text * mask_text_re.reshape(mask_text_re.size()[0], 1).cuda().detach() + text_all #TODO: Modification

        full_x = full_image + full_rna+full_text #TODO: Modification
        
        # MOE 
        # Apply MoE to local representations instead of Gumbel-Softmax
        proto_wsi_local, gate_wsi = self.moE_wsi(proto_wsi_local.cuda())   #TODO: Modification
        rna_9, gate_rna = self.moE_rna(rna_emb.permute(0, 2, 1).cuda())   #TODO: Modification
        text_9, gate_text = self.moE_text(proto_text_local.cuda())   #TODO: Modification


        image_rna_a_9, map = self.image_rna_9(proto_wsi_local, rna_9)     #TODO: Modification

        image_text_a_9, map = self.image_text_9(proto_wsi_local, text_9)    #TODO: Modification
        image_a_9 = (image_rna_a_9 + image_text_a_9 ) / 2
        image_all_9 = torch.mul(image_a_9, mask_img.reshape(mask_img.size()[0], 1,1).cuda().detach())    #TODO: Modification

        rna_image_a_9, map = self.rna_image_9(rna_9, proto_wsi_local)        #TODO: Modification
        rna_text_a_9, map = self.rna_text_9(rna_9, text_9)
        rna_a_9 = (rna_image_a_9 + rna_text_a_9 ) / 2
        rna_all_9 = torch.mul(rna_a_9, mask_rna.reshape(mask_rna.size()[0], 1,1).cuda().detach())     #TODO: Modification

        text_image_a_9, map = self.text_image_9(text_9, proto_wsi_local)
        text_rna_a_9, map = self.text_rna_9(text_9, rna_9)
        text_a_9 = (text_image_a_9 + text_rna_a_9 ) / 2
        text_all_9 = torch.mul(text_a_9, mask_text.reshape(mask_text.size()[0], 1,1).cuda().detach())

        image_all_9 = torch.sum(image_all_9, dim=1) / 2
        rna_all_9 = torch.sum(rna_all_9, dim=1) / 2
        text_all_9 = torch.sum(text_all_9, dim=1) / 2
        x_9 = image_all_9 + rna_all_9 + text_all_9

        deco_image_9 = self.de_image_2048_9(x_9)
        deco_rna_9 = self.de_rna_2048_9(x_9)
        deco_text_9 = self.de_text_2048_9(x_9)

        full_image_9 = deco_image_9 * mask_img_re.reshape(mask_img_re.size()[0], 1).cuda().detach() + image_all_9
        full_rna_9 = deco_rna_9 * mask_rna_re.reshape(mask_rna_re.size()[0], 1).cuda().detach() + rna_all_9
        full_text_9 = deco_text_9 * mask_text_re.reshape(mask_text_re.size()[0], 1).cuda().detach() + text_all_9
        stacked_tensor = torch.stack([full_image_9, full_rna_9, full_text_9, full_image, full_rna, full_text], dim=1)

        final = self.final(stacked_tensor)
        final_x = final[0].sum(1)
        


        logits = self.fc_4096_100(final_x)
        logits_img = self.fc_2048_100_1(full_image)
        logits_rna = self.fc_2048_100_1(full_rna)
        logits_text = self.fc_2048_100_1(full_text)

        logits_img_9 = self.fc_2048_100_9(full_image_9)
        logits_rna_9 = self.fc_2048_100_9(full_rna_9)
        logits_text_9 = self.fc_2048_100_9(full_text_9)
        final_x_text=[]

        gate_wsi=gate_rna=gate_text=0
        x_text=x_9_text=x
        return wsi_emb, rna_emb, rna_emb_mlp, rna_reconstruction, logits, image_all, rna_all, text_all, deco_image, deco_rna, deco_text, x, image_all_9, rna_all_9, text_all_9, \
            deco_image_9, deco_rna_9,deco_text_9, x_9,logits_img,logits_rna,final_x,final_x_text,logits_img_9,logits_rna_9,x_text, x_9_text, gate_wsi,gate_rna,gate_text, logits_text, logits_text_9     
    def get_features(self, wsi_emb):
        wsi_emb = self.wsi_embedder(wsi_emb)
        return wsi_emb
        
    def get_slide_attention(self, wsi_emb):
        _, attention = self.wsi_embedder(wsi_emb, return_attention=True)
        return attention
        
    def get_expression_features(self, rna_emb):
        rna_emb = self.rna_embedder(rna_emb)
        return rna_emb
        

class MLP_rna(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP_rna, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.blocks=nn.Sequential(
            self.build_block(in_dim=self.input_dim, out_dim=hidden_dim),
            self.build_block(in_dim=hidden_dim, out_dim=hidden_dim),
            nn.Linear(in_features=hidden_dim, out_features=self.output_dim),
        )
        

    def build_block(self, in_dim, out_dim):
        return nn.Sequential(
                nn.Linear(in_features=in_dim, out_features=out_dim),
                nn.LayerNorm(out_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
        )

    def forward(self, x):
        x = self.blocks(x)
        return x
    

class ProjHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ProjHead, self).__init__()
        self.layers = nn.Linear(in_features=input_dim, out_features=int(output_dim))

    def forward(self, x):
        x = self.layers(x)
        return x


class ABMILEmbedder(nn.Module):
    """
    """

    def __init__(
        self,
        pre_attention_params: dict = None,
        attention_params: dict = None,
        aggregation: str = 'regular',
    ) -> None:
        """
        """
        super(ABMILEmbedder, self).__init__()

        # 1- build pre-attention params 
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.pre_attention_params = pre_attention_params
        if pre_attention_params is not None:
            self._build_pre_attention_params(params=pre_attention_params)

        # 2- build attention params
        self.attention_params = attention_params
        if attention_params is not None:
            self._build_attention_params(
                attn_model=attention_params['model'],
                params=attention_params['params']
            )

        # 3- set aggregation type 
        self.agg_type = aggregation  # Option are: mean, regular, additive, mean_additive

    def _build_pre_attention_params(self, params):
        """
        Build pre-attention params 
        """
        self.pre_attn = nn.Sequential(
            nn.Linear(params['input_dim'], params['hidden_dim']),
            nn.LayerNorm(params['hidden_dim']),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(params['hidden_dim'], params['hidden_dim']),
            nn.LayerNorm(params['hidden_dim']),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def _build_attention_params(self, attn_model='ABMIL', params=None):
        """
        Build attention params 
        """
        if attn_model == 'ABMIL':
            self.attn = BatchedABMIL(**params)
        else:
            raise NotImplementedError('Attention model not implemented -- Options are ABMIL, PatchGCN and TransMIL.')
        

    def forward(
        self,
        bags: torch.Tensor,
        return_attention: bool = False, 
    ) -> torch.tensor:
        """
        Foward pass.

        Args:
            bags (torch.Tensor): batched representation of the tokens 
            return_attention (bool): if attention weights should be returned (raw attention)
        Returns:
            torch.tensor: Model output.
        """

        # pre-attention
        if self.pre_attention_params is not None:
            embeddings = self.pre_attn(bags)
        else:
            embeddings = bags

        # compute attention weights  
        if self.attention_params is not None:
            if return_attention:
                attention, raw_attention = self.attn(embeddings, return_raw_attention=True) #[64, 2048, 1]
            else:
                attention = self.attn(embeddings)  # return post softmax attention [64, 2048, 1]

        if self.agg_type == 'regular':
            embeddings = embeddings * attention #[64, 2048, 1024] => attention[64,2048] multiply emb, for 
            if self.attention_params["params"]["activation"] == "sigmoid":
                slide_embeddings = torch.mean(embeddings, dim=1)
            else:
                slide_embeddings = torch.sum(embeddings, dim=1)

        else:
            raise NotImplementedError('Agg type not supported. Options are "additive" or "regular".')

        if return_attention:
            return slide_embeddings, raw_attention
        
        return slide_embeddings


class ABMILEmbedder_MH(nn.Module):
    """ """

    def __init__(
        self,
        pre_attention_params: dict = None,
        attention_params: dict = None,
        aggregation: str = "regular",
    ) -> None:
        """ """
        super(ABMILEmbedder_MH, self).__init__()

        # 1- build pre-attention params
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pre_attention_params = pre_attention_params
        self.n_heads = attention_params["params"]["n_heads"]

        if pre_attention_params is not None:
            self._build_pre_attention_params(params=pre_attention_params)

        # 2- build attention params
        self.attention_params = attention_params
        if attention_params is not None:
            self._build_attention_params(
                attn_model=attention_params["model"], params=attention_params["params"]
            )

        # 3- set aggregation type
        self.agg_type = (
            aggregation  # Option are: mean, regular, additive, mean_additive
        )

        # 4- projection post multi-head
        # self.proj_multihead = ProjHead(input_dim=pre_attention_params['hidden_dim']*self.n_heads, output_dim=pre_attention_params['hidden_dim'])
        self.proj_multihead = nn.Linear(
            in_features=pre_attention_params["hidden_dim"] * self.n_heads,
            out_features=pre_attention_params["hidden_dim"],
        )

    def _build_pre_attention_params(self, params):
        """
        Build pre-attention params
        """

        self.pre_attn = nn.Sequential(
            nn.Linear(1, params["hidden_dim"]),
            nn.LayerNorm(params["hidden_dim"]),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(params["hidden_dim"], params["hidden_dim"]),
            nn.LayerNorm(params["hidden_dim"]),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(params["hidden_dim"], params["hidden_dim"] * self.n_heads),
            nn.LayerNorm(params["hidden_dim"] * self.n_heads),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.linear_feature_resize = nn.Linear(256, 1024)  #==>256 to 1024


    def _build_attention_params(self, attn_model="ABMIL", params=None):
        """
        Build attention params
        """
        if attn_model == "ABMIL":
            del params['n_heads']
            self.attn = nn.ModuleList(
                [BatchedABMIL(**params).to(self.device) for i in range(self.n_heads)]
            )
        else:
            raise NotImplementedError(
                "Attention model not implemented -- Options are ABMIL, PatchGCN and TransMIL."
            )

    def forward(
        self,
        bags: torch.Tensor,
        return_attention: bool = False,
        n_views=1,
    ) -> torch.tensor:
        """
        Foward pass.

        Args:
            bags (torch.Tensor): batched representation of the tokens
            return_attention (bool): if attention weights should be returned (raw attention)
        Returns:
            torch.tensor: Model output.
        """
        # pre_attn common to all heads
        if self.pre_attention_params is not None:
            embeddings = self.pre_attn(bags) #[32, 4999, 4096=16*256]
        else:
            embeddings = bags
    
        embeddings = rearrange(embeddings, "b n (d h) -> b n d h", h=self.n_heads) #[32, 4999, 256, 16])

        # get attention for each head
        attention = []
        raw_attention = []
        for i, attn_net in enumerate(self.attn):
            c_attention, c_raw_attention = attn_net(
                embeddings[:, :, :, i], return_raw_attention=True
            )
            attention.append(c_attention)
            raw_attention.append(c_raw_attention)
        attention = torch.stack(attention, dim=-1)  # return post softmax attention #[32, 4999, 1, 16])
        raw_attention = torch.stack(
            raw_attention, dim=-1
        )  # return post softmax attention

        if self.agg_type == "regular":
            if n_views == 1:
                embeddings = embeddings * attention
                slide_embeddings = torch.sum(embeddings, dim=1)
            else:
                list_of_indices = [
                    np.random.randint(0, embeddings.shape[1] - 1, 1024)
                    for _ in range(n_views)
                ]
                embeddings = torch.cat(
                    [
                        embeddings[:, indices, :, :].unsqueeze(1)
                        for indices in list_of_indices
                    ],
                    dim=1,
                )  # additional dimension for heads
                attention = torch.cat(
                    [
                        F.softmax(raw_attention[:, indices, :, :], dim=1).unsqueeze(1)
                        for indices in list_of_indices
                    ],
                    dim=1,
                )  # additional dimension for heads
                embeddings = embeddings * attention
                slide_embeddings = torch.sum(embeddings, dim=2)
                slide_embeddings = rearrange(
                    slide_embeddings, "b nv e nh -> (b nv) e nh"
                )

        else:
            raise NotImplementedError(
                'Agg type not supported. Options are "additive" or "regular".'
            )

        

        # Apply the linear layer to the input tensor
        # You need to permute the tensor to align the feature dimension
        slide_embeddings = slide_embeddings.permute(0, 2, 1)  # [32, 16, 256]
        slide_embeddings = self.linear_feature_resize(slide_embeddings)  # [32, 16, 1024]

        # Permute back to original shape
        slide_embeddings = slide_embeddings.permute(0, 2, 1)  # [32, 1024, 16]
        return slide_embeddings #[32, 1024, 16])

class FC_a(nn.Module):
    def __init__(self, in_size, out_size, dropout_r=0., use_relu=True):
        super(FC_a, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu

        self.linear = nn.Linear(in_size, out_size)

        if use_relu:
            self.relu = nn.ReLU(inplace=True)

        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        x = self.linear(x)

        if self.use_relu:
            x = self.relu(x)

        if self.dropout_r > 0:
            x = self.dropout(x)

        return x



# ------------------------------
# Mixture of Experts (MoE) Module
# ------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

    
class MoE_hard(nn.Module):
    def __init__(self, input_dim, num_experts, expert_dim, top_k=2, active_k=1):
        super(MoE_hard, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k  # Number of experts selected per local representation
        self.active_k = active_k  # Number of active local representations

        # Expert Networks (each expert is an independent linear layer)
        # self.experts = nn.ModuleList([nn.Linear(input_dim, expert_dim) for _ in range(num_experts)])
        self.experts = nn.ModuleList([
    nn.Sequential(
        nn.Linear(input_dim, expert_dim),
        nn.ReLU(),  # Add ReLU activation
        nn.Linear(expert_dim, expert_dim)  # Optional: Second layer
    ) for _ in range(num_experts)])

        # Gating Network - Determines which experts to use
        self.gate = nn.Linear(input_dim, num_experts)

        # Local Representation Masking - Determines which local positions are active
        self.local_gate = nn.Linear(input_dim, 1)  # Outputs a score for each local representation

    def forward(self, x):
        """
        x shape: (batch_size, num_local_representations, feature_dim) = (32, 16, 1024)
        """

        batch_size, num_local, feature_dim = x.shape

        # Step 1: Compute Gating Scores (Expert Selection)**
        gate_scores = self.gate(x)  # Shape: (batch_size, num_local, num_experts)
        gate_scores = F.softmax(gate_scores, dim=-1)  # Normalize over experts

        # Select top-k experts per local representation
        topk_values, topk_indices = torch.topk(gate_scores, self.top_k, dim=-1)  # Shape: (batch_size, num_local, top_k)
        topk_mask = torch.zeros_like(gate_scores).scatter_(-1, topk_indices, 1.0)  # Convert to one-hot mask

        # Normalize the selected scores
        gate_scores = gate_scores * topk_mask
        gate_scores = gate_scores / (gate_scores.sum(dim=-1, keepdim=True) + 1e-9)  # Prevent division by zero

        # **Step 2: Compute Expert Outputs**
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)  # Shape: (batch, num_local, feature_dim, num_experts)

        # Weighted sum over the selected experts
        output = torch.sum(gate_scores.unsqueeze(-2) * expert_outputs, dim=-1)  # Shape: (batch, num_local, feature_dim)

        # **Step 3: Local Representation Selection (Activating Only 3 of 16)**
        local_scores = self.local_gate(x).squeeze(-1)  # Shape: (batch_size, num_local)

        # Select the **top active_k** local representations (3 out of 16)
        top_local_values, top_local_indices = torch.topk(local_scores, self.active_k, dim=-1)
        local_mask = torch.zeros_like(local_scores).scatter_(-1, top_local_indices, 1.0).unsqueeze(-1)  # Shape: (batch, num_local, 1)

        # Apply the mask: **only 3 out of 16 will remain nonzero
        output = output * local_mask  # Shape remains (batch, num_local, feature_dim)

        return output, gate_scores  # Returning gate_scores

class MoE_hard_final(nn.Module):
    def __init__(self, input_dim, num_experts, expert_dim, top_k=2):
        super(MoE_hard_final, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k  # Number of experts selected per input

        # Expert Networks (each expert is a separate MLP)
        # self.experts = nn.ModuleList([nn.Linear(input_dim, expert_dim) for _ in range(num_experts)])
        self.experts = nn.ModuleList([
    nn.Sequential(
        nn.Linear(input_dim, expert_dim),
        nn.ReLU(),  # Add ReLU activation
        nn.Linear(expert_dim, expert_dim)  # Optional: Second layer
    ) for _ in range(num_experts)])
        

        # Gating Network - Determines which experts to use
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        """
        x shape: (batch_size, num_local_representations, feature_dim)
        """

        batch_size, num_local, feature_dim = x.shape  # Preserve batch and local structure

        # Compute gating scores for each local representation
        gate_scores = self.gate(x)  # (batch_size, num_local, num_experts)
        gate_scores = F.softmax(gate_scores, dim=-1)  # Normalize to sum to 1

        # **Adaptive Expert Selection (Top-K Hard Selection)**
        # Select top-k experts for each local representation
        topk_values, topk_indices = torch.topk(gate_scores, self.top_k, dim=-1)  # (batch, num_local, top_k)
        topk_mask = torch.zeros_like(gate_scores).scatter_(-1, topk_indices, 1.0)  # (batch, num_local, num_experts)

        # Normalize selected scores
        gate_scores = gate_scores * topk_mask
        gate_scores = gate_scores / gate_scores.sum(dim=-1, keepdim=True)  # Re-normalize selected experts

        # Process input through experts
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)  # (batch, num_local, feature_dim, num_experts)

        # Weighted sum of expert outputs
        output = torch.sum(gate_scores.unsqueeze(-2) * expert_outputs, dim=-1)  # (batch, num_local, feature_dim)

        return output, gate_scores  # Return expert usage scores for regularization

class MoE(nn.Module):
    def __init__(self, input_dim, num_experts, expert_dim):
        super(MoE, self).__init__()
        self.num_experts = num_experts

        # Expert networks (each expert learns different feature transformations)
        self.experts = nn.ModuleList([nn.Linear(input_dim, expert_dim) for _ in range(num_experts)])

        # Gating network (performs expert selection dynamically)
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        """
        x shape: (batch_size, num_local_representations, feature_dim)
        """

        batch_size, num_local, feature_dim = x.shape  # Keep the structure intact

        # Compute softmax gating scores
        gate_scores = F.softmax(self.gate(x), dim=-1)  # (batch_size, num_local, num_experts)

        # Process input through all experts (independent per modality)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)  # (batch, num_local, feature_dim, num_experts)

        # Weighted sum of expert outputs based on gating scores
        output = torch.sum(gate_scores.unsqueeze(-2) * expert_outputs, dim=-1)  # (batch_size, num_local, feature_dim)

        return output, gate_scores  # Return gate_scores for analysis

class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_r=0., use_relu=True):
        super(MLP, self).__init__()

        self.fc = FC_a(in_size, mid_size, dropout_r=dropout_r, use_relu=use_relu)
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        return self.linear(self.fc(x))

class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps

        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class FFN(nn.Module):
    def __init__(self, HIDDEN_SIZE,FF_SIZE,DROPOUT_R):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=HIDDEN_SIZE,
            mid_size=FF_SIZE,
            out_size=HIDDEN_SIZE,
            dropout_r=DROPOUT_R,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)

class MHAtt(nn.Module):
    def __init__(self, HIDDEN_SIZE,DROPOUT_R,MULTI_HEAD,HIDDEN_SIZE_HEAD):
        super(MHAtt, self).__init__()
        self.MULTI_HEAD = MULTI_HEAD
        self.HIDDEN_SIZE_HEAD = HIDDEN_SIZE_HEAD
        self.HIDDEN_SIZE = HIDDEN_SIZE
        self.DROPOUT_R = DROPOUT_R

        self.linear_v = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.linear_k = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.linear_q = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.linear_merge = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)

        self.dropout = nn.Dropout(DROPOUT_R)

    def forward(self, v, k, q):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.MULTI_HEAD,
            self.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.MULTI_HEAD,
            self.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.MULTI_HEAD,
            self.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        atted,map = self.att(v, k, q)

        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.HIDDEN_SIZE
        )

        atted = atted.squeeze()
        atted = self.linear_merge(atted)

        return atted,map

    def att(self, value, key, query):
        d_k = query.size(-1)
        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        # att_map = F.softmax(scores, dim=-1)
        att_map = F.sigmoid(scores)
        map=att_map

        return torch.matmul(att_map, value),map

class MHAtt_9(nn.Module):
    def __init__(self, HIDDEN_SIZE,DROPOUT_R,MULTI_HEAD,HIDDEN_SIZE_HEAD):
        super(MHAtt_9, self).__init__()
        self.MULTI_HEAD = MULTI_HEAD
        self.HIDDEN_SIZE_HEAD = HIDDEN_SIZE_HEAD
        self.HIDDEN_SIZE = HIDDEN_SIZE
        self.DROPOUT_R = DROPOUT_R

        self.linear_v = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.linear_k = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.linear_q = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.linear_merge = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)

        self.dropout = nn.Dropout(DROPOUT_R)

    def forward(self, v, k, q):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.MULTI_HEAD,
            self.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.MULTI_HEAD,
            self.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.MULTI_HEAD,
            self.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        atted,map = self.att(v, k, q)      #TODO: Modification


        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.HIDDEN_SIZE
        )

        atted = atted.squeeze()
        atted = self.linear_merge(atted)

        return atted,map

    def att(self, value, key, query):

        d_k = query.size(-1)
        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)
        att_map = F.softmax(scores, dim=-1)
        map=att_map

        return torch.matmul(att_map, value),map

class GA(nn.Module):
    def __init__(self, HIDDEN_SIZE,DROPOUT_R,MULTI_HEAD,HIDDEN_SIZE_HEAD):
        super(GA, self).__init__()

        self.mhatt1 = MHAtt(HIDDEN_SIZE,DROPOUT_R,MULTI_HEAD,HIDDEN_SIZE_HEAD)
        self.ffn = FFN(HIDDEN_SIZE,HIDDEN_SIZE*2,DROPOUT_R)

        self.dropout1 = nn.Dropout(DROPOUT_R)
        self.norm1 = LayerNorm(HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(DROPOUT_R)
        self.norm2 = LayerNorm(HIDDEN_SIZE)

    def forward(self, x, y):
        x = x.squeeze()
        atted,map=self.mhatt1(y, y, x)
        x = self.norm1(x + self.dropout1(atted))
        x = self.norm2(x + self.dropout2(
            self.ffn(x)
        ))

        return x,map

class GA_9(nn.Module):
    def __init__(self, HIDDEN_SIZE,DROPOUT_R,MULTI_HEAD,HIDDEN_SIZE_HEAD):
        super(GA_9, self).__init__()

        self.mhatt1 = MHAtt_9(HIDDEN_SIZE,DROPOUT_R,MULTI_HEAD,HIDDEN_SIZE_HEAD)
        self.ffn = FFN(HIDDEN_SIZE,HIDDEN_SIZE*2,DROPOUT_R)

        self.dropout1 = nn.Dropout(DROPOUT_R)
        self.norm1 = LayerNorm(HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(DROPOUT_R)
        self.norm2 = LayerNorm(HIDDEN_SIZE)

    def forward(self, x, y):
        x = x.squeeze()
        atted,map=self.mhatt1(y, y, x)       #TODO: Modification
        x = self.norm1(x + self.dropout1(atted))
        x = self.norm2(x + self.dropout2(
            self.ffn(x)
        ))

        return x,map        