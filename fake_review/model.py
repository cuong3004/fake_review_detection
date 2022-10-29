from typing import Optional, Tuple, Union
from transformers import BertForSequenceClassification
import torch 
from transformers.modeling_outputs import SequenceClassifierOutput
from operator import mod
from typing import List, Optional, Tuple, Union
from numpy import dtype
from transformers import BertConfig, BertForMaskedLM, BertModel
from transformers.models.bert.modeling_bert import BertEmbeddings, BertAttention, BertSelfAttention, BertLayer, BertEncoder, BertForSequenceClassification, BertSelfOutput
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
import torch
import torch.nn as nn
import numpy as np
from delight_modules.dextra_unit import DExTraUnit
import copy 

class TinyBertForSequenceClassification(BertForSequenceClassification):
    def forward(self, input_ids: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None, token_type_ids: Optional[torch.Tensor] = None, position_ids: Optional[torch.Tensor] = None, head_mask: Optional[torch.Tensor] = None, inputs_embeds: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None, output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        return super().forward(input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, labels, output_attentions, output_hidden_states, return_dict)

        
from turtle import forward


class DenyBertForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertDelightModel(config)
        

class DenyBertModel(BertModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        self.encoder = DenyBertEncoder(config)
        assert False

class DenyBertEncoder(BertEncoder):
    def __init__(self, config):
        super().__init__(config)

        assert config.encode_min_depth < config.encode_max_depth

        dextra_depths = np.linspace(start=config.encode_min_depth,
                                         stop=config.encode_max_depth,
                                         num=config.num_hidden_layers,
                                         dtype=np.int32)
        
        depth_ratio = (config.encode_max_depth * 1.0) / config.encode_min_depth
        

        width_multipliers = np.linspace(start=config.encode_width_mult,
                        stop=config.encode_width_mult + (depth_ratio - 1.0), # subtraction by 1 for max==min case
                        num=config.num_hidden_layers,
                        dtype=np.float32
                        )
        
        self.layer.extend(
                [DenyBertLayer(config=config,
                                    width_multiplier=round(width_multipliers[idx], 3),
                                    dextra_depth=layer_i)
                 for idx, layer_i in enumerate(dextra_depths)
                 ]
            )
        # assert False
    
class DenyBertLayer(BertLayer):
    def __init__(self, config, width_multiplier, dextra_depth):
        super().__init__(config)
        self.attention = DenyBertAttention(config, width_multiplier, dextra_depth)

class DenyBertAttention(BertAttention):
    def __init__(self, config, width_multiplier, dextra_depth, position_embedding_type=None, dextra_proj=2):
        super().__init__(config, position_embedding_type)

        self.embed_dim = config.hidden_size
        assert self.embed_dim % dextra_proj == 0

        self.proj_dim = self.embed_dim // dextra_proj

        self.dextra_layer = DExTraUnit(in_features=self.embed_dim,
                                       in_proj_features=self.proj_dim,
                                       out_features=self.proj_dim,
                                       width_multiplier=width_multiplier,
                                       dextra_depth=dextra_depth,
                                       dextra_dropout=0.1,
                                       max_glt_groups=4,
                                       act_type="gelu",
                                       use_bias=True,
                                       norm_type="ln",
                                       glt_shuffle=False,
                                       is_iclr_version=False
                                       )
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.FloatTensor] = None, head_mask: Optional[torch.FloatTensor] = None, encoder_hidden_states: Optional[torch.FloatTensor] = None, encoder_attention_mask: Optional[torch.FloatTensor] = None, past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None, output_attentions: Optional[bool] = False) -> Tuple[torch.Tensor]:
        hidden_states = self.dextra_layer(hidden_states)
        return super().forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, output_attentions)
    


# class BertDeLightEmbeddings(BertEmbeddings):
#     def __init__(self, config):
#         super().__init__(config)
#         self.word_embeddings = DExTraEmb(
#             config.args,
#             self.word_embeddings,
#         )

class BertDelightModel(BertModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)

        # self.embeddings = BertDeLightEmbeddings(config)
        self.encoder = BertDeLightEncoder(config)
    
class BertDeLightSelfOutput(BertSelfOutput):
    def __init__(self, config):
        super().__init__(config)
        # print(config.hidden_size/2)
        self.dense = nn.Linear(int(config.hidden_size/2), config.hidden_size)

class BertDelightSelfAttention(BertSelfAttention):
    def __init__(self, config, width_multiplier, dextra_depth, position_embedding_type=None, dextra_proj=2):
        config_new = copy.copy(config)
        config_new.hidden_size = int(config_new.hidden_size/2)
        super().__init__(config_new, position_embedding_type)
        self.embed_dim = config.args.delight_emb_out_dim
        assert self.embed_dim % dextra_proj == 0
        
        self.proj_dim = self.embed_dim // dextra_proj

        # print(config.args.delight_enc_max_groups)

        self.dextra_layer = DExTraUnit(in_features=self.embed_dim,
                                       in_proj_features=self.proj_dim,
                                       out_features=self.proj_dim,
                                       width_multiplier=width_multiplier,
                                       dextra_depth=dextra_depth,
                                       dextra_dropout=config.args.delight_dropout,
                                       max_glt_groups=config.args.delight_enc_max_groups,
                                       act_type=config.args.act_type,
                                       use_bias=True,
                                       norm_type=config.args.norm_type,
                                       glt_shuffle=config.args.glt_shuffle,
                                       is_iclr_version=config.args.define_iclr
                                       )
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.FloatTensor] = None, head_mask: Optional[torch.FloatTensor] = None, encoder_hidden_states: Optional[torch.FloatTensor] = None, encoder_attention_mask: Optional[torch.FloatTensor] = None, past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None, output_attentions: Optional[bool] = False) -> Tuple[torch.Tensor]:
        hidden_states = self.dextra_layer(hidden_states)
        return super().forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, output_attentions)
        

class BertDeLightAttention(BertAttention):
    def __init__(self, config, width_multiplier, dextra_depth, position_embedding_type=None):
        super().__init__(config, position_embedding_type)
        
        self.self = BertDelightSelfAttention(config, width_multiplier, dextra_depth)
        self.output = BertDeLightSelfOutput(config)
                                       
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.FloatTensor] = None, head_mask: Optional[torch.FloatTensor] = None, encoder_hidden_states: Optional[torch.FloatTensor] = None, encoder_attention_mask: Optional[torch.FloatTensor] = None, past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None, output_attentions: Optional[bool] = False) -> Tuple[torch.Tensor]:
        # print("before ",hidden_states.shape)
        
        # print("affter",hidden_states.shape)
        # assert False
        return super().forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, output_attentions)
    

class BertDeLightLayer(BertLayer):
    def __init__(self, config, width_multiplier, dextra_depth):
        super().__init__(config)
        self.attention = BertDeLightAttention(config, width_multiplier, dextra_depth)

class BertDeLightEncoder(BertEncoder):
    def __init__(self, config):
        super().__init__(config)

        self.layer = nn.ModuleList([])
        if config.args.delight_enc_scaling == 'block' and (config.args.delight_enc_min_depth == config.args.delight_enc_max_depth):
            config.args.delight_enc_scaling = 'uniform'


        if config.args.delight_enc_scaling == 'uniform':
            assert config.args.delight_enc_min_depth == config.args.delight_enc_max_depth
            self.layer.extend(
                [BertDeLightLayer(config=config,
                                width_multiplier=config.args.delight_enc_width_mult,
                                dextra_depth=config.args.delight_enc_min_depth)
                 for _ in range(config.args.define_enc_layers)]
            )
        else:
            assert config.args.delight_enc_min_depth < config.args.delight_enc_max_depth

            dextra_depths = np.linspace(start=config.args.delight_enc_min_depth,
                                         stop=config.args.delight_enc_max_depth,
                                         num=config.args.delight_enc_layers,
                                         dtype=np.int32)

            depth_ratio = (config.args.delight_enc_max_depth * 1.0) / config.args.delight_enc_min_depth

            # print(depth_ratio)

            width_multipliers = np.linspace(start=config.args.delight_enc_width_mult,
                                      stop=config.args.delight_enc_width_mult + (depth_ratio - 1.0), # subtraction by 1 for max==min case
                                      num=config.args.delight_enc_layers,
                                      dtype=np.float32
                                      )
            # print(width_multipliers)

            self.layer.extend(
                [BertDeLightLayer(config=config,
                                    width_multiplier=round(width_multipliers[idx], 3),
                                    dextra_depth=layer_i)
                 for idx, layer_i in enumerate(dextra_depths)
                 ]
            )
