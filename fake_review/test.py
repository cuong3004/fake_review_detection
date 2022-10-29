from operator import mod
from typing import List, Optional, Tuple, Union
from numpy import dtype
from transformers import BertConfig, BertForMaskedLM, BertModel
from transformers.models.bert.modeling_bert import BertEmbeddings, BertAttention, BertLayer, BertEncoder, BertForSequenceClassification
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
import torch
import torch.nn as nn
import numpy as np
from delight_modules.dextra_unit import DExTraUnit
from model import DenyBertForSequenceClassification


config = BertConfig.from_json_file("config_delight.json")
from delight_config import args 
config.args = args
model = DenyBertForSequenceClassification(config=config)
x = torch.ones([2, 128], dtype=torch.long)
attention_mask = torch.zeros([2, 128], dtype=torch.long)
out = model(x, attention_mask,
    output_hidden_states=True,
    output_attentions=True,
        )
out
