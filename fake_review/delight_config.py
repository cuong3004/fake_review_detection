# %%
from ast import arg
from email.mime import base
import math
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# from fake_bert.delight_modules.delight_transformer_layer import DeLighTTransformerEncoderLayer

DEFAULT_WIDTH_MULTIPLIER = 2.0
DEFAULT_MIN_DEXTRA_LAYERS = 8
DEFAULT_MAX_DEXTRA_LAYERS = 16
DEFAULT_BASE_GROUPS = 16
MIN_ELEMENTS_PER_GROUP = 32
DEFAULT_FFN_RED_FACTOR = 4
DEFAULT_DROPOUT = 0.1
DEFAULT_STD_DROPOUT = 0.1
ADAPTIVE_SCALE_FACTOR = 2

# %%

def base_architecture(args):
    # DeLighT Embedding layer
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.delight_emb_map_dim = getattr(args, "delight_emb_map_dim", 384)
    args.delight_emb_out_dim = getattr(args, "delight_emb_out_dim", 384)
    # compute the max groups in GLT
    assert args.delight_emb_out_dim % MIN_ELEMENTS_PER_GROUP == 0, 'remainder({}, {}) should be equal to 0'.format(
        args.delight_emb_out_dim, MIN_ELEMENTS_PER_GROUP)
    max_groups = 2 ** math.ceil(math.log(args.delight_emb_out_dim // MIN_ELEMENTS_PER_GROUP, 2))
    # max_groups = 8

    args.delight_emb_max_groups = getattr(args, "delight_emb_max_groups", max_groups)
    args.delight_emb_dropout = getattr(args, "delight_emb_dropout", DEFAULT_DROPOUT)
    args.delight_emb_depth = getattr(args, "delight_emb_depth", DEFAULT_MIN_DEXTRA_LAYERS)
    args.delight_emb_width_mult = getattr(args, "delight_emb_width_mult", DEFAULT_WIDTH_MULTIPLIER)

    # Encoder arguments in DeLighT
    args.delight_enc_scaling = getattr(args, "delight_enc_scaling", 'block')
    args.delight_enc_layers = getattr(args, "delight_enc_layers", 4) # DEFAULT_MAX_DEXTRA_LAYERS
    args.delight_enc_min_depth = getattr(args, "delight_enc_min_depth", DEFAULT_MIN_DEXTRA_LAYERS)
    args.delight_enc_max_depth = getattr(args, "delight_enc_max_depth", DEFAULT_MAX_DEXTRA_LAYERS)
    args.delight_enc_width_mult = getattr(args, "delight_enc_width_mult", DEFAULT_WIDTH_MULTIPLIER)
    args.delight_enc_ffn_red = getattr(args, "delight_enc_ffn_red", DEFAULT_FFN_RED_FACTOR)
    args.delight_enc_max_groups = getattr(args, "delight_enc_max_groups", max_groups)

    # Decoder arguments in DeLighT
    # args.delight_dec_scaling = getattr(args, "delight_dec_scaling", 'block')
    # args.delight_dec_layers = getattr(args, "delight_dec_layers", DEFAULT_MAX_DEXTRA_LAYERS)
    # args.delight_dec_min_depth = getattr(args, "delight_dec_min_depth", DEFAULT_MIN_DEXTRA_LAYERS)
    # args.delight_dec_max_depth = getattr(args, "delight_dec_max_depth", DEFAULT_MAX_DEXTRA_LAYERS)
    # args.delight_dec_width_mult = getattr(args, "delight_dec_width_mult", DEFAULT_WIDTH_MULTIPLIER)
    # args.delight_dec_ffn_red = getattr(args, "delight_dec_ffn_red", DEFAULT_FFN_RED_FACTOR)
    # args.delight_dec_max_groups = getattr(args, "delight_dec_max_groups", max_groups)

    ## Others
    args.no_glt_shuffle = getattr(args, "no_glt_shuffle", False)
    args.glt_shuffle = not args.no_glt_shuffle
    args.define_iclr = getattr(args, "define_iclr", False)
    args.delight_dropout = getattr(args, "delight_dropout", DEFAULT_DROPOUT)

    # normalization and activation layers
    args.norm_type = getattr(args, "norm_type", 'ln')
    args.act_type = getattr(args, "act_type", 'swish')

    # ADAPTIVE INPUT AND OUTPUT PARAMS
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.adaptive_softmax_factor = getattr(args, 'adaptive_softmax_factor', 4)
    args.tie_adaptive_weights = getattr(args, 'tie_adaptive_weights', False)
    args.tie_adaptive_proj = getattr(args, 'tie_adaptive_proj', False)

    # Print  stats
    args.print_stats = getattr(args, "print_stats", False)
    args.src_len_ps = getattr(args, "src_len_ps", 20)
    args.tgt_len_ps = getattr(args, "tgt_len_ps", 20)

    # DROPOUTS
    args.attention_dropout = getattr(args, "attention_dropout", DEFAULT_DROPOUT)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.dropout = getattr(args, "dropout", DEFAULT_DROPOUT)
    args.delight_dropout = getattr(args, "delight_dropout", 0.0)
    args.pe_dropout = getattr(args, "pe_dropout", DEFAULT_DROPOUT)
    args.ffn_dropout = getattr(args, "ffn_dropout", DEFAULT_DROPOUT)

    # Other parameters
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)

    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)

# %%

args = argparse.Namespace()

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--delight_emb_width_mult', default=2.0, type=float)

args = parser.parse_args()

base_architecture(args)
print(args)

# print(args)

# layers = nn.ModuleList([])

# dextra_depths = np.linspace(start=args.delight_enc_min_depth,
#                                          stop=args.delight_enc_max_depth,
#                                          num=args.delight_enc_layers,
#                                          dtype=np.int32)

# depth_ratio = (args.delight_enc_max_depth * 1.0) / args.delight_enc_min_depth

# width_multipliers = np.linspace(start=args.delight_enc_width_mult,
#                             stop=args.delight_enc_width_mult + (depth_ratio - 1.0), # subtraction by 1 for max==min case
#                             num=args.delight_enc_layers,
#                             dtype=np.float32
#                             )

# embed_dim = 128

# # %%

# layers.extend(
#     [DeLighTTransformerEncoderLayer(args=args,
#                                     embed_dim=embed_dim,
#                                     width_multiplier=round(width_multipliers[idx], 3),
#                                     dextra_depth=layer_i)
#         for idx, layer_i in enumerate(dextra_depths)
#         ]
# )
# # %%

# nn.TransformerEncoder