import functools as ft
import pandas as pd
import numpy as np
import jax
import jax.nn as jnn
import jax.random as jr
import jax.numpy as jnp

from jax.scipy.special import logsumexp

import optax

import einops

import haiku as hk
import haiku.initializers as hki


class MLP(hk.Module):
    def __init__(self, hidden_units, dropout):
        super().__init__()
        self.hidden_units = hidden_units
        self.dropout = dropout

    def __call__(self, x, *, is_training: bool):
        dropout = self.dropout if is_training else 0.0
        w_init = hki.VarianceScaling()
        b_init = hki.Constant(0)
        for units in self.hidden_units:
            x = hk.Linear(units, w_init=w_init, b_init=b_init)(x)
            x = jnn.gelu(x, approximate=False)
            x = hk.dropout(hk.next_rng_key(), dropout, x)
        return x
    

class ViT(hk.Module):
    def __init__(self, 
                 num_patches, 
                 enc_projection_dim, 
                 enc_layers, 
                 enc_num_heads, 
                 num_classes,
                 transformer_units=[2048, 1024], 
                 mlp_head_units=(2048, 1024), 
                 dropout_rate=0.5,
                 **kwargs):
        super(ViT, self).__init__()
        self.num_patches = num_patches
        self.projection_dim = enc_projection_dim
        self.num_blocks = enc_layers
        self.num_heads = enc_num_heads
        self.transformer_units_1 = transformer_units[0]
        self.transformer_units_2 = transformer_units[1]
        self.mlp_head_units = mlp_head_units
        self.dropout = dropout_rate
        self.num_classes = num_classes
        self.norm = lambda: hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, eps=1e-6,
                                         scale_init=hki.Constant(1.0), offset_init=hki.Constant(0.0))

    def __call__(self, encoded_patches, labels, is_training: bool):
        dropout = self.dropout if is_training else 0.0

        for _ in range(self.num_blocks):
            x1 = self.norm()(encoded_patches)
            attention = hk.MultiHeadAttention(self.num_heads, self.projection_dim // self.num_heads,w_init_scale=1.0)(x1,x1,x1)
            x2 = attention + encoded_patches
            x3 = self.norm()(x2)
            x3 = MLP((self.transformer_units_1, self.transformer_units_2), self.dropout)(x3, is_training=is_training)

            encoded_patches = x3 + x2

        representation = self.norm()(encoded_patches)
        representation = einops.rearrange(representation, 'b h t -> b (h t)')
        representation = hk.dropout(hk.next_rng_key(), dropout, representation)

        features = MLP(self.mlp_head_units, self.dropout)(representation, is_training=is_training)

        logits = hk.Linear(self.num_classes)(features)
        loss = jnp.mean(optax_cross_entropy_loss(labels, logits, self.num_classes))

        return logits - logsumexp(logits, axis=1, keepdims=True), loss
    

def optax_cross_entropy_loss(labels, logits, num_classes):
    one_hot_actual = jax.nn.one_hot(labels, num_classes)
    return optax.softmax_cross_entropy(logits, one_hot_actual)

def focal_loss(labels, y_pred, ce, gamma, alpha):
    weight = labels * jnp.power(1 - y_pred, gamma)
    f_loss = alpha * (weight * ce)
    f_loss = jnp.sum(f_loss, axis=1)
    f_loss = jnp.mean(f_loss, axis=0)
    return f_loss

