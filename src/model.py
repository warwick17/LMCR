from typing import Type
import jax
import haiku as hk
import jax.numpy as jnp
# import numpy as np
from typing import Optional
from operator import getitem
import numpy as np
import pickle 
import optax
import os


def save(ckpt_dir: str, state) -> None:
    with open(os.path.join(ckpt_dir, "arrays.npy"), "wb") as f:
        for x in jax.tree_leaves(state):
            np.save(f, x, allow_pickle=False)

    tree_struct = jax.tree_map(lambda t: 0, state)
    with open(os.path.join(ckpt_dir, "tree.pkl"), "wb") as f:
        pickle.dump(tree_struct, f)


def load(ckpt_dir):
    with open(os.path.join(ckpt_dir, "tree.pkl"), "rb") as f:
        tree_struct = pickle.load(f)

    leaves, treedef = jax.tree_util.tree_flatten(tree_struct)
    with open(os.path.join(ckpt_dir, "arrays.npy"), "rb") as f:
        flat_state = [np.load(f) for _ in leaves]

    return jax.tree_util.tree_unflatten(treedef, flat_state)


# class SelfAttention(hk.MultiHeadAttention):
#     """Self attention with a causal mask applied."""

#     def __call__(
#             self,
#             query: jnp.ndarray,
#             key: Optional[jnp.ndarray] = None,
#             value: Optional[jnp.ndarray] = None,
#             mask: Optional[jnp.ndarray] = None,
#     ) -> jnp.ndarray:
#         key = key if key is not None else query
#         value = value if value is not None else query

#         seq_len = query.shape[1]
#         causal_mask = np.tril(np.ones((seq_len, seq_len)))
#         mask = mask * causal_mask if mask is not None else causal_mask

#         return super().__call__(query, key, value, mask)


class MLP(hk.Module):
    def __init__(self, hidden_units, name=None):
        super().__init__(name=name)
        self.hidden_units = hidden_units

    def __call__(self, x, is_training):
        #keys = jax.random.split(key, num=len(self.hidden_units))
        for out in self.hidden_units:
            x = hk.Linear(out)(x)
            x = jax.nn.gelu(x)
            if is_training:
                x = hk.dropout(hk.next_rng_key(), 0.5, x)
        return x


def cross_entropy_loss(labels, logits, weight, num_classes):
    logits = jax.nn.log_softmax(logits)
    loss = jax.vmap(getitem)(logits, labels)
    loss = loss*weight
    loss = -loss.sum()/weight.sum()
    return loss


def optax_cross_entropy_loss(labels, logits, num_classes):
    one_hot_actual = jax.nn.one_hot(labels, num_classes)
    return optax.softmax_cross_entropy(logits, one_hot_actual)


def label_smoothing(one_hot_targets, label_smoothing):
    on_value = 1.0 - label_smoothing
    num_classes = one_hot_targets.shape[-1]
    off_value = label_smoothing / num_classes
    one_hot_targets = one_hot_targets * on_value + off_value
    return one_hot_targets


def smoothed_loss(labels, logits, num_classes, smoothing_value=0.1):
    one_hot_targets = jax.nn.one_hot(labels, num_classes)
    soft_targets = label_smoothing(one_hot_targets, smoothing_value)
    loss = -jnp.sum(soft_targets * jax.nn.log_softmax(logits), axis=-1)
    return loss


def focal_loss(gamma=2., alpha=4., num_classes=None):

    gamma = float(gamma)
    alpha = float(alpha)

    def focal_loss_fixed(labels, logits):
        """Focal loss for multi-classification
        FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
        Notice: y_pred is probability after softmax
        gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
        d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
        """
        epsilon = 1.e-9
        logits = logits + epsilon
        one_hot_labels = jax.nn.one_hot(labels, num_classes)

        one_hot_labels = label_smoothing(one_hot_labels, 0.1)

        cross_enpropy = -one_hot_labels * jax.nn.log_softmax(logits)
        weight = jax.lax.pow((1. - jax.nn.softmax(logits)), gamma)
        fl = alpha * weight * cross_enpropy
        reduced_fl = jnp.max(fl, axis=-1)
        return jnp.mean(reduced_fl)
    return focal_loss_fixed  


class ActionHeadClassification(hk.Module):
    def __init__(self, 
                 dropout_rate,
                 num_classes,
                 norm_eps, 
                 hidden_dim=1024, 
                 name=None, 
                 **kwargs
                 ):
        super().__init__(name=name)
        self.num_classes = num_classes
        # self.dropout_rate = dropout_rate
        self.hidden_dim = hidden_dim
        self.eps = norm_eps
        
    def __call__(self, x, labels, weights, is_training=False):
        # L, P = feat.shape
        # x = feat.mean(axis=1)
        # L, J, P = feat.shape
        # x = feat.reshape(-1, 2, J, P)
        # x = np.mean(feat, axis=1)
        # x = hk.Linear(64)(feat)
        # x = jax.nn.relu(x)
        # # x = x.reshape(-1, 2, J, 17, 3)
        # # x = feat.reshape(feat.shape[0], -1)
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, eps=self.eps)(x)
        x = hk.Flatten()(x)
        if is_training: 
            x = hk.dropout(hk.next_rng_key(), 0.5, x)
        x = MLP(hidden_units=[4096, 2048])(x, is_training)
        # x = hk.Linear(2048)(x)
        # x = jax.nn.relu(x)
        # x = jnp.transpose(x, (0, 1, 3, 4, 2))
        # x = x.mean(axis=-1)
        # x = x.reshape(*x.shape[:2], -1)
        # x = x.mean(axis=1)
        # x = hk.Linear(self.hidden_dim, name='ClassLin2')(x)
        # x = hk.BatchNorm(create_offset=True, 
        #                  create_scale=True, 
        #                  decay_rate=0.9)(x, is_training=training)
        # x = jax.nn.relu(x)
        cls_logits = hk.Linear(self.num_classes, name='ClassLin3')(x)
        # classifier_loss = focal_loss(num_classes=self.num_classes)(labels, cls_logits)
        # classifier_loss = jnp.mean(cross_entropy_loss(labels, cls_logits, weights, None))
        # classifier_loss = jnp.mean(smoothed_loss(labels, cls_logits, self.num_classes, 0.))
        classifier_loss = jnp.mean(optax_cross_entropy_loss(labels, cls_logits, self.num_classes))
        return cls_logits, classifier_loss


class Encoder(hk.Module):
    def __init__(self, enc_num_heads, 
                 enc_layers, 
                 enc_projection_dim, 
                 enc_transformer_units, 
                 norm_eps, 
                 name=None,
                 **kwargs):
        super().__init__(name=name+self.__class__.__name__)
        self.num_heads = enc_num_heads
        self.num_layers = enc_layers
        self.proj_dim = enc_projection_dim
        self.transf_units = enc_transformer_units
        self.eps = norm_eps
        
    def __call__(self, input, is_training):
        for _ in range(self.num_layers):
            # Layer normalization 1.
            x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, eps=self.eps)(input)
            # Create a multi-head attention layer.
            attention_output = hk.MultiHeadAttention(num_heads=self.num_heads,
                                                     key_size=self.proj_dim,
                                                     model_size=self.proj_dim,
                                                     w_init=hk.initializers.TruncatedNormal(stddev=0.02),
                                                     #w_init=hk.initializers.RandomNormal(),
                                                     name='EncMha')(x, x, x)
            # attention_output = SelfAttention(num_heads=self.num_heads,
            #                                  key_size=self.proj_dim,
            #                                  model_size=self.proj_dim,
            #                                  w_init=hk.initializers.TruncatedNormal(stddev=0.02),
            #                                  name='EncMha')(x,x)

            # Skip connection 1.
            x = attention_output + input
            # Layer normalization 2.
            x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, eps=self.eps)(x)
            # MLP.
            x_m = MLP(hidden_units=self.transf_units)(x, is_training)
            # Skip connection 2.
            input = x + x_m

        outputs = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, eps=self.eps)(input)
        return outputs
    

class Decoder(hk.Module):
    def __init__(self, dec_layers, 
                 dec_num_heads, 
                 image_size, 
                 dec_projection_dim, 
                 dec_transformer_units, 
                 norm_eps, 
                 name=None,
                 **kwargs
                 ):
        super().__init__(name=name+self.__class__.__name__)
        self.num_heads = dec_num_heads
        self.num_layers = dec_layers
        self.im_sz = image_size
        self.proj_dim = dec_projection_dim
        self.transf_units = dec_transformer_units
        self.eps = norm_eps
        self.logits_out = self.im_sz * self.im_sz * 3
        
    def __call__(self, input, is_training=True):
        x = hk.Linear(self.proj_dim)(input)

        for _ in range(self.num_layers):
            # Layer normalization 1.
            n = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, eps=self.eps)(x)
            # Create a multi-head attention layer.
            attention_output = hk.MultiHeadAttention(num_heads=self.num_heads,
                                                     key_size=self.proj_dim,
                                                     model_size=self.proj_dim,
                                                     #w_init=hk.initializers.RandomNormal(), 
                                                     w_init=hk.initializers.TruncatedNormal(stddev=0.02),
                                                     name='DecMha')(n, n, n)
            # Skip connection 1.
            x = attention_output + x
            # Layer normalization 2.
            x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, eps=self.eps)(x)
            # MLP.
            x_m = MLP(hidden_units=self.transf_units, name='DecMlp')(x, is_training)
            # Skip connection 2.
            x = x + x_m

        x = x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, eps=self.eps)(x)
        x = hk.Flatten(name='DecFlat')(x)
        logits = hk.Linear(self.logits_out)(x)
        img = jax.nn.tanh(logits)
        img = jnp.reshape(img, (-1, self.im_sz, self.im_sz, 3))

        return img