import functools
from typing import Any, NamedTuple, Callable
import os
import random
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

import pickle
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
 
 leaves, treedef = jax.tree_flatten(tree_struct)
 with open(os.path.join(ckpt_dir, "arrays.npy"), "rb") as f:
   flat_state = [np.load(f) for _ in leaves]

 return jax.tree_unflatten(treedef, flat_state)


def img_to_patch(x, patch_size, flatten_channels=False):
    B, H, W, C = x.shape
    x = jnp.reshape(x, (B, H//patch_size, patch_size, W//patch_size, patch_size, C))
    x = jnp.transpose(x, (0, 1, 3, 2, 4, 5))    # [B, H', W', p_H, p_W, C]
    x = jnp.reshape(x, (B, -1, *x.shape[3:]))   # [B, H'*W', p_H, p_W, C]
    if flatten_channels:
        x = jnp.reshape(x, (B, x.shape[1], -1)) # [B, H'*W', p_H*p_W*C]
    return x 


class PatchesAlt(hk.Module):
    def __init__(self, 
                 patch_size, 
                 batch, 
                 **kwargs):
        super().__init__(kwargs['name']+self.__class__.__name__)
        self.patch_size = patch_size
        self.projection_dim = kwargs['enc_projection_dim']
        self.builded = False
        # Assuming the image has three channels each patch would be
        # of size (patch_size, patch_size, 3).
        self.resize = functools.partial(jnp.reshape, newshape=(batch, -1, patch_size * patch_size * 3))
        self.projection = hk.Linear(self.projection_dim)

    def create(self, patches):
        self.num_patches = patches.shape[1]
        self.position_embedding = hk.Embed(self.num_patches, self.projection_dim, 
                                           w_init=hk.initializers.TruncatedNormal(stddev=0.02),
                                           #w_init=hk.initializers.VarianceScaling(distribution="uniform"),
                                           lookup_style="ARRAY_INDEX")

    def __call__(self, images):
        # Create patches from the input images
        patches = img_to_patch(
            images, 
            self.patch_size
        )
        if not self.builded:
            self.create(patches)
            self.builded = True

        patches = self.resize(patches)
        
        positions = np.arange(start=0, stop=self.num_patches, step=1)
        pos_embeddings = self.position_embedding(positions)

        pos_embeddings = jnp.tile(pos_embeddings, [patches.shape[0], 1, 1])  # (B, num_patches, projection_dim)

        # Embed the patches.
        patch_embeddings = (
            self.projection(patches) + pos_embeddings
        )  # (B, num_patches, projection_dim)

        return patch_embeddings


class Patches(hk.Module):
    def __init__(self, 
                 patch_size, 
                 batch, 
                 **kwargs):
        super().__init__(kwargs['name']+self.__class__.__name__)
        self.patch_size = patch_size

        # Assuming the image has three channels each patch would be
        # of size (patch_size, patch_size, 3).
        self.resize = functools.partial(jnp.reshape, newshape=(batch, -1, patch_size * patch_size * 3))

    def __call__(self, images):
        # Create patches from the input images
        patches = img_to_patch(
            images, 
            self.patch_size
        )

        # Reshape the patches to (batch, num_patches, patch_area) and return it.
        patches = self.resize(patches)
        return patches

    def show_patched_image(self, images, patches):
        # This is a utility function which accepts a batch of images and its
        # corresponding patches and help visualize one image and its patches
        # side by side.
        idx = np.random.choice(patches.shape[0])
        
        # for i, patch in enumerate(patches[idx]):
        patch_img = np.reshape(patches[idx], (-1, self.patch_size, self.patch_size, 3))
               

        # Return the index chosen to validate it outside the method.
        return idx


    # taken from https://stackoverflow.com/a/58082878/10319735
def reconstruct_from_patch(patch, patch_size):
    # This utility function takes patches from a *single* image and
    # reconstructs it back into the image. This is useful for the train
    # monitor callback.
    num_patches = patch.shape[0]
    n = int(np.sqrt(num_patches))
    patch = jnp.reshape(patch, (num_patches, patch_size, patch_size, 3))
    # rows = jnp.split(patch, n, axis=0)
    # rows = [jnp.concatenate(jnp.split(x, x.shape[0]), axis=1) for x in rows]
    # reconstructed = jnp.concatenate(rows, axis=2)
    # return reconstructed.squeeze(0)
    samples = jnp.split(patch, n, axis=0)
    samples = jnp.concatenate(samples, axis=1)
    samples = jnp.split(samples, n, axis=0)
    samples = jnp.concatenate(samples, axis=2).squeeze(0)
    return samples
    

def create_patchtest_fn(fn, *args):
   def fwd_pass(input):
        model = fn(*args)
        return model(input)
   return hk.without_apply_rng(hk.transform(fwd_pass))


class PatchEncoder(hk.Module):
    def __init__(
        self,
        patch_size,
        enc_projection_dim,
        mask_proportion,
        downstream=False,
        **kwargs,
    ):
        super().__init__(name=kwargs['name']+self.__class__.__name__)
        self.patch_size = patch_size
        self.projection_dim = enc_projection_dim
        self.mask_proportion = mask_proportion
        self.downstream = downstream

        # This is a trainable mask token initialized randomly from a normal
        # distribution.
        self.mask_token = hk.get_parameter(name='masking', 
                                     shape=(1, patch_size * patch_size * 3), 
                                     dtype=np.float32, 
                                     init=hk.initializers.TruncatedNormal(stddev=0.02)
                                     #init=hk.initializers.VarianceScaling(distribution="normal")
                                     )
        
        self.builded = False

    def build(self, input_shape):
        (_, self.num_patches, self.patch_area) = input_shape

        # Create the projection layer for the patches.
        self.projection = hk.Linear(self.projection_dim)

        # Create the positional embedding layer.
        # self.position_embedding = layers.Embedding(
        #     input_dim=self.num_patches, output_dim=self.projection_dim
        # )
        # self.position_embedding = hk.get_parameter(name='position_embedding', 
        #                              shape=(self.num_patches, self.projection_dim), 
        #                              dtype=np.float32, 
        #                              init=hk.initializers.VarianceScaling(distribution="uniform"))
        self.position_embedding = hk.Embed(self.num_patches, self.projection_dim, 
                                           w_init=hk.initializers.TruncatedNormal(stddev=0.02),
                                           #w_init=hk.initializers.VarianceScaling(distribution="uniform"),
                                           lookup_style="ARRAY_INDEX")


        # Number of patches that will be masked.
        self.num_mask = int(self.mask_proportion * self.num_patches)

    def __call__(self, patches):
        if not self.builded:
           self.build(patches.shape)
           self.builded = True
        # Get the positional embeddings.
        batch_size = patches.shape[0]
        positions = np.arange(start=0, stop=self.num_patches, step=1)
        # pos_embeddings = jnp.matmul(jnp.expand_dims(positions, axis=0), self.position_embedding) 
        # pos_embeddings = self.position_embedding.at[:,0].set(positions)
        pos_embeddings = self.position_embedding(positions)

        pos_embeddings = jnp.tile(
            pos_embeddings, [batch_size, 1, 1]
        )  # (B, num_patches, projection_dim)

        # Embed the patches.
        patch_embeddings = (
            self.projection(patches) + pos_embeddings
        )  # (B, num_patches, projection_dim)

        if self.downstream:
            return patch_embeddings
        else:
            mask_indices, unmask_indices = self.get_random_indices(batch_size)
            # The encoder input is the unmasked patch embeddings. Here we gather
            # all the patches that should be unmasked.
            # unmasked_embeddings = jnp.gather(
            #     patch_embeddings, unmask_indices, axis=1, batch_dims=1
            # )  # (B, unmask_numbers, projection_dim)
            unmasked_embeddings = np.take_along_axis(
                patch_embeddings, jnp.expand_dims(unmask_indices, axis=-1), axis=1
            )

            # Get the unmasked and masked position embeddings. We will need them
            # for the decoder.
            # unmasked_positions = jnp.gather(
            #     pos_embeddings, unmask_indices, axis=1, batch_dims=1
            # )  # (B, unmask_numbers, projection_dim)
            unmasked_positions = np.take_along_axis(
                pos_embeddings, jnp.expand_dims(unmask_indices, axis=-1), axis=1
            )
            # masked_positions = jnp.gather(
            #     pos_embeddings, mask_indices, axis=1, batch_dims=1
            # )  # (B, mask_numbers, projection_dim)
            masked_positions = np.take_along_axis(
                pos_embeddings, jnp.expand_dims(mask_indices, axis=-1), axis=1
            )

            # Repeat the mask token number of mask times.
            # Mask tokens replace the masks of the image.
            mask_tokens = jnp.repeat(self.mask_token, repeats=self.num_mask, axis=0)

            mask_tokens = jnp.repeat(
                jnp.expand_dims(mask_tokens, axis=0), repeats=batch_size, axis=0
            )

            # Get the masked embeddings for the tokens.
            masked_embeddings = self.projection(mask_tokens) + masked_positions


            # loss = jnp.sum(masked_embeddings)
            return (
                unmasked_embeddings,  # Input to the encoder.
                # masked_embeddings,  # First part of input to the decoder.
                unmasked_positions,  # Added to the encoder outputs.
                # mask_indices,  # The indices that were masked.
                # unmask_indices,  # The indices that were unmaksed.
            )

    def get_random_indices(self, batch_size):
        # Create random indices from a uniform distribution and then split
        # it into mask and unmask indices.
        # # DETERMINISTIC
        # umsk_ind = [i for i in range(self.num_patches) if i % 8 < 4 and i < 32]
        umsk_ind = []
        while len(umsk_ind) != self.num_patches // 2:
            iddx = random.randint(0, self.num_patches - 1)
            if iddx not in umsk_ind:
                umsk_ind.append(iddx)
        msk_ind = [i for i in range(self.num_patches) if i not in umsk_ind]
        rand_indices = np.array(msk_ind + umsk_ind, dtype=np.longlong)
        rand_indices = np.tile(np.expand_dims(rand_indices, axis=0), (batch_size, 1))
        # # RANDOM
        # rand_indices = np.argsort(
        #     np.random.uniform(size=(batch_size, self.num_patches)), axis=-1
        # )
        # # #####-------#####
        mask_indices = rand_indices[:, : self.num_mask]
        unmask_indices = rand_indices[:, self.num_mask :]
        return mask_indices, unmask_indices


def generate_masked_image(patches, unmask_indices):
    # Choose a random patch and it corresponding unmask index.
    idx = np.random.choice(patches.shape[0])
    patch = patches[idx]
    unmask_index = unmask_indices[idx]

    # Build a numpy array of same shape as patch.
    new_patch = np.zeros_like(patch)

    # Iterate of the new_patch and plug the unmasked patches.
    count = 0
    for i in range(unmask_index.shape[0]):
        new_patch[unmask_index[i]] = patch[unmask_index[i]]
    return new_patch, idx