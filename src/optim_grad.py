import haiku as hk
import jax
import optax
import jax.numpy as jnp
import functools

from patching import Patches, PatchEncoder, create_patchtest_fn, PatchesAlt
from models import Encoder, Decoder, MLP, ActionHeadClassification
# from model_vit import VitBase
from model_vit_kg import ViT


def create_forward_fn(fns, args):
    def fwd_pass(input, is_training=True):
        # Patch input images
        data, labels, weights = input
        patches = fns['Patches'](**args)(data)

        # Encode the patches.
        (
            unmasked_embeddings,
            masked_embeddings,
            unmasked_positions,
            mask_indices,
            unmask_indices,
        ) = fns['PatchEncoder'](**args)(patches)

        # Pass the unmaksed patche to the encoder.
        encoder_outputs = fns['Encoder'](**args)(unmasked_embeddings)

        # Create the decoder inputs.
        # encoder_outputs = encoder_outputs + unmasked_positions

        logits, classifier_loss = fns['ActionHeadClassification'](**args)(encoder_outputs, labels, weights, is_training)

        return classifier_loss, dict({
            'classif_loss': classifier_loss,
            'logits': logits
        })

        # total_loss = jnp.ones((1), dtype=jnp.float32)
        # return total_loss, dict({
        #     'encoder_outputs': encoder_outputs,
        # })

    # return hk.transform(fwd_pass)
    return hk.transform_with_state(fwd_pass)


class MaskedAE:
    def __init__(self, arguments):
        self.args = arguments

    def init_params(self, rng, dummy_input):
        key, sub = jax.random.split(rng, num=2)

        self.forward = create_forward_fn(dict({
            'Encoder': Encoder,
            'Decoder': Decoder,
            'PatchEncoder': PatchEncoder,
            'Patches': Patches,
            'ActionHeadClassification': ActionHeadClassification
        }), self.args)
        params, st = self.forward.init(key, dummy_input)

        lr_logic = optax.sgdr_schedule(
            [
            {"init_value":0.0003, "peak_value":0.0004, "decay_steps":10000, "warmup_steps":1000, "end_value":0.0002},
            {"init_value":0.0002, "peak_value":0.0003, "decay_steps":10000, "warmup_steps":1000, "end_value":0.0001},
            {"init_value":0.0001, "peak_value":0.0002, "decay_steps":10000, "warmup_steps":1000, "end_value":0.00005},
            ]
        )

        self.optim = optax.adamw(learning_rate=self.args['lr'], weight_decay=self.args['wd'])
        # self.optim = optax.adam(learning_rate=self.args['lr'])
        # self.optim = optax.adamw(learning_rate=lr_logic, weight_decay=self.args['wd'])
        optim_params = self.optim.init(params)

        states = dict({
            'params': params,
            'state': st,
            'optim': optim_params,
            'key': sub
        })
        return states
    
    @functools.partial(jax.jit, static_argnums=0)
    def update_params(self, states, patches):
        key, sub = jax.random.split(states['key'], num=2)

        def adapt_forward(params, state, key, data):
            # Pack model output and state together.
            (loss, model_output), state = self.forward.apply(params, state, key, data, is_training=True)
            return loss, (model_output, state)
        
        grads, (model_output, state) = (jax.grad(adapt_forward, has_aux=True)(states['params'], states['state'], key, patches))
        # (loss, model_output), grads = jax.value_and_grad(self.forward.apply, has_aux=True)(states['params'], key, patches)

        updates, opt_state = self.optim.update(grads, states['optim'], states['params'])
        params = optax.apply_updates(states['params'], updates)

        states = dict({
            'params': params,
            'state': state,
            'optim': opt_state,
            'key': sub
        })
        return states, model_output
    





### --------- ImageClassification ------------ ###
def classification_forward_fn(fns, args):
    def fwd_pass(input, is_training=True):
        # Patch input images
        data, labels, weights = input

        # logits, classifier_loss = fns['ViT'](**args)(data, labels)

        # patches = fns['PatchesAlt'](**args)(data)
        # logits, classifier_loss = fns['ViT'](**args)(patches, labels, is_training)

        patches = fns['PatchesAlt'](**args)(data)
        encoder_outputs = fns['Encoder'](**args)(patches, is_training)

        logits, classifier_loss = fns['ActionHeadClassification'](**args)(encoder_outputs, labels, weights, is_training)

        return classifier_loss, dict({
            'classif_loss': classifier_loss,
            'logits': logits
        })

    return hk.transform_with_state(fwd_pass)


class ImageClassification:
    def __init__(self, arguments):
        self.args = arguments

    def init_params(self, rng, dummy_input):
        key, sub = jax.random.split(rng, num=2)

        self.forward = classification_forward_fn(dict({
            'Encoder': Encoder,
            'PatchesAlt': PatchesAlt,
            'ViT': ViT,
            'ActionHeadClassification': ActionHeadClassification
        }), self.args)
        params, st = self.forward.init(key, dummy_input)


        self.optim = optax.adamw(learning_rate=self.args['lr'], weight_decay=self.args['wd'])
        # self.optim = optax.adam(learning_rate=self.args['lr'])
        # self.optim = optax.adamw(learning_rate=lr_logic, weight_decay=self.args['wd'])
        optim_params = self.optim.init(params)

        states = dict({
            'params': params,
            'state': st,
            'optim': optim_params,
            'key': sub
        })
        return states
    
    @functools.partial(jax.jit, static_argnums=0)
    def update_params(self, states, patches):
        key, sub = jax.random.split(states['key'], num=2)

        def adapt_forward(params, state, key, data):
            # Pack model output and state together.
            (loss, model_output), state = self.forward.apply(params, state, key, data, is_training=True)
            return loss, (model_output, state)
        
        grads, (model_output, state) = (jax.grad(adapt_forward, has_aux=True)(states['params'], states['state'], key, patches))
        # (loss, model_output), grads = jax.value_and_grad(self.forward.apply, has_aux=True)(states['params'], key, patches)

        updates, opt_state = self.optim.update(grads, states['optim'], states['params'])
        params = optax.apply_updates(states['params'], updates)

        states = dict({
            'params': params,
            'state': state,
            'optim': opt_state,
            'key': sub
        })
        return states, model_output


### --------- PartialImageClassification ------------ ###
def classification_partial_forward_fn(fns, args):
    def fwd_pass(input, is_training=True):
        # Patch input images
        data, labels, weights = input
        patches = fns['Patches'](**args)(data)

        (
            unmasked_embeddings,
            unmasked_positions
        ) = fns['PatchEncoder'](**args)(patches)        

        encoder_outputs = fns['Encoder'](**args)(unmasked_embeddings)
        
        encoder_outputs = encoder_outputs + unmasked_positions
        logits, classifier_loss = fns['ActionHeadClassification'](**args)(encoder_outputs, labels, weights, is_training)

        return classifier_loss, dict({
            'classif_loss': classifier_loss,
            'logits': logits
        })

    return hk.transform_with_state(fwd_pass)


class PartialImageClassification:
    def __init__(self, arguments):
        self.args = arguments

    def init_params(self, rng, dummy_input):
        key, sub = jax.random.split(rng, num=2)

        self.forward = classification_partial_forward_fn(dict({
            'Encoder': Encoder,
            'Patches': Patches,
            'PatchEncoder': PatchEncoder,
            'ActionHeadClassification': ActionHeadClassification
        }), self.args)
        params, st = self.forward.init(key, dummy_input)

        self.optim = optax.adamw(learning_rate=self.args['lr'], weight_decay=self.args['wd'])
        # self.optim = optax.adam(learning_rate=self.args['lr'])
        # self.optim = optax.adamw(learning_rate=lr_logic, weight_decay=self.args['wd'])
        optim_params = self.optim.init(params)

        states = dict({
            'params': params,
            'state': st,
            'optim': optim_params,
            'key': sub
        })
        return states
    
    @functools.partial(jax.jit, static_argnums=0)
    def update_params(self, states, patches):
        key, sub = jax.random.split(states['key'], num=2)

        def adapt_forward(params, state, key, data):
            # Pack model output and state together.
            (loss, model_output), state = self.forward.apply(params, state, key, data, is_training=True)
            return loss, (model_output, state)
        
        grads, (model_output, state) = (jax.grad(adapt_forward, has_aux=True)(states['params'], states['state'], key, patches))
        # (loss, model_output), grads = jax.value_and_grad(self.forward.apply, has_aux=True)(states['params'], key, patches)

        updates, opt_state = self.optim.update(grads, states['optim'], states['params'])
        params = optax.apply_updates(states['params'], updates)

        states = dict({
            'params': params,
            'state': state,
            'optim': opt_state,
            'key': sub
        })
        return states, model_output


