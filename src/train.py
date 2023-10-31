import haiku as hk
import jax
import time
import jax.numpy as jnp
import numpy as np
import functools
from itertools import count
import cv2

from cifar import get_loader
from patching import generate_masked_image, reconstruct_from_patch
from optim_grad import ImageClassification, PartialImageClassification, MaskedAE#, Classifier
from models import save, load

from absl import app
from absl import flags


FLAGS = flags.FLAGS


flags.DEFINE_integer('image_size', 64, 'image spatial size', lower_bound=32)
flags.DEFINE_integer('batch', 64, 'batch_size', lower_bound=1)
flags.DEFINE_integer('num_training_updates', int(1e6), 'number of steps to pass', lower_bound=1000)
flags.DEFINE_integer('patch_size', 8, 'size of a token', lower_bound=4)
flags.DEFINE_integer('enc_projection_dim', 256, 'latent space', lower_bound=32)
flags.DEFINE_integer('enc_num_heads', 8, 'size of a token', lower_bound=1)
flags.DEFINE_integer('enc_layers', 16, 'size of a token', lower_bound=1)
flags.DEFINE_integer('dec_projection_dim', 64, 'latent space', lower_bound=32)
flags.DEFINE_integer('dec_num_heads', 4, 'size of a token', lower_bound=1)
flags.DEFINE_integer('dec_layers', 2, 'size of a token', lower_bound=1)
flags.DEFINE_integer('mlp_dim', 1024, 'mlp_dim', lower_bound=1)
flags.DEFINE_integer('num_classes', 100, 'total number of calsses in the task')
flags.DEFINE_float('norm_eps', 1e-6, 'part of image square to mask')
flags.DEFINE_float('mask_proportion', 0.5, 'part of image square to mask')
flags.DEFINE_float('dropout_rate', 0.1, 'probability to drop any node')
flags.DEFINE_float('grad_clip_value', 1.0, 'grad_clip_value')
flags.DEFINE_float('wd', 1e-4, 'weight_decay')
flags.DEFINE_float('lr', 1e-4, 'learning_rate')
flags.DEFINE_list('transformer_units', [2048, 256], '')
flags.DEFINE_list('mlp_head_units', [2048, 1024], '')
flags.DEFINE_string('name', 'Masked_VAE', 'Model name')
flags.DEFINE_bool('var', True, 'calculate variance of the dataset')
flags.DEFINE_bool('pretrained', False, 'load parameters from folder')
flags.DEFINE_string('params_path', 'weights', 'path to the folder containing parameters')


def load_params(states, loaded_states):
    for k, v in loaded_states.items():
        try: 
            states[k]
        except:
            continue
        else:
            states[k] = v
    return states

def load_params_optims(states, loaded_states):
    states['params'] = load_params(states['params'], loaded_states['params'])

    for k, v in loaded_states['optim'][0].mu.items():
        try: 
            states['optim'][0].mu[k]
        except:
            continue
        else:
            states['optim'][0].mu[k] = v

    for k, v in loaded_states['optim'][0].nu.items():
        try: 
            states['optim'][0].nu[k]
        except:
            continue
        else:
            states['optim'][0].nu[k] = v

    return states


def make_grid(samples, num_cols=8, rescale=True):
    batch_size, height, width, ch = samples.shape
    assert batch_size % num_cols == 0
    num_rows = batch_size // num_cols
    samples = jnp.split(samples, num_rows, axis=0)
    samples = jnp.concatenate(samples, axis=1)
    samples = jnp.split(samples, num_cols, axis=0)
    samples = jnp.concatenate(samples, axis=2).squeeze(0)
    return samples

def save_grid(output, idx, losses, img, step, number):
    output = []
    for i in range(number):
        masked = ((output + 1.0) / 2.0) * 255.
        outputs = ((losses['decoder_outputs'][idx] + 1.0) / 2.0) * 255.
        image = ((img[idx] + 1.0) / 2.0) * 255. 
        output.append(np.concatenate([masked, outputs, image], axis=1))
    up = np.concatenate(output[:number//2], axis=1)
    down = np.concatenate(output[number//2:], axis=1)
    cv2.imwrite(f'output/result_{step}_epoque.jpg', np.hstack([up, down]))


def train(argv):
    rng = jax.random.PRNGKey(42)
    key1, key2, rng = jax.random.split(rng, num=3)
    train_loader, test_loader, variance = get_loader(FLAGS.batch, FLAGS.var)

    num_patches = (FLAGS.image_size // FLAGS.patch_size) ** 2
    
    arguments = {FLAGS[i].name: FLAGS[i].value for i in dir(FLAGS)}
    arguments.update({'enc_transformer_units': [FLAGS.enc_projection_dim*2, FLAGS.enc_projection_dim],
                      'dec_transformer_units': [FLAGS.dec_projection_dim*2, FLAGS.dec_projection_dim],
                      'num_patches': num_patches})
    
    # model = MaskedAE(arguments)
    # cls_model = Classifier(arguments)
    model = ImageClassification(arguments)
    # model = PartialImageClassification(arguments)

    dummy = next(iter(train_loader))
    weights = jnp.ones((FLAGS.batch), dtype=np.float32)
    states = model.init_params(key1, (*dummy, weights))
    # cls_states = cls_model.init_params(key2, 
    #                                    (jnp.zeros((FLAGS.batch, 
    #                                               int(dummy[0].shape[1] * (1 - FLAGS.mask_proportion)), 
    #                                               FLAGS.enc_projection_dim), np.float32), 
    #                                    dummy[1],
    #                                    weights))

    # loaded_states = load(FLAGS.params_path)
    # states = load_params_optims(states, loaded_states)

    for i in count():
        train_losses, test_loasses = [], []
        train_accs, test_accs = [], []
        start = time.time()
        for step, img in enumerate(train_loader):

            # Get the embeddings and positions.
            states, losses = model.update_params(states, (*img, weights))
            # _, model_output = model.forward.apply(states['params'],
            #                                            key2, 
            #                                            img[0]) 
            # cls_input = jax.lax.stop_gradient(model_output['encoder_outputs'])
            # cls_states, losses = cls_model.update_params(cls_states, (cls_input, img[1], weights))
            losses = jax.device_get(losses)
            train_losses.append(losses["classif_loss"])

            train_pred = jnp.argmax(losses['logits'], axis=-1)
            train_accuracy = jnp.mean(train_pred == img[1]).item()
            train_accs.append(train_accuracy)
        train_elapsed = time.time() - start
        start = time.time()

        # save(FLAGS.params_path, states)       
        
        for step, tinp in enumerate(test_loader):
            key1, key2, rng = jax.random.split(rng, num=3)
            # Get the embeddings and positions.
            (_, tloss), _ = model.forward.apply(states['params'], 
                                                       states['state'], 
                                                       key2,
                                                       (*tinp, weights),
                                                       is_training=False)
            # _, model_output = model.forward.apply(states['params'],
            #                                            key1, 
            #                                            tinp[0]) 
            # classifier_test_inp = model_output['encoder_outputs']
            # (_, tloss), _ = cls_model.forward.apply(cls_states['params'], 
            #                                          cls_states['state'], 
            #                                          key2, 
            #                                          (classifier_test_inp, tinp[1], weights), 
            #                                          is_training=False)
            tloss = jax.device_get(tloss)
            test_loasses.append(tloss["classif_loss"])

            test_pred = jnp.argmax(tloss['logits'], axis=-1)
            test_accuracy = jnp.mean(test_pred == tinp[1]).item()
            test_accs.append(test_accuracy)
        test_elasped = time.time() - start

        print(f'Epoch {i} Train_loss: {np.mean(train_losses):.5f} | test_loss: {np.mean(test_loasses):.5f} |',
              f'train accuracy: {np.mean(train_accs):.5f} | test accuracy: {np.mean(test_accs):.5f} |'
              f'time: {train_elapsed:.3f} / {test_elasped:.3f}'
              , flush=True)


if __name__ == '__main__':
    local_devices = jax.local_devices()
    global_devices = jax.devices()
    print(local_devices)
    print(global_devices)
    
    with jax.default_device(global_devices[0]):
        app.run(train)
