import torchvision
from torchvision import transforms as tsf
import torch
import numpy as np
import albumentations as A


img_size = 64

def collate(input):
   image = np.zeros((len(input), img_size, img_size, 3))
   labels = []
   for i, (im, lb) in enumerate(input):
        image[i] = im
        labels.append(lb)
   return image, np.longlong(labels)

transform = tsf.Compose(
                        [
                         tsf.ToTensor(),
                         tsf.Resize(img_size, antialias=True),
                         lambda x: x.numpy(), 
                         lambda x: x.transpose(1,2,0),
                         lambda x: A.ColorJitter(p=.5)(image=x),
                         lambda x: A.RandomBrightnessContrast(brightness_limit=1, contrast_limit=1, p=.5)(image=x['image']),
                        #  lambda x: A.RandomFog(p=.5)(image=x['image']),
                        #  lambda x: A.RGBShift(p=.5)(image=x['image']),
                         lambda x: A.RandomRotate90(p=.5)(image=x['image']),
                         lambda x: A.HorizontalFlip(p=.5)(image=x['image']),
                        #  lambda x: A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=.5)(image=x['image']),
                         lambda x: A.Blur(blur_limit=3, p=.5)(image=x['image']),
                         lambda x: A.OpticalDistortion(p=.5)(image=x['image']),
                        #  lambda x: A.GridDistortion(p=.5)(image=x['image']),
                        #  lambda x: A.HueSaturationValue(p=.5)(image=x['image']),
                         lambda x: x['image'], 
                         lambda x: x * 2 - 1.0 # put image data in range [-1, 1] for more stable convergence
                        ]
                        )

train_cifar100 = torchvision.datasets.CIFAR100('./cifar100', train=True, 
                                         transform=transform, 
                                         download=True)

test_cifar100 = torchvision.datasets.CIFAR100('./cifar100', train=False, 
                                         transform=transform, 
                                         download=True)

def compute_variance(tricked=False):
    if tricked:
        return 4.4245717472233694e-06
    train_data, _ = get_loader(1, False, False, False)
    data = np.zeros((len(train_cifar100), img_size, img_size, 3))
    for id, img in enumerate(train_data):
        data[id] = img
    return np.var(data / 255.0)


def get_loader(batch, var=False, shuffle=True, drop_last=True):
    dtldr_train =  torch.utils.data.DataLoader(train_cifar100, batch_size=batch, shuffle=shuffle,
                                         collate_fn=collate, drop_last=drop_last)
    dtldr_test = torch.utils.data.DataLoader(test_cifar100, batch_size=batch, shuffle=False,
                                         collate_fn=collate, drop_last=drop_last)
    variance = compute_variance(tricked=True) if var else None

    return dtldr_train, dtldr_test, variance



import gym
import cv2
from stable_baselines3.common.atari_wrappers import StickyActionEnv, NoopResetEnv, FireResetEnv, EpisodicLifeEnv, ClipRewardEnv


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, width: int = 84, height: int = 84, channels: int = 3) -> None:
        super().__init__(env)
        self.width = width
        self.height = height
        self.channels = channels
        assert isinstance(env.observation_space, gym.spaces.Box), f"Expected Box space, got {env.observation_space}"

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.height, self.width, self.channels),
            dtype=env.observation_space.dtype,  # type: ignore[arg-type]
        )

    def observation(self, frame: np.ndarray) -> np.ndarray:
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame


class AtariWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        noop_max: int = 30,
        screen_size: int = 64,
        terminal_on_life_loss: bool = True,
        clip_reward: bool = True,
        action_repeat_probability: float = 0.0,
    ) -> None:
        if action_repeat_probability > 0.0:
            env = StickyActionEnv(env, action_repeat_probability)
        if noop_max > 0:
            env = NoopResetEnv(env, noop_max=noop_max)
        if terminal_on_life_loss:
            env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():  # type: ignore[attr-defined]
            env = FireResetEnv(env)
        env = WarpFrame(env, width=screen_size, height=screen_size, channels=env.observation_space.shape[-1])
        if clip_reward:
            env = ClipRewardEnv(env)

        super().__init__(env)


class TransposeAndNormalizeImage(gym.ObservationWrapper):
    def __init__(self, env=None):
        """
        Transpose observation space for images
        """
        super().__init__(env)

    def observation(self, ob):
        return ob / 255. * 2 - 1.0


class ToTensorEnv(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ToTensorEnv, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0], [
                obs_shape[0], obs_shape[1], obs_shape[2]
            ],
            dtype=self.observation_space.dtype)
        #self.device = device

    def observation(self, observation):
        #return torch.tensor(observation, device=self.device, dtype=torch.float32)
        return observation
