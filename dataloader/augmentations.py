import numpy as np
import torch


def DataTransform(sample, config):

    weak_aug = scaling(sample, config.augmentation.jitter_scale_ratio)
    strong_aug = jitter(permutation(sample, max_segments=config.augmentation.max_seg), config.augmentation.jitter_ratio)

    return weak_aug, strong_aug


def jitter(x, sigma=0.8):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)

# x is of shape (batch_size, num_channels, num_timesteps)
# generate random scaling factor for each sample in batch at each timestep
# then apply the same scaling factor for every channel of the sample
def scaling(x, sigma=1.1):
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=2., scale=sigma, size=(x.shape[0], x.shape[2]))
    ai = []
    for i in range(x.shape[1]):
        xi = x[:, i, :]
        ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])
    return np.concatenate((ai), axis=1)

# split each sample into a random number of time segments
# shuffle these segments and apply the new temporal order to all channels
def permutation(x, max_segments=5, seg_mode="random"):
    #orig_steps is this vector [0,1,2,...,num_timesteps-1]
    orig_steps = np.arange(x.shape[2])

    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))
    # matrix to return
    ret = np.zeros_like(x)
    # i is sample index, pat is of shape (num_channels, num_timesteps)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            # np.random.permutation(splits) to have splits in random order
            # then we concatenate the permutated splits and flatten 
            # .ravel is .flatten but faster and more memory efficient 
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[0,warp]
        else:
            ret[i] = pat
    return torch.from_numpy(ret)

