import random
import math
import numpy as np

class RandomMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2

        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_mask = int(mask_ratio * self.num_patches)

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.num_patches, self.num_mask
        )
        return repr_str

    def __call__(self):
        mask = np.hstack([
            np.zeros(self.num_patches - self.num_mask),
            np.ones(self.num_mask),
        ])
        np.random.shuffle(mask)
        mask = np.hstack([
            np.zeros([1]),
            mask
        ])
        return mask # [197]

if __name__ == '__main__':
    window_size = (14, 14)
    mask_generator = RandomMaskingGenerator(window_size, 0.75)
    masks = mask_generator()
    print()