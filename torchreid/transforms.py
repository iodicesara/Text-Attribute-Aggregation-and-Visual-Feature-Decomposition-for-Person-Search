from __future__ import absolute_import
from __future__ import division

from torchvision.transforms import *

from PIL import Image
import random
import numpy as np
import math


class Random2DTranslation(object):
    """
    With a probability, first increase image size to (1 + 1/8), and then perform random crop.

    Args:
    - height (int): target height.
    - width (int): target width.
    - p (float): probability of performing this transformation. Default: 0.5.
    """
    def __init__(self, height, width, p=0.5, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.p = p
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
        - img (PIL Image): Image to be cropped.
        """
        if random.uniform(0, 1) > self.p:
            return img.resize((self.width, self.height), self.interpolation)
        
        new_width, new_height = int(round(self.width * 1.125)), int(round(self.height * 1.125))
        resized_img = img.resize((new_width, new_height), self.interpolation)
        x_maxrange = new_width - self.width
        y_maxrange = new_height - self.height
        x1 = int(round(random.uniform(0, x_maxrange)))
        y1 = int(round(random.uniform(0, y_maxrange)))
        croped_img = resized_img.crop((x1, y1, x1 + self.width, y1 + self.height))
        return croped_img



class RandomSizedRectCrop(object):
    def __init__(self, height, width,  p=0.5,interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.interpolation = interpolation
        self.p = p


    def __call__(self, img):
        scale = Random2DTranslation(self.height, self.width,
                                    interpolation=self.interpolation)
        if random.uniform(0, 1) > self.p:
            return scale(img)
        else:
            for attempt in range(10):
                area = img.size[0] * img.size[1]
                target_area = random.uniform(0.02, 0.4) * area
                aspect_ratio = random.uniform(0.3, 3.3)


                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))

                if w <= img.size[0] and h <= img.size[1]:
                    x1 = random.randint(0, img.size[0] - w)
                    y1 = random.randint(0, img.size[1] - h)

                    img = img.crop((x1, y1, x1 + w, y1 + h))
                    assert(img.size == (w, h))

                    return img.resize((self.width, self.height), self.interpolation)

        # Fallback

        return scale(img)



class RandomEraising(object):
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3,r2=3.3, mean=[0.4914, 0.4822, 0.4465]):
        self.p = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.rl = r1
        self.rh=r2
        self.max_attempt=100


    def __call__(self, image):

        image = np.asarray(image).copy()

        if np.random.random() > self.p:
            return image

        h, w = image.shape[:2]
        image_area = h * w

        for _ in range(self.max_attempt):
            mask_area = np.random.uniform(self.sl, self.sh) * image_area
            aspect_ratio = np.random.uniform(self.rl, self.rh)
            mask_h = int(np.sqrt(mask_area * aspect_ratio))
            mask_w = int(np.sqrt(mask_area / aspect_ratio))

            if mask_w < w and mask_h < h:
                x0 = np.random.randint(0, w - mask_w)
                y0 = np.random.randint(0, h - mask_h)
                x1 = x0 + mask_w
                y1 = y0 + mask_h
                image[y0:y1, x0:x1] = np.random.uniform(0, 1)
                break

        return image
