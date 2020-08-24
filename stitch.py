# %%
import os
import numpy as np
import pandas as pd
from PIL import Image


# %%
data_dir = os.path.join(os.getcwd(), 'data')
result_dir = os.path.join(os.getcwd(), 'results')


# %%
height = np.array([13500, 19500, 26000])
width = 24000
name_list = ['4.5_1', '4.5_2', '4.5_3', '6.5_1', '6.5_2', '', '9.5_1', '9.5_2', '9.5_3']


# %%
imarray_big = np.zeros((sum(height), width * 3), dtype='uint8')
Image.MAX_IMAGE_PIXELS = None
for i in range(3):
    for j in range(3):
        name = name_list[3*i + j]
        if name == '':
            continue
        im = Image.open(os.path.join(data_dir, 'nuclei_PCW{}.tif'.format(name)))
        src = np.array(im)
        print(src.shape)
        dst = src / 256
        print(dst.max())
        dst = (dst-dst.min()) / (dst.max()-dst.min()) * 200
        dst = dst.astype('uint8')
        imarray_big[height[:i].sum():height[:i].sum() + src.shape[0], j*width:j*width + src.shape[1]] = dst
im_big = Image.fromarray(imarray_big)
print(imarray_big.shape)
im_big.save(os.path.join(result_dir, 'nuclei_heart.tif'), format='tiff', dpi=(96, 96))


