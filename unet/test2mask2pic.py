# -*- coding:utf-8 -*-
# -*- coding:utf-8 -*-

from unet import *
from data import *

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";

# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="1";
from tensorflow.keras.models import Sequential

myunet = myUnet()
model = myunet.get_unet()
model.load_weights('unet.hdf5')

# test2mask
imgs_train, imgs_mask_train, imgs_test = myunet.load_data()
imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
np.save('./results/imgs_mask_test.npy', imgs_mask_test)

# mask2pic
myunet.save_img()
