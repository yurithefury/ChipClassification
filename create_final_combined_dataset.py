from hdf5cvtools import *
from utils import CC, load_labels, load_image

import random
import pandas as pd
import string

seed = int("".join(str(string.ascii_lowercase.index(x)) for x in "sugarbyte"))
random.seed(seed)

# img_list = set()
# A_PS_labels = load_labels("data/raw/amazon/labels.csv", CC.LABELS.CATEGORICAL.CLOUD, to_dataframe=True)
# img_list |= set(map(lambda x : os.path.join("data/raw/amazon/tif", x), A_PS_labels.index))
#
# T_PS_labels = load_labels("data/raw/tropics/planetlabs/labels.csv", CC.LABELS.CATEGORICAL.CLOUD, to_dataframe=True)
# img_list |= set(map(lambda x : os.path.join("data/raw/tropics/planetlabs/tif", x), T_PS_labels.index))
#
# T_S2_labels = load_labels("data/raw/tropics/sentinel2/labels.csv", CC.LABELS.CATEGORICAL.CLOUD, to_dataframe=True)
# img_list |= set(map(lambda x : os.path.join("data/raw/tropics/sentinel2/tif", x), T_S2_labels.index))
img_list = set()
A_PS_labels = load_labels("data/raw/amazon/labels.csv", CC.LABELS.ALL, to_dataframe=True)
img_list |= set(map(lambda x : os.path.join("data/raw/amazon/tif", x), A_PS_labels.index))

T_PS_labels = load_labels("data/raw/tropics/planetlabs/labels.csv", CC.LABELS.ALL, to_dataframe=True)
img_list |= set(map(lambda x : os.path.join("data/raw/tropics/planetlabs/tif", x), T_PS_labels.index))

T_S2_labels = load_labels("data/raw/tropics/sentinel2/labels.csv", CC.LABELS.ALL, to_dataframe=True)
img_list |= set(map(lambda x : os.path.join("data/raw/tropics/sentinel2/tif", x), T_S2_labels.index))


labels = pd.concat([A_PS_labels, T_PS_labels, T_S2_labels], verify_integrity=True)

num_val = int(0.1*len(img_list))
num_test = int(0.1*len(img_list))
num_train = len(img_list) - num_test - num_val

not_train_imgs = set(random.sample(img_list, num_test+num_val))
train_imgs = set(img for img in img_list if img not in not_train_imgs)
val_imgs = set(random.sample(not_train_imgs, num_val))
test_imgs = set(img for img in not_train_imgs if img not in val_imgs)

train_ids = set(map(os.path.basename, train_imgs))
test_ids = set(map(os.path.basename, test_imgs))
val_ids = set(map(os.path.basename, val_imgs))

print("Training:")
print(labels.loc[labels.index.map(lambda x : x in train_ids), :].describe())

print("Validation:")
print(labels.loc[labels.index.map(lambda x : x in val_ids), :].describe())

print("Test:")
print(labels.loc[labels.index.map(lambda x : x in test_ids), :].describe())

def get_img(fpath):
    img = load_image(fpath, (128,128), list(range(6)))
    if np.any(np.isnan(img)):
        raise Exception("NaNs in img: {}".format(fpath))
    return img

# train = "data/proc/cloud_train.hdf5"
# test = "data/proc/cloud_test.hdf5"
# val = "data/proc/cloud_val.hdf5"
train = "data/proc/all_train.hdf5"
test = "data/proc/all_test.hdf5"
val = "data/proc/all_val.hdf5"

for dset, imgs in zip((train, test, val), (train_imgs, test_imgs, val_imgs)):
    print(f'building {dset}...')
    print("\treading images...")
    create_classification_dataset_from_chips(dset, list(imgs), load=get_img, labels=labels, verbose=True)

print("\tcomputing statistics...")
compute_image_statistics(train, band_axis=3)
print("\tnormalising images")
normalise_images(train, band_axis=3, batch=5000)
normalise_images(val,band_axis=3,batch=5000, stats_from=train)
normalise_images(test,band_axis=3,batch=5000, stats_from=train)

