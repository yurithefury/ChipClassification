from hdf5cvtools import *
from skimage.transform import resize
import glob

zipfile = "data/scene.zip"
safedir = "data/scene.SAFE"
hdf5file1 = "classification_train.hdf5"
hdf5file2 = "classification_apply_safe.hdf5"
hdf5file3 = "classification_apply_zip.hdf5"
flist = glob.glob("data/chips/*.TIF")
labels = pd.read_csv("data/chiplabels.csv", index_col=0)

def preprocess(img):
    ndvi = (img[:,:,3] - img[:,:,0])/(img[:,:,0] + img[:,:,3])
    new_img = np.concatenate((img, ndvi[:,:,None]), axis=-1)
    return img, resize(new_img, (256,256))

def get_labels(f):
    row = labels.loc[f,'tags'].split()
    return [t in row for t in ['clear', 'haze', 'partly_cloudy', 'cloudy']]

def load(f):
    img = read_chip_tif(f)
    return preprocess(img)

f1 = create_classification_dataset_from_chips(hdf5file1, flist, load, image_set_names=['original', 'ndvi'],
                                              labels=get_labels)
print_info(f1)

f2 = create_classification_dataset_from_scene(hdf5file2, safedir, window_size=(128,128), image_set_names=["original", "ndvi"],
                                              preprocess=preprocess)
print_info(f2)

f3 = create_classification_dataset_from_scene(hdf5file3, zipfile, window_size=(128,128))
print_info(f3)

create_sample_heatmap(f2, "classification_heatmap1.tif")
create_sample_heatmap(f3, "classification_heatmap2.tif")
