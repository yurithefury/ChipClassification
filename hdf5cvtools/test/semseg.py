from hdf5cvtools import *
from skimage.transform import resize

datafile = "data/semseg_data.tif"
maskfile = "data/semseg_mask.tif"
hdf5file = "semseg.hdf5"

def preprocess(img):
    return img, resize(img, (121,121))

f1 = create_semseg_dataset_from_scene(hdf5file, datafile, (128,64), (64,32), mask=maskfile, preprocess=preprocess)
f2, f3, f4 = create_semseg_dataset_from_scene(hdf5file, datafile, (100,100), (75,75), mask=maskfile, train_val_test_split=(0.5,0.3,0.2))


print_info(f1)
print_info(f2)
print_info(f3)
print_info(f4)

create_sample_heatmap(f1, "semseg_heatmap1.tif")
create_sample_heatmap(f2, "semseg_heatmap2.tif")
create_sample_heatmap(f3, "semseg_heatmap3.tif")
create_sample_heatmap(f4, "semseg_heatmap4.tif")

create_raster_mask(f2, "semseg_mask.tif", (0,1))
create_raster_mask(f2, "semseg_mask.tif", {0:1, 1:2}, nodata=0)

