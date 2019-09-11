"""
An example script for running inference,
"""

from utils import inference, CC
from skimage.transform import resize
import numpy as np
import os

# This are from the training dataset (Amazon and Tropics combined)
MEANS = np.array([[[4.43031152e+03, 3.83204468e+03, 2.86734009e+03, 5.79344238e+03, 1.48621738e-01, 2.15694770e-01]]])
STDS = np.array([[[2.12724219e+03, 1.86171362e+03, 1.77039026e+03, 2.19306738e+03, 1.90254897e-01, 1.84691772e-01]]])


def preprocess(img, resize_shape=None):
    if resize_shape is not None:
        img = resize(img, resize_shape, preserve_range=True, mode='constant', anti_aliasing=False)
    ndvi = (img[:, :, 3] - img[:, :, 0]) / (img[:, :, 3] + img[:, :, 0])
    gndvi = (img[:, :, 3] - img[:, :, 1]) / (img[:, :, 3] + img[:, :, 1])
    img = np.concatenate([img, ndvi[..., None], gndvi[..., None]], axis=-1)
    img -= MEANS
    img /= STDS
    return img


def make_prediction_cloud_only(scores):
    """
    This function specifies the logic for choosing which labels to assign a chip.  It takes as input the prediction scores
    of a single chip as a 1D vector with length equal to the number of classes.  It must output a list of labels
    (NON-NEGATIVE integers) for this particular chip.
    :param scores: 1D numpy array
    :return: list[int]
    """
    return [scores.argmax()]

def make_prediction_multilabel(scores):
    cloud_pred, _ = max((scores[i], i) for i in CC.LABELS.CATEGORICAL.CLOUD.values())
    shadow_pred, _ = max((scores[i], i) for i in CC.LABELS.CATEGORICAL.SHADOW.values())
    return [cloud_pred, shadow_pred]


if __name__ == '__main__':
    os.makedirs('scrap/', exist_ok=True)
    # -----------------------------------------------------------------------------------------------------------------
    # EXAMPLE 1 - outputting model scores
    input_files = [
        "data/raw/scenes/ps/496378_5530810_2017-05-09_1015_BGRN_Analytic_metadata_TOA.tif",
        "data/raw/scenes/s2/S2A_MSIL1C_20161208T003702_N0204_R059_T55KCA_20161208T003914_TOA.tif"
    ]
    output_files = [
        "scrap/ps_cloud_scores.tif",
        "scrap/s2_cloud_scores.tif",
        # Supply a string for each output file since here "make_prediction" is not given in "run()"
    ]

    # Window size in pixels. 128 for PS = 400m = 40 for S2
    window_sizes = [
        128,
        20
    ]

    for inf, outf, ws in zip(input_files, output_files, window_sizes):
        inference.run("models/sugarbyte_cloud_labels.hdf5", inf, outf,
                      batch_size=32,
                      window_size=ws,
                      preprocess=preprocess
                      )
    quit()
    # -----------------------------------------------------------------------------------------------------------------
    # EXAMPLE 2 - outputting model prediction (part 1)
    input_files = [
        "data/raw/scenes/ps/496378_5530810_2017-05-09_1015_BGRN_Analytic_metadata_TOA.tif",
        "data/raw/scenes/s2/S2A_MSIL1C_20161208T003702_N0204_R059_T55KCA_20161208T003914_TOA.tif"
    ]
    output_files = [
        ["scrap/ps_cloud.tif"],
        ["scrap/s2_cloud.tif"],
        # Supply a string for each output file since here "make_prediction" is not given in "run()"
    ]


    window_sizes = [128, 40]

    for inf, outf, ws in zip(input_files,output_files,window_sizes):
        inference.run("models/sugarbyte_cloud_labels.hdf5", inf, outf,
                      batch_size=32,
                      window_size=ws,
                      make_prediction=make_prediction_cloud_only,
                      preprocess=preprocess
                      )
    # -----------------------------------------------------------------------------------------------------------------
    #EXAMPLE 3 - outputting model prediction (part 2)
    input_files = [
        "data/raw/scenes/ps/496378_5530810_2017-05-09_1015_BGRN_Analytic_metadata_TOA.tif",
        "data/raw/scenes/s2/S2A_MSIL1C_20161208T003702_N0204_R059_T55KCA_20161208T003914_TOA.tif"
    ]
    output_files = [
        ["scrap/ps_cloud2.tif", "scrap/ps_shadow2.tif"],
        ["scrap/s2_cloud2.tif", "scrap/s2_shadow2.tif"],
        # now we have to supply two files per input file,
        # because make_prediction_multilabel returns two labels
    ]
    window_sizes = [128, 40]

    for inf, outf, ws in zip(input_files,output_files,window_sizes):
        inference.run("models/sugarbyte_all_labels.hdf5", inf, outf,
                      batch_size=32,
                      window_size=ws,
                      make_prediction=make_prediction_multilabel,
                      preprocess=preprocess
                      )
