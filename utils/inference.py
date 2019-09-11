import rasterio as rio
import numpy as np
from hdf5cvtools import calculate_windows, read_scene
from .cnn import load_model
import tqdm


def run(cnn, tif_in, tif_out, batch_size=32, window_size=128, preprocess=None, make_prediction=None):
    """
    Run inference with a Keras multilabel/multiclass CNN on a single satellite scene.
    :param cnn: Path to the Keras model HDF5 file (architecture+weights).
    :param tif_in: Path to input scene.
    :param tif_out: Path/list of path to output scene/s.  Only give a list if make_prediction_cloud_only is not None.
    :param batch_size: Inference batch size, does not affect output, only speed.
    :param window_size: Chip window size to use.
    :param preprocess: Function which takes a single image and performs any necessary resizing/standardisation/band
        calculation.
    :param make_prediction: By default, run creates a single raster with a band for each layer containing
        the prediction scores (between 0 and 1) for that label.  If make_prediction_cloud_only (a function) is given, run
        will create a raster for each element of make_prediction_cloud_only's output.  The length of make_prediction_cloud_only's output
        should be equal to the length of tif_out.  For example, if make_prediction_cloud_only outputs a two-element list where
        the first element is a cloud label and the second a shadow label, then tif_out should be
        [cloud_mask.tif, shadow_mask.tif]
        See *make_prediction_cloud_only()* for an example and additional information.
    :return:
    """

    img_arr, crs, trans, nodata, height, width = read_scene(tif_in)

    mask = np.logical_and.reduce(img_arr!=nodata, axis=-1)
    img_arr = img_arr.astype(np.float)

    windows = calculate_windows(mask, (window_size, window_size))
    num_batches = int(np.ceil(len(windows) / batch_size))

    if isinstance(tif_out, list) and make_prediction is None:
        raise Exception("make_prediction_cloud_only is None - do not give multiple outputs.")
    elif isinstance(tif_out, str) and make_prediction is not None:
        raise Exception("make_prediction_cloud_only is specified but only a single output is given.")


    cnn = load_model(cnn, compile=False)
    img_shp = cnn.input_shape[1:3]
    num_labels = cnn.output_shape[1]

    if make_prediction is None:
        output_arr = np.zeros((width,height,num_labels), dtype=np.float)
        output_scores = True
        make_prediction = lambda x : x
    else:
        output_scores = False
        output_arr = np.zeros((width, height, len(tif_out)), dtype=np.uint8)


    output_mask = np.zeros((width, height), dtype=np.bool)

    for i in tqdm.trange(num_batches):
        batch_windows = windows[i * batch_size:(i + 1) * batch_size]
        X = np.stack([preprocess(img_arr[x:x + window_size, y:y + window_size, :], img_shp) for x, y in batch_windows])
        y_scores = cnn.predict(X)
        for i,(x,y) in enumerate(batch_windows):
            output_arr[x:x+window_size, y:y+window_size, :] = make_prediction(y_scores[i])
            output_mask[x:x+window_size, y:y+window_size] = True

    del img_arr

    output_arr -= (~output_mask)[...,None]

    if output_scores:
        with rio.open(tif_out, mode='w', driver="GTiff", crs=crs, transform=trans, height=height, width=width,
                      count=num_labels, dtype=output_arr.dtype, nodata=-1) as tif:
            for i in range(num_labels):
                tif.write(output_arr[:,:, i].T, i+1)
    else:
        for i, t in enumerate(tif_out):
            with rio.open(t, mode='w', driver="GTiff", crs=crs, transform=trans, height=height, width=width,
                          count=1, dtype=np.uint8, nodata=np.uint8(-1)) as tif:
                tif.write(output_arr[:, :, i].astype(np.uint8).T, 1)
