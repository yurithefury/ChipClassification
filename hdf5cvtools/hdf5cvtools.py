import numpy as np
import h5py
import rasterio
import fiona
import pandas as pd

import os
import re
import glob
import shutil
import fnmatch
import zipfile
import collections

from tqdm import tqdm
from xml.etree.ElementTree import ElementTree

from skimage.measure import label

class HDF5SchemaError(Exception):
    pass


class _C:
    SENTINEL2_BAND_SUFFIX = {"B": "02", "G": "03", "R": "04", "NIR": "08"}
    SENTINEL2_BAND_SUFFIX_INV = {v: k for k, v in SENTINEL2_BAND_SUFFIX.items()}
    BAND_ORDER = ("R", "G", "B", "NIR")

    class ROOT_ATTR:
        GEOTRANSFORM = 'geotransform'
        CRS =  'crs'
        WIDTH = 'width'
        HEIGHT = 'height'
        WINDOW_X = 'window_x'
        WINDOW_Y = 'window_y'

        _ALL = [GEOTRANSFORM, CRS, WIDTH, HEIGHT, WINDOW_X, WINDOW_Y]
        _GEO = [GEOTRANSFORM, CRS, WIDTH, HEIGHT, WINDOW_X, WINDOW_Y]

def _handle_hdf5filename_arg(f, mode='r'):
    if isinstance(f, h5py.File):
        return f, False
    elif isinstance(f, str):
        return h5py.File(f, mode=mode), True
    else:
        raise TypeError

def _get_metadata_from_safe_xml(xmlfile, _sv=dict()):
    if isinstance(xmlfile, str):
        t = ElementTree()
        r = t.parse(xmlfile)
        return _get_metadata_from_safe_xml(r)
    else:
        result = xmlfile.findall("Special_Values")
        if len(result) > 0:
            for r in list(result):
                _sv[r.findtext("SPECIAL_VALUE_TEXT")] = r.findtext("SPECIAL_VALUE_INDEX")
            return _sv
        else:
            for child in list(xmlfile):
                _sv.update(_get_metadata_from_safe_xml(child, _sv=_sv))
            return _sv


def get_scene_information(h5f):
    h5f, close = _handle_hdf5filename_arg(h5f,'r')
    bad_h5 = False
    for key in _C.ROOT_ATTR._GEO:
        if key not in h5f.attrs:
            bad_h5 = True
            break
    ids = h5f['ids'][:]
    ids = list(map(lambda x: x.decode(), ids))
    re_pattern = re.compile('x(\d+)y(\d+)')
    windows = []
    for id in ids:
        m = re.fullmatch(re_pattern, id)
        if m is None:
            bad_h5 = True
            break
        else:
            windows.append(tuple(map(int, m.groups())))

    if bad_h5:
        h5f.close()
        raise HDF5SchemaError("no/incomplete scene geodata.")

    px = h5f.attrs[_C.ROOT_ATTR.WINDOW_X]
    py = h5f.attrs[_C.ROOT_ATTR.WINDOW_Y]

    crs = rasterio.crs.CRS.from_string(h5f.attrs[_C.ROOT_ATTR.CRS])
    trans = rasterio.transform.Affine.from_gdal(*h5f.attrs[_C.ROOT_ATTR.GEOTRANSFORM])
    height = h5f.attrs[_C.ROOT_ATTR.HEIGHT]
    width = h5f.attrs[_C.ROOT_ATTR.WIDTH]
    window_size = (px, py)

    if close:
        h5f.close()

    return windows, crs, trans, height, width, window_size



def _read_scene_tif(scene):
    """
    Get image data from a TIF image
    """
    with rasterio.open(scene) as f:
        img_arr = np.stack([f.read(i) for i in f.indexes], axis=-1)
        coordinate_sys = f.crs
        geotransform = f.transform
        tif_no_data_value = f.nodata
        height = f.height
        width = f.width

    img_arr = img_arr.transpose(1, 0, 2)

    return img_arr, coordinate_sys, geotransform, tif_no_data_value, height, width


def _read_scene_safe(safedir):
    """
    Get image data from a Sentinel-2 SAFE-archive directory
    """
    if len(glob.glob(safedir + "/manifest.safe")) == 0:
        raise Exception("Not a valid SAFE archive - missing manifest.safe")

    tile = glob.glob(safedir + "/GRANULE/*/IMG_DATA/")
    if len(tile) != 1:
        raise Exception("SAFE must contain only one tile (found {})".format(len(tile)))

    tile = tile[0]
    jp2files = glob.glob(tile + "*B??.jp2".format(tile))
    regexp = re.compile(r"B(\w\w)\.jp2$")
    bandjp2s = dict()

    for j in jp2files:
        m = re.search(regexp, j)
        if m is not None:
            bandjp2s[m.group(1)] = j

    img_arr = []
    for b in _C.BAND_ORDER:
        with rasterio.open(bandjp2s[_C.SENTINEL2_BAND_SUFFIX[b]], 'r') as img:
            img_arr.append(img.read(1))
            crs = img.crs
            trans = img.transform
            height = img.height
            width = img.width

    img_arr = np.stack(img_arr, axis=-1)

    # search for correct xml file:
    xmlfile = glob.glob(safedir + "/*.xml")
    xmlfile.remove(safedir + "/INSPIRE.xml")
    assert len(xmlfile) == 1
    xmlfile = xmlfile[0]
    nodata = int(_get_metadata_from_safe_xml(xmlfile)["NODATA"])

    return img_arr, crs, trans, nodata, height, width


def _read_scene_safe_zip(zp):
    """
    Get image data from a zipped Sentinel-2 SAFE-archive directory (only extracts what is absolutely necessary)
    """
    workdir = os.path.dirname(zp)

    z = zipfile.ZipFile(zp, "r")

    contents = z.namelist()

    if len(fnmatch.filter(contents, "*/manifest.safe")) == 0:
        z.close()
        raise Exception("Not a valid SAFE archive - missing manifest.safe")

    tile = fnmatch.filter(contents, "*/GRANULE/*/IMG_DATA/")
    if len(tile) != 1:
        raise Exception("zip folder must contain only one tile (found {})".format(len(tile)))

    tile = tile[0]
    jp2files = fnmatch.filter(contents, tile + "*??.jp2")
    regexp = re.compile(r"B(\w\w)\.jp2$")
    bandjp2s = dict()

    for j in jp2files:
        m = re.search(regexp, j)
        if m is not None:
            bandjp2s[m.group(1)] = j

    img_arr = []
    example_jp2 = list(bandjp2s.values())[0]
    SAFE_folder = os.path.join(workdir, example_jp2.split("/")[0])

    for b in _C.BAND_ORDER:
        jp2_path = z.extract(bandjp2s[_C.SENTINEL2_BAND_SUFFIX[b]], path=workdir)
        with rasterio.open(jp2_path, 'r') as img:
            img_arr.append(img.read(1))
            crs = img.crs
            trans = img.transform
            height = img.height
            width = img.width

    img_arr = np.stack(img_arr, axis=-1)

    # search for correct xml file:
    for f in contents:
        p = f.split("/")
        if len(p) == 2 and p[1].endswith(".xml") and p[1] != "INSPIRE.xml":
            xmlfile = f
            break

    xmlfile = z.extract(xmlfile, path=workdir)
    nodata = int(_get_metadata_from_safe_xml(xmlfile)["NODATA"])
    z.close()

    shutil.rmtree(SAFE_folder)

    return img_arr, crs, trans, nodata, height, width


def _remove_overlapping_windows(fromlist, tolist, splitaxis, window_size):
    # We assume here that the fromlist and tolist are sorted according to sorted(list, key=lambda x : x[groupby_axis])
    groupby_axis = (splitaxis + 1) % 2
    tolist_start_row = tolist[0][groupby_axis]
    move = list(filter(lambda x : x[groupby_axis] == tolist_start_row, fromlist))
    fromlist = fromlist[:-len(move)]
    tolist = move + tolist
    fromlist = list(filter(lambda x : x[groupby_axis] <= tolist_start_row - window_size[groupby_axis], fromlist))
    return fromlist, tolist


def calculate_windows(bitmask, window_size, window_stride=None, splitmode='x', data_split=None):
    if window_stride is None:
        window_stride = window_size

    if data_split is None:
        data_split = np.array([1])
    else:
        data_split = np.array(data_split, dtype=np.float)
        data_split /= data_split.sum()

    windows = []
    y = 0
    px, py = window_size
    while y < bitmask.shape[1]:
        x = 0
        while x < bitmask.shape[0]:
            window_bitmask = bitmask[x:x + px, y:y + py]
            if window_bitmask.shape == (px, py) and np.all(window_bitmask):
                windows.append((x, y))
            x += window_stride[0]
        y += window_stride[1]

    num_windows = (data_split*len(windows)).astype(np.int)
    num_windows[0] = len(windows) - int(num_windows[1:].sum())
    assert np.all(num_windows > 0), "cannot produce datasets with zero length."

    if splitmode == "x":
        splitaxis = 0
    elif splitmode == "y":
        windows = sorted(windows)
        splitaxis = 1
    else:
        raise Exception("splitmode must be `x` or `y`.")

    # Some windows can be moved from the validation to test dataset, and from the training to the validation dataset.
    # Others which overlap must be deleted.
    windows_grouped = []
    offset = 0
    for n in num_windows:
        windows_grouped.append(windows[offset:offset+n])
        offset += n

    for i in range(len(num_windows)-1):
        w1, w2 = _remove_overlapping_windows(windows_grouped[i], windows_grouped[i+1], splitaxis, window_size)
        windows_grouped[i] = w1
        windows_grouped[i+1] = w2

    if len(data_split) == 1:
        return windows_grouped[0]
    else:
        return tuple(windows_grouped)


def _side_verts(side_idx, i, j):
    if side_idx == 0:
        return [(i, j), (i, j+1)]
    elif side_idx == 2:
        return [(i, j+1), (i+1,j + 1)]
    elif side_idx == 4:
        return [(i+1, j+1), (i+1, j)]
    elif side_idx == 6:
        return [(i+1, j), (i,j)]
    else:
        raise ValueError("side_idx not in {0,2,4,6}")


def clockwise_walk(arr):
    """
    Given a 2D boolean array that contains a single True region with no holes, return a list which is a polyline
    (in pixel coordinates) that walks around the region clockwise.  The top-right corner of the image is taken to be
    the origin.
    """
    _NHBR_OFFSET = [
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, -1)
    ]
    X = np.pad(arr, mode='constant', constant_values=0, pad_width=(1,1))
    I = range(1,X.shape[0]-1)
    J = range(1,X.shape[1]-1)

    # find starting pixel and side
    for i in I:
        for j in J:
            if X[i,j]:
                neighbour_vals = [X[i+d[0], j+d[1]] for d in _NHBR_OFFSET[::2]]
                c_side = neighbour_vals.index(False)*2
                c_px = (i,j)
                break
        else:
            continue
        break
    else:
        return []
    poly_line = _side_verts(c_side, *c_px)


    while poly_line[0] != poly_line[-1]:
        # Walk around each pixel clockwise to find boundary edges
        for s in range(1,8):
            nhbr_idx = (c_side + s) % 8
            query_px = (c_px[0] + _NHBR_OFFSET[nhbr_idx][0], c_px[1] + _NHBR_OFFSET[nhbr_idx][1])
            if X[query_px[0], query_px[1]]: # is this neighbour occupied?
                c_side = ((nhbr_idx // 2) * 2 + 6) % 8  # determine boundary side, relative to this neighbour
                vert = _side_verts(c_side, *query_px)[1]
                if poly_line[-1][0] == poly_line[-2][0] == vert[0] or poly_line[-1][1] == poly_line[-2][1] == vert[1]:
                    # Check whether we can simply replace the last point in the polyline
                    poly_line[-1] = vert
                else:
                    poly_line.append(vert)
                c_px = query_px
                break
            elif nhbr_idx % 2 == 0: # Add a new side from the current pixel
                vert = _side_verts(nhbr_idx, *c_px)[1]
                if poly_line[-1][0] == poly_line[-2][0] == vert[0] or poly_line[-1][1] == poly_line[-2][1] == vert[1]:
                    poly_line[-1] = vert
                else:
                    poly_line.append(vert)

        else:
            raise Exception("bug encountered")

    poly_line = np.array(poly_line, dtype=np.uint32)-1
    return poly_line



class RunningStatistics():
    # numerically stable variance computation - see Chan et. al (1983)
    def __init__(self):
        self._n = 0
        self._T = None
        self._SS = None

    def update(self, X, axis=None):
        """

        :param X: New data
        :param axis: Which axes to consider as containing independent samples.  For example, when computing the average
            band values of an image X, with shape (n_channels, height, width), set this to (1,2).  Or, if X has shape
            (n_images, n_channels, height, width) set this to (0,2,3)
        :return:
        """
        # statistics for this batch
        if axis is None:
            n_b = np.prod(X.shape)
        elif isinstance(axis, int):
            n_b = X.shape[axis]
        else:
            n_b = np.prod(np.array(X.shape)[list(axis)])

        T_b = np.sum(X, axis=axis, keepdims=True)
        S_b = np.sum(np.square(X - T_b / n_b), axis=axis)
        T_b = T_b.reshape(tuple(d for d in T_b.shape if d > 1))

        # update global statistics
        if self._n == 0:
            self._T = T_b
            self._SS = S_b
        else:
            self._SS += S_b + (self._n / (n_b * (n_b + self._n))) * np.square((self._T * n_b / self._n) - T_b)
            self._T += T_b

        self._n += n_b


    def mean(self):
        return (self._T / self._n)

    def std(self):
        return np.sqrt(self._SS / self._n)


def read_scene(scene):
    """
    Get image array and georeferencing data for a single scene.
    :returns: A tuple containing:
        - img_arr: image data in (width, height, bands) order (numpy array)
        - coordinate_sys: a rasterio.CRS object
        - geotransform: a rasterio.Affine object
        - tif_no_data_value: the TIF nodata valid for invalid pixels (float/int)
        - height: (int)
        - width: (int)
    """
    if scene.endswith(".tif") or scene.endswith(".TIF"):
        return _read_scene_tif(scene)
    elif scene.endswith(".zip"):
        return _read_scene_safe_zip(scene)
    elif scene.endswith(".SAFE"):
        return _read_scene_safe(scene)
    else:
        raise Exception("bad file type: {}".format(os.path.splitext(scene)[1]))


def read_chip_tif(scene):
    """
    Get image data from a TIF image
    """
    with rasterio.open(scene) as f:
        img_arr = np.stack([f.read(i) for i in f.indexes], axis=-1)

    img_arr = img_arr.transpose(1, 0, 2)

    return img_arr


def create_classification_dataset_from_chips(h5file_path, filelist, load, labels=None, chunked=None, verbose=False):
    """
    Create a dataset suitable for per-image classification tasks from individual (possibly labelled) images.

    Arguments
    --------
    :param h5file_path: Destination path
    :param filelist: List of paths to source images.
    :param load: Function which loads an image (and possibly performs some preprocessing). The function should take a
        an element of *filelist* return a list of output images (as numpy arrays).  Each output image will be placed in
        a HDF5 dataset in the `images` group.  This is useful for creating a dataset with images resized to different
        sizes for training different CNN.
    :param labels: Either a dict, pandas.Dataframe or callable which maps the filenames (without path) in *filelist*
        to a binary label array.
    :params chunked: Whether to use chunked HDF5 storage.  Recommended if you are going to be accessing images in a
        randomised order.  The default is True when `labels` are given and False otherwise.

    Returns
    -------
    :return: Path to created dataset.
    """

    ids = list(map(lambda x: os.path.basename(x), filelist))
    h5f = h5py.File(h5file_path, "w")

    if labels is not None:
        if callable(labels):
            get_label = labels
        elif isinstance(labels, pd.DataFrame):
            get_label = lambda x: labels.loc[x, :].values.astype(np.bool)
        else:
            get_label = lambda x: np.array(labels[x], dtype=np.bool)

    sample_output = load(filelist[0])
    output_size = sample_output.shape

    h5f.create_dataset("ids", data=np.array(ids, dtype='S'))

    if (chunked is None and labels is not None) or chunked:
        chunks = (1,) + output_size
    else:
        chunks = None

    h5f.create_dataset('images', shape=(len(filelist),) + output_size, dtype=np.float32, chunks=chunks)

    if load is not None:
        num_classes = len(get_label(ids[0]))
        h5f.create_dataset('labels', (len(filelist), num_classes), dtype=np.bool)

    for k, fname in enumerate(tqdm(filelist, disable=not verbose)):
        h5f['images'][k] = load(fname)
        if labels is not None:
            h5f['labels'][k] = get_label(ids[k])

    h5f.close()

    return h5file_path


def create_classification_dataset_from_scene(h5file_path, scene, window_size, preprocess=None, nodata=None, verbose=False):
    """
    Create a dataset suitable for per-image classification tasks from a single large satellite scene.  Note this is
    only applicable for applying an existing model

    Arguments
    --------
    :param h5file_path: Destination path
    :param scene: Source path
    :param window_size: Size of windows to use (#x pix, #y pix)
    :param preprocess: Pre-processing function.  The function should take a chunk of the input scene, with size
        given by *window_size* and return a list of output images.  Each output image will be placed in an HDF5 dataset
        in the `images` group.  This is useful for creating a dataset with images resized to different sizes for training
        different CNN.
    :param image_set_names: List of strings specifying the name of each dataset.  Should have length equal to the number
        of outputs given by *preprocess*.  Default is ['set1','set2','set3',...]
    :param nodata: Raster no-data value to use.  Defaults to the no-data value in *scene*.
    :param data_order: Which order to keep the data in, either 'NBXY' for (num_images, bands, px, py) or
        'NXYB' for (num_images, px, py, bands).

    Returns
    -------
    :return: Path to created dataset.
    """

    img_arr, coordinate_sys, geotransform, tif_no_data_value, height, width = read_scene(scene)
    if nodata is None:
        nodata = tif_no_data_value
    if nodata is None:
        nodata = 0  # fallback

    if preprocess is None:
        preprocess = lambda x: (x,)

    px, py = window_size

    bitmask = np.logical_and.reduce(img_arr != nodata, axis=-1)
    if verbose: print("calculating windows...")
    windows = calculate_windows(bitmask, window_size)

    sample_chip = img_arr[windows[0][0]:windows[0][0] + px, windows[0][1]:windows[0][1] + py, :]
    sample_outputs = preprocess(sample_chip)
    output_size = sample_outputs.shape

    h5f = h5py.File(h5file_path, 'w')
    # Image chip data
    h5f.create_dataset("images", shape=(len(windows),) + output_size, dtype=np.float32)

    h5f.attrs['crs'] = coordinate_sys.to_string()
    h5f.attrs['geotransform'] = tuple(geotransform.to_gdal())
    h5f.attrs['width'] = width
    h5f.attrs['height'] = height
    h5f.attrs['window_x'] = px
    h5f.attrs['window_y'] = py

    ids = list()
    if verbose: print("writing...")

    for k, pos in enumerate(tqdm(windows, disable=not verbose)):
        if (k + 1) % 1000 == 0:
            h5f.flush()
        x, y = pos

        h5f['images'][k] = preprocess(img_arr[x:x + px, y:y + py, :])

        ids.append("x{:d}y{:d}".format(x, y))

    h5f.create_dataset("ids", data=np.array(ids, dtype='S'))
    h5f.close()

    return h5file_path

def create_classification_schema_from_scene(h5file_path, scene, window_size, nodata=None, verbose=False):
    """
    Create just the dataset schema, suitable for per-image classification tasks from a single large satellite scene.
    Note this is will not create the "images" group, only the "ids" and "labels" groups.  This is designed for very large
    scenes where the actual classification should be do_normalise on-the-fly with pre-processing to avoid excessive disk usage.
    The function copies georeferencing data and calculates windows, the latter of which are stored in the "ids" field in
    the format "x<X>y<Y>" where X and Y are integers representing the pixel coordinates of the top-left corner of the
    window.

    Arguments
    --------
    :param h5file_path: Destination path
    :param scene: Source path
    :param window_size: Size of windows to use (#x pix, #y pix)
    :param preprocess: Pre-processing function.  The function should take a chunk of the input scene, with size
        given by *window_size* and return a list of output images.  Each output image will be placed in an HDF5 dataset
        in the `images` group.  This is useful for creating a dataset with images resized to different sizes for training
        different CNN.
    :param image_set_names: List of strings specifying the name of each dataset.  Should have length equal to the number
        of outputs given by *preprocess*.  Default is ['set1','set2','set3',...]
    :param nodata: Raster no-data value to use.  Defaults to the no-data value in *scene*.

    Returns
    -------
    :return: Path to created dataset.
    """

    img_arr, coordinate_sys, geotransform, tif_no_data_value, height, width = read_scene(scene)
    if nodata is None:
        nodata = tif_no_data_value
    if nodata is None:
        nodata = 0  # fallback

    px, py = window_size

    bitmask = np.logical_and.reduce(img_arr != nodata, axis=-1)
    if verbose: print("calculating windows...")
    windows = calculate_windows(bitmask, window_size)

    h5f = h5py.File(h5file_path, 'w')

    h5f.attrs[_C.ROOT_ATTR.CRS] = coordinate_sys.to_string()
    h5f.attrs[_C.ROOT_ATTR.GEOTRANSFORM] = tuple(geotransform.to_gdal())
    h5f.attrs[_C.ROOT_ATTR.WIDTH] = width
    h5f.attrs[_C.ROOT_ATTR.HEIGHT] = height
    h5f.attrs[_C.ROOT_ATTR.WINDOW_X] = px
    h5f.attrs[_C.ROOT_ATTR.WINDOW_Y] = py

    if verbose: print("writing...")

    ids = list(map(lambda xy : "x{:d}y{:d}".format(xy[0],xy[1]), windows))

    h5f.create_dataset("ids", data=np.array(ids, dtype='S'))
    h5f.close()

    return h5file_path


def create_semseg_dataset_from_scene(h5file_path, scene, window_size, window_stride, mask=None, preprocess=None,
                                     nodata=None, split=None, splitmode="x", verbose=False,
                                     mask_preprocess=None, chunks=None):
    """
    Create a segmentation HDF5 dataset from a satellite image scene.

    Arguments
    ---------
    :param h5file_path: Path to HDF5 file to create.
    :param scene: Path to image scene
    :param window_size: Sliding window size, (# x pixels, # y pixels)
    :param window_stride: Sliding window stride (# x pixels, # y pixels)
    :param mask: Path to mask.  Mask is assumed to be a 1-band raster source with categorical labels.
    :param preprocess: Pre-processing function.  The function should take a chunk of the input scene, with size
        given by *window_size* and return an output image
    :param mask_preprocess: Mask pre-processing function.  The function should take a chunk of the input mask, with shape
        (window_size, window_size, number_classes) and return a single mask.
        given by *window_size*, and return a single array corresponding
    :param image_set_names: List of strings specifying the name of each dataset.  Should have length equal to the number
        of outputs given by *preprocess*.  Default is ['set0','set1','set2',...]
    :param nodata: Raster no-data value to use.  Defaults to the no-data value in *scene*.
    :param split: Split in N datasets, by specifing N comma-separated values to be used as proportions of the
        total number of samples. If OUTPUT is output.hdf5, this will be named output0.hdf5, output1.hdf5, etc.
    :param splitmode: `x` (default) means scene will be split like this:\n
        +-------------+
        |    test     |
        |             |
        |-------------|
        | validation  |
        |             |
        |-------------|
        |  training   |
        |             |
        +-------------+
        and `y` splits along vertical axis:\n
        +-------------+---------------+-----------+
        |    test     |  validation   |  training |
        |             |               |           |
        +-------------+---------------+-----------+
        Only matters when *mask* is given.

    Returns
    -------
    :return: Either a single path to the newly created HDF5 file, or if multiple files were created, returns multiple
        paths.
    """

    img_arr, coordinate_sys, geotransform, tif_no_data_value, height, width = read_scene(scene)

    if nodata is None:
        nodata = tif_no_data_value
    if nodata is None:
        nodata = 0  # fallback

    if not isinstance(window_size, tuple) or not isinstance(window_stride, tuple):
        raise TypeError("window_size and window_stride must be tuples, (x,y)")

    px, py = window_size

    if preprocess is None:
        preprocess = lambda x: x

    if mask_preprocess is None:
        mask_preprocess = lambda x : x


    if mask is not None:
        mask_arr, _, mask_geotransform, mask_nodata, _, _ = read_scene(mask)

        # First, align the mask and scene in pixel-space as best as possible to make the top-right corners line up.
        pixel_transform = (~geotransform)*mask_geotransform
        pixel_transform = np.array([pixel_transform[i] for i in range(6)]).reshape(2,3)
        if not np.all(np.isclose(pixel_transform[:,:2],np.eye(2,2), atol=1e-8)):
            raise ValueError("Pixels in scene and mask must have the same resolution and orientation.")
        delta = pixel_transform[:, 2].round(0).astype(np.int)
        d_px, d_py = delta

        new_offset = [0,0]
        if d_px > 0:
            img_arr = img_arr[d_px:, :, :]
            new_offset[0] = d_px
        elif d_px < 0:
            mask_arr = mask_arr[-d_px:, :, :]

        if d_py > 0:
            img_arr = img_arr[:, d_py:, :]
            new_offset[1] = d_py
        elif d_py < 0:
            mask_arr = mask_arr[:, -d_py:, :]
        new_offset = geotransform*new_offset
        geotransform = rasterio.Affine(geotransform[0], geotransform[1], new_offset[0], geotransform[3],
                                       geotransform[4], new_offset[1])

        width = min(img_arr.shape[0], mask_arr.shape[0])
        height = min(img_arr.shape[1], mask_arr.shape[1])

        img_arr = img_arr[:width, :height, :]
        mask_arr = mask_arr[:width, :height, :]

        # Bitmask to check for no-data.  Bitmask[x,y] is True if pixel (x,y) is valid in both the image and mask.
        valid_mask = np.concatenate((img_arr != nodata, mask_arr != mask_nodata), axis=-1)
        valid_mask = np.logical_and.reduce(valid_mask, axis=-1)
    else:
        valid_mask = (img_arr == nodata)
        valid_mask = np.logical_and.reduce(valid_mask, axis=-1)

    window_groups = calculate_windows(valid_mask, window_size, window_stride, splitmode, split)
    if split is None:
        window_groups = [window_groups]

    sample_window = window_groups[0][0]
    sample_chip = img_arr[sample_window[0]:sample_window[0] + px, sample_window[1]:sample_window[1] + py, :]
    sample_output = preprocess(sample_chip)
    output_size = sample_output.shape

    if mask is not None:
        sample_mask = mask_arr[sample_window[0]:sample_window[0] + px, sample_window[1]:sample_window[1] + py, :]
        sample_mask = mask_preprocess(sample_mask)
        mask_output_size = sample_mask.shape


    if len(window_groups) > 1:
        filename_template = os.path.splitext(h5file_path)[0] + "{IDX:d}.hdf5"
        h5filenames = [filename_template.format(IDX=i) for i in range(len(window_groups))]
    else:
        h5filenames = [h5file_path]


    if (chunks is None and mask is not None) or chunks:
        scene_chunks = (1,) + output_size
        mask_chunks = (1,) + mask_output_size
    else:
        scene_chunks = None
        mask_chunks = None

    for fname, windows in zip(h5filenames, window_groups):
        h5f = h5py.File(fname, mode="w")
        if verbose: print(f"writing output file {h5f.filename}...")

        h5f.create_dataset('images', shape=(len(windows),) + output_size, dtype=np.float32, chunks=scene_chunks)

        if mask is not None:
            h5f.create_dataset("labels", shape=(len(windows),) + mask_output_size, dtype=np.bool, chunks=mask_chunks)

        h5f.attrs[_C.ROOT_ATTR.CRS] = coordinate_sys.to_string()
        h5f.attrs[_C.ROOT_ATTR.GEOTRANSFORM] = tuple(geotransform.to_gdal())
        h5f.attrs[_C.ROOT_ATTR.WIDTH] = width
        h5f.attrs[_C.ROOT_ATTR.HEIGHT] = height
        h5f.attrs[_C.ROOT_ATTR.WINDOW_X] = px
        h5f.attrs[_C.ROOT_ATTR.WINDOW_Y] = py

        ids = list()

        for k, pos in enumerate(tqdm(windows, disable=not verbose)):
            if (k + 1) % 1000 == 0:
                h5f.flush()
            x, y = pos


            h5f['images'][k] = preprocess(img_arr[x:x + px, y:y + py, :])

            if mask is not None:
                h5f['labels'][k] = mask_preprocess(mask_arr[x:x + px, y:y + py, :])

            ids.append("x{:d}y{:d}".format(x, y))

        h5f.create_dataset("ids", data=np.array(ids, dtype='S'))

        h5f.close()

    return h5filenames


def create_sample_heatmap(hdf5_path, raster_path):
    """
    Create a `sampling' heatmap of a HDF5 dataset where each pixel represents number of times it occurs in the
    dataset.   Valid for scene datasets only.

    Arguments
    --------
    :param hdf5_path:  Path to input HDF5 dataset.
    :param raster_path: Output path
    """
    h5f, close = _handle_hdf5filename_arg(hdf5_path, 'r')
    windows, crs, trans, height, width, (px, py) = get_scene_information(h5f)
    if close:
        h5f.close()

    counts = np.zeros((width, height), dtype=np.uint16)
    for (x, y) in windows:
        counts[x:x + px, y:y + py] += 1

    with rasterio.open(raster_path, "w", driver="GTiff", height=height, width=width, count=1,
                       crs=crs, dtype=np.uint16, transform=trans) as tif:
        tif.write(counts.T, 1)

    return raster_path


def create_extent_outline(hdf5_path, shapefile_path):
    h5f, close = _handle_hdf5filename_arg(hdf5_path, 'r')
    windows, crs, transform, height, width, (px, py) = get_scene_information(h5f)
    if close:
        h5f.close()

    mask = np.zeros((width, height), dtype=np.bool)
    for (x, y) in windows:
        mask[x:x + px, y:y + py] = True

    mask, connected_regions = label(mask, connectivity=1, return_num=True)

    schema = {
        "geometry" : "Polygon",
        "properties" : collections.OrderedDict()
    }

    shp = fiona.open(shapefile_path, mode='w', driver="Shapefile", crs=crs, schema=schema)

    mask, connected_regions = label(mask, connectivity=1, return_num=True)
    for i in range(1, connected_regions+1):
        # line is currently in matrix coordinates
        line = clockwise_walk(mask==i)
        line = [transform*(x,y) for x,y in line]
        shp.write({
            "geometry" : {
                "type" : "Polygon",
                "coordinates" : [line]
            },
            "properties" : collections.OrderedDict()
        })

    shp.close()

    return shapefile_path


def create_raster_mask(hdf5_path, raster_path, classes=None, nodata=255):
    """
    Create a raster scene mask from a labelled HDF5 dataset.
    Arguments
    --------
    :param hdf5_path: Path to input HDF5 dataset.
    :param raster_path: Path to output raster.
    :param classes: A dict which maps a class index (ranging from 0 to (#classes-1)) to an integer.
        Default is the identity map.  Note that each pixel in the output mask may only have one class.  For multi-label
         problems, you should create multiple masks.  Passing an OrderedDict will prioritise the labels according to
         the order given by the OrderedDict.
    :param nodata: No-data value for the output raster.  Default is 255.
    """
    h5f, close = _handle_hdf5filename_arg(hdf5_path)
    windows, crs, trans, height, width, window_size = get_scene_information(h5f)

    mask_arr = np.full((width, height), nodata, dtype=np.uint8)
    px, py = window_size

    if 'labels' in h5f.keys():
        if classes is None:
            numclasses = h5f['labels'].shape[1]
            classes = collections.OrderedDict((i, i) for i in range(numclasses))

        for k, (x, y) in enumerate(windows):
            window_labels = h5f['labels'][k]
            for c in classes:
                if window_labels[c]:
                    mask_arr[x:x + px, y:y + py] = classes[c]
                    break

    else:
        h5f.close()
        raise Exception("no labels found in {}".format(hdf5_path))

    with rasterio.open(raster_path, "w", driver="GTiff", height=height, width=width, count=1,
                       crs=crs, dtype=np.uint16, transform=trans, nodata=nodata) as tif:
        tif.write(mask_arr.T.astype(np.uint16), 1)

    if close:
        h5f.close()

    return raster_path


def compute_image_statistics(hdf5_path, band_axis, batch=1000, overwrite=False):
    """
    Compute mean and standard deviation of each band for an image set and add them to a HDF5 file.  Does not actually
    perform any normalisation, call :func:`normalise_images` for this.

    Arguments
    ---------
    :param hdf5_path: Path to source HDF5 file.
    :param band_axis: Which axis is the channels or band axis in the "images" dataset.
    :param batch: Batch size to use (for limiting memory usage only, does not affect result).
    :param overwrite: Whether to overwrite existing stats in an image set (default is False).
    """

    h5f, close = _handle_hdf5filename_arg(hdf5_path, 'r+')

    if not overwrite and ('mean' in h5f['images'].attrs or 'std' in h5f['images'].attrs):
        raise Exception("`images` already contains statistics")

    assert 0 < band_axis < len(h5f['images'].shape), "band_axis is out of range"
    ax = tuple(i for i in range(len(h5f['images'].shape)) if i != band_axis)

    stats = RunningStatistics()

    for i in range(int(np.ceil(h5f['images'].shape[0] / batch))):
        X = h5f['images'][i * batch:(i + 1) * batch]
        stats.update(X, axis=ax)

    h5f['images'].attrs['mean'] = tuple(stats.mean())
    h5f['images'].attrs['std'] = tuple(stats.std())
    h5f.flush()
    if close:
        h5f.close()

def compute_label_frequencies(hdf5filename, label_axis, batch=1000, overwrite=False):
    h5f, close = _handle_hdf5filename_arg(hdf5filename, 'r+')

    if not overwrite and 'freq' in h5f['labels'].attrs:
        raise Exception("`labels` already contains frequencies")
    if 'labels' not in h5f:
        raise Exception(f"{h5f.filename} does not contain labels.")

    label_cnts = np.zeros(shape=h5f['labels'].shape[label_axis], dtype=np.uint)
    assert label_axis < len(h5f['labels'].shape), "label_axis is out of range"
    ax = tuple(i for i in range(len(h5f['images'].shape)) if i != label_axis)

    for i in range(int(np.ceil(h5f['labels'].shape[0] / batch))):
        X = h5f['labels'][i * batch:(i + 1) * batch]
        label_cnts += X.sum(axis=ax).astype(np.uint)

    n = float(np.prod([d for i,d in enumerate(h5f['labels'].shape) if i != label_axis]))
    label_freq = label_cnts / n
    h5f['labels'].attrs['freq'] = tuple(label_freq)
    h5f.flush()
    if close:
        h5f.close()


def normalise_images(hdf5_path, band_axis, batch=1000, stats_from=None):
    """
    Normalise an image set to mean 0, std 1 using the mean and std stored in the image set.  Stats must be present, see
    :func:`compute_image_statistics`.

    Arguments
    ---------
    :param hdf5_path: Path to source HDF5 dataset.
    :param band_axis: Which axis of the `images` array is the bands dimension.
    :param image_sets: Which image sets for normalise (default is all).
    :param batch: Batch size to use (for limiting memory usage only, does not affect result).
    :param stats_from: Use stats from another HDF5 file.

    """
    h5f, close = _handle_hdf5filename_arg(hdf5_path, 'r+')

    if stats_from is not None:
        stats_from, close_starts_from = _handle_hdf5filename_arg(stats_from, 'r')
    else:
        stats_from = h5f
        close_starts_from = False

    if not ('mean' in stats_from['images'].attrs and 'std' in stats_from['images'].attrs):
        raise Exception("images are missing statistics")

    mu = np.array(stats_from['images'].attrs['mean'])
    sig = np.array(stats_from['images'].attrs['std'])

    if close_starts_from:
        stats_from.close()

    shp = [1 for _ in range(len(h5f['images'].shape))]
    shp[band_axis] = len(mu)

    mu = mu.reshape(shp)
    sig = sig.reshape(shp)

    for i in range(int(np.ceil(h5f['images'].shape[0] / batch))):
        X = h5f['images'][i * batch:(i + 1) * batch]
        X -= mu
        X /= sig
        h5f['images'][i * batch:(i + 1) * batch] = X

    h5f.flush()

    if close:
        h5f.close()



def validate_schema(hdf5_path, print_msgs=False, raise_errors=False):
    h5, close = _handle_hdf5filename_arg(hdf5_path, 'r')
    errors = []
    warnings = []
    datasets = ('labels', 'ids', 'images')


    data_lengths = dict()

    for k in h5:
        if k in datasets:
            if not isinstance(h5[k], h5py.Dataset):
                errors.append(f"`{k}` is not a HDF5 Dataset.")
            data_lengths[k] = h5[k].shape[0]

    if len(set(data_lengths.values())) > 1:
        msg = "mismatched number of samples:\n"
        for k,v in data_lengths.items():
            msg += f"\t{k:<20s} : {v:>10,d}\n"
        errors.append(msg[:-1])

    att = set(h5.attrs.keys())
    geo_keys = set(_C.ROOT_ATTR._GEO)
    if 'labels' not in h5:
        for gk in geo_keys:
            if gk not in h5.attrs:
                warnings.append("unlabelled dataset but geo-ref key `{geokey}` is missing.")

    missing_geokeys = geo_keys - att
    if 0 < len(missing_geokeys):
        errors.append("incomplete geo-ref information - missing keys: "+",".join(missing_geokeys))

    other_keys = set(_C.ROOT_ATTR._ALL) - geo_keys
    missing_other_keys = other_keys - att
    if 0 < len(missing_other_keys):
        errors.append("missing attributes: "+",".join(missing_other_keys))

    if close:
        h5.close()

    if raise_errors and len(errors) > 0:
        for w in warnings:
            print(f"WARNING: {w}")
        msg = "HDF5 schema validation errors/warnings found:\nWARNING: "
        msg += "\nWARNING: ".join(errors)
        msg += '\n'
        msg += "\nERROR: ".join(errors)
        raise Exception(msg)

    elif print_msgs:
        for w in warnings:
            print(f"WARNING: {w}")
        for e in errors:
            print(f"ERROR: {e}")


    return errors, warnings


def print_info(hdf5_path):
    """Pretty-print information about a HDF5 dataset."""
    f, close = _handle_hdf5filename_arg(hdf5_path, 'r')
    print(os.path.basename(hdf5_path))
    if 'images' in f.keys():
        print("/images: {} images of size ".format(f['images'].shape[0]) + " x ".join(map(str, f['images'].shape[1:])))
        for stat in ['mean', 'std']:
            if stat in f['images'].attrs:
                print("\t" + stat + ": " + " ".join(map(lambda x: "{:.4f}".format(x), f['images'].attrs[stat])))

    if 'labels' in f.keys():
        print("/labels: {} labels of size ".format(f['labels'].shape[0]) + " x ".join(map(str, f['labels'].shape[1:])))
        if 'freq' in f['labels'].attrs:
            print("\tfreq: " + " ".join(map(lambda x: "{:.4f}".format(x), f['labels'].attrs['freq'])))

    if 'ids' in f.keys():
        print("/ids: {} IDs with maximum length {}".format(f['ids'].shape[0], f['ids'].dtype.itemsize))

    scene_info = ""
    for atr in _C.ROOT_ATTR._ALL:
        if atr in f.attrs:
            if atr == "geotransform":
                scene_info += "\t" + atr + ":  " + "  ".join("{:.8e}".format(x) for x in f.attrs[atr]) + "\n"
            else:
                scene_info += "\t" + atr + ": " + str(f.attrs[atr]) + "\n"
    if len(scene_info) > 0:
        print("Scene information:\n" + scene_info)
    else:
        print()

    if close:
        f.close()


