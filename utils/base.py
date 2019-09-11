"""
This file contains all sorts of miscellaneous functions and bits.
"""


from skimage.transform import resize

import rasterio
import numpy as np
import csv

import random
import warnings
from tqdm import tqdm, trange
from sklearn.metrics import fbeta_score
from sklearn.exceptions import UndefinedMetricWarning
from math import ceil
import re
import os
import pandas as pd
import yaml
import h5py


class CloudClassifyError(BaseException):
    pass

class BiDict(dict):
    """
    Bijective Dictionary.  Functions like a dictionary but values as well as keys must be unique.  Main feature is fast
    access to the inverse dictionary, bidict.inv.

    >>> bd = BiDict(); bd['a'] = 1
    >>> bd.inv[1]

    """

    def __init__(self, *args, **kwargs):
        super(BiDict, self).__init__(*args, **kwargs)
        self.inv = dict()
        assert len(set(self.keys())) == len(set(self.values()))
        for key, value in self.items():
            self.inv[value] = key


    def __setitem__(self, key, value):
        if key in self:
            if self[key] == value:
                return
            else:
                assert value not in self.inv
                del self.inv[self[key]]
        self.inv[value] = key
        super(BiDict, self).__setitem__(key, value)


    def __delitem__(self, key):
        del self.inv[self[key]]
        super(BiDict, self).__delitem__(key)

    def update(self, other):
        for k,v in other.items():
            self[k] = v
        return self

    def select(self,*keys):
        """
        Return a new BiDict which is a subset of this one, indexed by `keys`.

        >>> bd = BiDict({'egg' : 3, 'bacon' : 6, 'tree' : 0})
        >>> bd.select('egg','tree')
        BiDict({'egg' : 3, 'tree' : 0})
        """
        new_bidict = BiDict()
        for k in keys:
            new_bidict[k] = self[k]
        return new_bidict

    def to_dict(self):
        return dict(self.items())


    def copy(self):
        return BiDict(super().copy())


    def keys_sorted_by_value(self, **kwargs):
        """
        Convience method to get the keys as a list, sorted by values.  Useful with numerical values.

        """
        return list(self.inv[v] for v in sorted(self.values(), **kwargs))

# Static class to hold global constants
class CC:
    SENTINEL2_BAND = BiDict({"B": 2, "G": 3, "R": 4, "NIR": 8})

    # Global TIF band index reference
    CH = BiDict(**{
        "R"    : 0,
        "G"    : 1,
        "B"    : 2,
        "NIR"  : 3,
        "NDVI" : 4,
        "GNDVI": 5
    })

    # Conversion labels from Amazon to Tropics data
    AMAZON_TO_TROPICS_CONV = {
        "clear"        : "clear",
        "cloudy"       : "cloudy",
        "partly_cloudy": "partly_cloudy",
        "agriculture"  : "agriculture",
        "bare_ground"  : "bare_ground",
        "blow_down"    : "lodged_crop",
        "cultivation"  : "agriculture",
        "habitation"   : "habitation",
        "haze"         : "haze",
        "primary"      : "forest",
        "water"        : "water"
    }

    # This is where labels are defined.  Each has a unique (across all labels) sequential integer index, starting at 0.
    class LABELS:
        """
        LABELS has the following structure:
        LABELS
          +-CATEGORICAL
          |   +-CLOUD
          |   +-SHADOW
          |   +-GROUPS
          |   +-ALL
          +-BINARY
          +-ALL
          +-ABBREV
        """


        # Categorical labels belong in here
        class CATEGORICAL:
            CLOUD = BiDict(**{
                "clear"        : 0,
                "haze"         : 1,
                "partly_cloudy": 2,
                "cloudy"       : 3
            })

            SHADOW = BiDict(**{
                "unshaded"     : 4,
                "partly_shaded": 5,
                "shaded"       : 6
            })


            # Make sure to add new categories to here too.
            GROUPS = [CLOUD, SHADOW]

            ALL = BiDict()
            for bd in GROUPS:
                ALL.update(bd)

        # Binary variables live here
        BINARY = BiDict(**{
            "agriculture": 7,
            "bare_ground": 8,
            "habitation" : 9,
            "forest"     : 10,
            "water"      : 11
        })

        ALL = BiDict()
        ALL.update(BINARY)
        ALL.update(CATEGORICAL.ALL)

        # Short names for labels
        ABBREV = BiDict(**{
            "clear"        : "Cr",
            "cloudy"       : "Cl",
            "partly_cloudy": "Pc",
            "agriculture"  : "A",
            "bare_ground"  : "B",
            "lodged_crop"  : "L",
            "habitation"   : "Ha",
            "haze"         : "Hz",
            "forest"       : "F",
            "water"        : "W",
            "unshaded"     : "U",
            "shaded"       : "S",
            "partly_shaded": "Ps"
        })

        @staticmethod
        def get_group_by_name(string):
            """
            Arguments
            --------
            :param string: Any of {"shadow", "cloud", "binary", "landcover"} (not case sensitive)
            :return: label map
            :rtype: BiDict
            """
            mapping = {
                "shadow" : CC.LABELS.CATEGORICAL.SHADOW,
                "cloud" : CC.LABELS.CATEGORICAL.CLOUD,
                "binary" : CC.LABELS.BINARY,
                "landcover" : CC.LABELS.BINARY,
            }
            return mapping[string.lower()]

class Config(object):
    """
    Class for storing and validating directory and computing configuration.  To create a config, insantiate with a
    dictionary or a path to YAML file with the dictionary.  The dictionary should contain the following keys:

    Required:
    `'hdf5_train_set'` : Path the HDF5 file to use for training/application

    Optional:
    `'hdf5_test_set'` :  Path the HDF5 file to use for testing
    `'log_file'` : Path to log file
    `'model_directory'` : Path to directory in which to save models to.

    """

    config_file_keys = ['hdf5_train_set', 'hdf5_test_set', 'training_log', 'model_directory']

    def __init__(self, config_file):
        config = {key : None for key in Config.config_file_keys}
        self.filename = None

        if isinstance(config_file, dict):
            config.update(config_file)
        else:
            self.filename = config_file
            with open(config_file, 'r') as f:
                config.update(yaml.load(f))

        self.train_data = config['hdf5_train_set']
        self.test_data = config['hdf5_test_set']
        self.log = config['training_log']
        self.models_dir = config['model_directory']



    def verify(self):
        no_check = lambda *x: True

        checks = [
            (check_write, str, self.models_dir, "Models directory"),
            (check_read, str, self.train_data, "HDF5 train data"),
            (check_read, str, self.test_data, "HDF5 test data"),
            (check_write, str, self.log, "Log file")
        ]

        for validation, vartype, attr, name in checks:
            if type(attr) != vartype:
                print("Invalid config: {} must be {}".format(name, str(vartype).strip("<").strip(">")))
                return False
            if not validation(attr):
                return False

        print("Using config: {}".format(self.filename))
        return True


    @staticmethod
    def create_empty_config_file():
        """
        Create an example config file in the current working directory.
        """
        d = { k :'' for k in Config.config_file_keys}
        if os.path.exists('config.yaml'):
            raise Exception('config.yaml exists.')
        else:
            with open('config.yaml', 'w') as f:
                yaml.dump(d,f, default_flow_style=False)



class DataSequence(object):
    """
    DataSequence essentially acts as a list of batches.  Use len(DataSequence) to determine the total number of batches
    in the sequence.  Access a particular batch by indexing DataSequence[i].  Batches will be consistent.  Each batch
    consists of a tuple (X_train, y_train) and will only be loaded into memory upon indexing the sequence.

    Arguments
    ---------
    :param iterable[str] files: Image filenames including path
    :param int res: resolution to resize images to (preferrably base-2)
    :param list bands[int]: Image bands to use (elements must be referenced in CC.CH)
    :param int batch_size: size of the batches the generator should return

    Optional Arguments
    ------------------
    :param dict[str,ndarray] filenamemap: A dict mapping filename to binary vector representation of labels.  If not
        supplied then the training labels will not be available.
    :param function preprocessing: Preprocessing to be applied (eg standardisation).

    Returns
    -------
    :return: A DataSequence object which can be indexed by the batch number.
    :rtype: DataSequence
    """

    def __init__(self, files, res, bands, batch_size, filenamemap=None, preprocessing=None):
        self.filelist = files
        self.batch_size = batch_size
        self.filenamemap = filenamemap
        self.res = res
        self.bands = bands
        self.preprocess = preprocessing
        if filenamemap is None:
            self.num_classes = None
        else:
            self.num_classes = filenamemap.values()[0].shape[0]

    def __len__(self):
        return int(ceil(len(self.filelist) / float(self.batch_size)))

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError('index out of range')
        files_to_load = self.get_filenames(idx)

        # Get a batch of filenames
        X_train = np.zeros((len(files_to_load), self.res, self.res, len(self.bands)))
        if self.num_classes is not None:
            y_train = np.zeros((len(files_to_load), self.num_classes))
        else:
            y_train = None

        # Load them into memory
        for i, f in enumerate(files_to_load):
            img = load_image(f, self.res, self.bands)
            if self.preprocess is not None:
                img = self.preprocess(img)
            X_train[i, :, :, :] = img
            if self.num_classes is not None:
                y_train[i, :] = self.filenamemap[f]

        return (X_train, y_train)

    def get_filenames(self, idx):
        """
        Get the filenames associated with a batch index.

        Arguments
        ---------
        :param int idx: Batch index

        Returns
        -------
        :return: List of filenames belonging to batch idx
        :rtype: list[str]
        """
        if idx >= len(self):
            raise IndexError('index out of range')
        return self.filelist[idx * self.batch_size:(idx + 1) * self.batch_size]


def get_augmentation_func(x_axis=1, y_axis=2, bands = np.s_[:]):
    """
    Get augmentation function based on location of x and y axes

    Optional Arguments
    ------------------
    :param int x_axis: The index of the x axis in the input matrix (defaults to 0)
    :param int y_axis: The index of the y axis in the input matrix (defaults to 1)

    Returns
    -------
    :return: augmentation function
    :rtype: function
    """
    augmentation_func = lambda x : apply_augmentation(x[:,:,:,bands], x_axis=x_axis, y_axis=y_axis)

    return augmentation_func


def get_chip_id_strings(h5f):
    """
    Retrieve the chip IDs stored in a h5f file, in the same order the data is stored.

    Arguments
    ----
    :param str h5f: Path to hdf5 file

    Returns
    ----
    :return: chip_ids
    :rtype: list
    """
    h5f = h5py.File(h5f, 'r')
    chip_names = h5f['chip_id'][()].tolist()
    if len(h5f['chip_id'].shape) > 1:
        chip_names = list(map(lambda x: "X_{:d}_Y_{:d}".format(x[0], x[1]), chip_names))
    else:
        chip_names = list(map(lambda x : x.decode(), chip_names))
    h5f.close()
    return chip_names


def apply_augmentation(matrix, k=None, x_axis=1, y_axis=2):
    """
    Apply augmentation to a matrix of images

    Arguments
    ---------
    :param ndarray matrix: Matrix containing pixel values.  First axis is assumed to be number of samples.

    Optional Arguments
    ------------------
    :param int x_axis: index of pixel x-axis
    :param int y_axis: index of pixel y-axis
    :param int band_axis: index of bands

    :param int k: Type of augmentation (defaults to randomly chosen) \n
        0-3 : rotate k*90 deg \n
        4-7 : flip horizontally, vertically, diagonally, antidiagonally (respectively)

    Returns
    -------
    :return: matrix containing pixel values, with same shape
    :rtype: ndarray
    """
    # Apply a random 90 degree rotation to an image (elements of the group D4)
    matrix = matrix.copy()
    for n in range(matrix.shape[0]):
        if k is None:
            k = random.randrange(8)
        else:
            assert k in range(8)
        # ROTATIONS
        if 1 <= k <= 3:
            matrix[n, None] = np.rot90(matrix[n, None], k, axes=(x_axis, y_axis))
        # FLIPS
        elif k == 4:
            matrix[n, None] = np.flip(matrix[n, None], x_axis)  # Horizontal
        elif k == 5:
            matrix[n, None] = np.flip(matrix[n, None], y_axis)  # Vertical
        elif k == 6:
            matrix[n, None] = np.flip(np.rot90(matrix[n, None], 1, axes=(x_axis, y_axis)), x_axis)  # diagonal
        elif k == 7:
            matrix[n, None] = np.rot90(np.flip(matrix[n, None], x_axis), 1, axes=(x_axis, y_axis))  # antidiagonal

    return matrix


def calculate_ndvi_band(img, R=CC.CH["R"], NIR=CC.CH["NIR"]):
    """
    Calculates the NDVI band.

    Arguments
    ---------
    :param ndarray img: Array containing pixel values with shape (res, res, bands).

    Optional Arguments
    ------------------
    :param int R: The index of the red band in img (e.g img[:,:,R] is the red band)
    :param int NIR: The index of the NIR band

    Returns
    -------
    :return: Array containing NDVI values with shape (res, res)
    :rtype: ndarray
    """
    denom = img[:, :, NIR] + img[:, :, R] + 0.0
    return (img[:, :, NIR] - img[:, :, R]) / denom


def calculate_gndvi_band(img, G=CC.CH["G"], NIR=CC.CH["NIR"]):
    """
    Calculates the Green-NDVI band.

    Arguments
    ---------
    :param ndarray img: Array containing pixel values with shape (res, res, bands).

    Optional Arguments
    ------------------
    :param int G: The index of the green band in img (e.g img[:,:,R] is the red band)
    :param int NIR: The index of the NIR band

    Returns
    -------
    :return: Array containing green NDVI values with shape (res, res)
    :rtype: ndarray
    """
    denom = img[:, :, NIR] + img[:, :, G] + 0.0
    return (img[:, :, NIR] - img[:, :, G]) / denom


def CL(s):
    """Execute a string as a shell command."""
    print(s)
    os.system(s)
    return


def check_read(path, quiet = False):
    """Check whether user has read access for *path*"""
    if not os.access(path, os.R_OK):
        if not quiet:
            print("Cannot read from: " + path +
                  "\nEither the path does not exist or you do not have read permissions.")
        return False
    return True


def check_write(path):
    """Check whether user has read **and** write access for *path*"""
    if not os.access(path, os.W_OK | os.R_OK):
        print("Cannot read from: {}\n"
              "Either the path does not exist or you do not have read/write permissions.".format(path))
        return False
    return True


def create_directory(root, dirname=None, ignore_existing=False):
    """
    Creates a new directory for a model

    Arguments
    ---------
    :param str root: The path to the parent directory
    :param str dirname: The name of the directory. Defaults to model_XXXXXXXX (random, unique)

    Optional Arguments
    ------------------
    :param bool ignore_existing: Only affects behaviour if id is not None. Suppresses exceptions if folder already
        exists.

    Returns
    -------
    :return: Path to the new directory
    :rtype: str
    """
    if dirname is None:
        getrandstr = lambda *x: "".join(str(random.randint(0, 9)) for _ in range(8))
        newdir = root + "/model_"
        randnumstr = getrandstr()
        while os.path.exists(newdir + randnumstr):
            randnumstr = getrandstr()
        newdir += randnumstr
    else:
        newdir = os.path.join(root, dirname)

    if os.path.exists(newdir):
        if ignore_existing:
            return newdir
        else:
            raise CloudClassifyError("{} already exists.".format(newdir))
    else:
        # Create directory
        print("Creating new directory: " + newdir)
        os.makedirs(newdir)
        return newdir


# def create_standardise_image_zeromean_population(files, res, bands, batch_size=2040, stats=None):
#     """
#     Creates a standardisation function which ensures the entire dataset has zero mean and unit standard deviation,
#     for each image band
#
#     Arguments
#     ---------
#     :param iterable[str] files: Paths to dataset files
#     :param int res: image resolution in pixels (res x res)
#     :param list[int] bands: Image bands to use (elements must be referenced in CC.CH)
#
#     Keyword Arguments
#     -----------------
#     :param int batch_size: Batch size to use when calculating statistics (use as big as you can fit into memory)
#     :param tuple stats: Manually supply dataset statistics. Elements should be ndarrays for each stat (mean, std) with
#         length = len(bands).  If this is supplied res, bands and batch_size will have no effect.
#
#     Returns
#     -------
#     :return: Standardisation function to be applied to a single image (ndarray).
#     :rtype: function
#     """
#     if stats is None:
#         mean, stdev, _ = get_dataset_statistics(files, res, bands, batch_size)
#     else:
#         mean, stdev = stats
#
#     def standardise_image_population_zeromean(img):
#         if any(stdev == 0):
#             raise ArithmeticError('divide by zero')
#         return (img - mean) / stdev
#
#     return standardise_image_population_zeromean


def delete(*files):
    """Deletes any files if they exist."""
    for f in files:
        if os.path.exists(f):
            os.remove(f)
    return


# def gen_filename_from_parameters(pars):
#     """
#     Create a safe filename from parameters.
#
#     Arguments
#     ---------
#     :param dict[str,any] pars: Model parameters
#
#     Returns
#     -------
#     :return: filename
#     :rtype: str
#     """
#     filename = ''
#     for k in sorted(pars.keys()):
#         x = pars[k]
#         t = type(x)
#         if t is NoneType:
#             s = "-"
#         elif t is BooleanType:
#             s = str(int(x))
#         elif t is StringType:
#             s = x
#         elif t is FloatType:
#             s = "{:.3E}".format(x)
#         elif t is IntType:
#             s = str(x)
#         elif t is FunctionType:
#             s = get_pretty_function_name(x)
#         else:
#             s = str(x)
#         filename += k.upper()[0:3] + s + "_"
#     return filename[0:-1]


def get_dataset_statistics(files, res, bands, batch_size=4096):
    """
    Calculates statistics on the whole dataset, loading batches into memory at a time
    See: http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf

    Arguments
    ---------
    :param iterable[str] files: Image filenames including path
    :param int res: resolution to resize images to (preferrably base-2)
    :param list[int] bands: Image bands to use (elements must be referenced in CC.CH)

    Optional Arguments
    ------------------
    :param int batch_size:  Batch size to use when calculating statistics (use as big as you can fit into memory)

    Returns
    -------
    :return: Arrays with mean, standard deviation for each band and total number of samples
    :rtype: (ndarray, ndarray, int)
    """
    numbands = len(bands)
    data = DataSequence(files, res, bands, batch_size)
    T_prev = np.zeros(numbands)  # total
    S_prev = np.zeros(numbands)  # total sum of squares
    n_prev = 0.0  # samples
    numsamples = len(files)

    print("Calculating dataset statistics on {} images...".format(numsamples))
    for batchno in tqdm(range(len(data))):
        X, _ = data[batchno]
        n_b = float(np.prod(X.shape[0:-1]))
        T_b = np.sum(X, axis=(0, 1, 2))
        S_b = np.sum(np.square(X - T_b / n_b), axis=(0, 1, 2))

        if batchno == 0:
            S_prev = S_b
            n_prev = n_b
            T_prev = T_b
        else:
            S_prev += S_b + (n_prev / (n_b * (n_b + n_prev))) * np.square((T_prev * n_b / n_prev) - T_b)
            T_prev += T_b
            n_prev += n_b

    mean = T_prev / n_prev
    std = np.sqrt(S_prev / (n_prev - 1))

    return (mean, std, n_prev)


def get_pretty_function_name(f, replace_blanks=True):
    """
    Get the name of a function as a string

    Arguments
    ---------
    :param function f: Function to get name of.

    Optional Arguments
    ------------------
    :param bool replace_blanks: Strip blanks and underscores.

    Returns
    -------
    :return: Formatted function name
    :rtype: str
    """
    s = re.sub(r" at .*>$", "", re.sub(r"^<function ", "", str(f)))
    if replace_blanks:
        s = s.replace("_", "").replace(" ", "")
    return s


def get_wkt_prj(epsg_code, cachedir = None):
    """
    Get the correct .prj file for a given EPSG code from http://spatialreference.org/.

    Arguments
    ---------
    :param int epsg_code: EPSG number

    Optional Arguments
    ------------------
    :param str cachedir: Directory to cache .prj files to.  Default is not to cache.

    Returns
    -------
    :return:  Contents of the .prj file
    :rtype: str
    """
    cachefile = "{}/{}.prj".format(cachedir,epsg_code)
    if cachedir and check_read(cachefile, quiet=True):
        with open(cachefile, "r") as f:
            output = f.read()
    else:
        import urllib
        wkt = urllib.urlopen("http://spatialreference.org/ref/epsg/{}/prettywkt/".format(epsg_code))
        output = re.sub(r"\s", "", wkt.read())
        if cachedir:
            with open(cachefile, "wb") as f:
                f.write(output)
    return output


def gpu_setup(gpus=None):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    if gpus is not None:
        if type(gpus) is int:
            num_gpus = gpus
            if num_gpus == 0:
                print("ignoring any gpu devices")
                gpus = "-1"
            else:
                gpus = ",".join(map(str, range(gpus)))
        elif type(gpus) is list:
            # User has selected specific PCI BUS IDs - eg 1,2,3
            num_gpus = len(gpus)
            gpus = ",".join(map(str, gpus))

        # Now mask the GPUs
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    else:
        # NOW we can import tensorflow
        from tensorflow.python.client.device_lib import list_local_devices
        num_gpus = len([x.name for x in list_local_devices() if x.device_type == "GPU"])

    # Which device to build the model on
    if num_gpus == 1:
        master_device = "/device:GPU:0"
    else:
        master_device = "/cpu:0"

    os.environ["KERAS_BACKEND"] = "tensorflow"
    return master_device, num_gpus



def evaluate_all_data(model, data_seq, labels):
    """
    Make predictions on every image in a data sequence.

    Arguments
    ---------
    :param model: Instance of model to run on, must be trained
    :param DataSequence data_seq: Instance of DataSequence to loop through.
    :param list[str] labels: Text labels matching the output of model.  For example, for a single image input,
        if *model* outputs a vector [0.9, 0.2, 0.3] predicting "clear", "cloudy" and "partly_cloudy" respectively,
        then *labels* should be ["clear", "cloudy", "partly_cloudy"]

    Returns
    -------
    :return: The predictions for each labels (pseudo-probabilities)
    :rtype: pandas.DataFrame
    """
    num_classes = model.get_layer(index=-1).output_shape[-1]
    assert num_classes == len(labels)
    y_pred = np.zeros((len(data_seq.filelist), num_classes))
    for i in trange(len(data_seq)):
        X, _ = data_seq.__getitem__(i)
        y_pred[i * data_seq.batch_size:(i + 1) * data_seq.batch_size] = model.predict_on_batch(X)

    return pd.DataFrame(y_pred,
                        index=pd.Index(map(lambda x: x.split("/")[-1], data_seq.filelist), name="name"),
                        columns=labels)

#
# def load_image_gdal(fpath, resize_res, bands):
#     """
#     Load an image from file.
#
#     Arguments
#     ---------
#     :param str fpath: name of file including path
#     :param int,tuple[int] resize_res: resolution to resize to [(xres,yres) or int].
#     :param list[int] bands: Image bands to use (elements must be referenced in CC.CH)
#
#     Returns
#     -------
#     :return: Array containing pixel values, with shape (px, py, band)
#     :rtype: numpy.ndarray
#     """
#     assert type(bands) == type(list())
#
#     img = gdal.Open(fpath)
#     if img is None:
#         raise CloudClassifyError("File not found: " + fpath)
#     img = img.ReadAsArray()
#     img = np.transpose(img, (1, 2, 0)).astype(float)  # Shuffle indices to [x,y,band] and normalise
#
#     img = fill_in_missing_bands(img, bands)
#
#     if isinstance(resize_res,tuple):
#         img = resize(img, resize_res, mode="reflect", preserve_range=True, anti_aliasing=True)
#     else:
#         if resize_res != img.shape[0]:
#             if img.shape[0] == 129 and img.shape[1] == 129 and "T55KCA" in fpath.split("/")[-1]:
#                 # Fix for bad T55KCA Sentinel-2 images
#                 img = img[1:,1:,:]
#                 assert not np.any(np.isnan(img)), "NaNs in bad T55KCA image"
#                 if resize_res != img.shape[0]:
#                     img = resize(img, (resize_res, resize_res), mode="reflect", preserve_range=True,anti_aliasing=True)
#             elif img.shape[0] == img.shape[1]:
#                 img = resize(img, (resize_res, resize_res), mode="reflect", preserve_range=True,anti_aliasing=True)
#             else:
#                 raise CloudClassifyError("Image size is bad: {} x {} pixels not good.\n\t Image: `{}`".format(
#                     img.shape[0], img.shape[1], fpath))
#     return img


def load_image(fpath, resize_res, bands):
    """
    Load an image from file.

    Arguments
    ---------
    :param str fpath: name of file including path
    :param tuple[int] resize_res: resolution to resize to (xres,yres).
    :param list[int] bands: Image band indices to use - elements must be referenced in CC.CH(.inv)

    Returns
    -------
    :return: Array containing pixel values, with shape (px, py, band)
    :rtype: numpy.ndarray
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", rasterio.errors.NotGeoreferencedWarning)
        with rasterio.open(fpath, 'r') as f:
            img = np.stack([f.read(i) for i in f.indexes], axis=-1)

    img = img.astype(float)  # Shuffle indices to [x,y,band] and normalise

    # Fix for bad T55KCA Sentinel-2 images
    if img.shape[:2] == (129, 129):
        fname = os.path.basename(fpath)
        if "S2" == fname[:2] and "T55KCA" in fname:
            img = img[1:,1:,:]
            assert not np.any(np.isnan(img)), "NaNs in bad T55KCA image"

    img = fill_in_missing_bands(img, bands)

    if resize_res != img.shape[:2]:
        img = resize(img, resize_res, mode="reflect", preserve_range=True, anti_aliasing=True)

    return img


def fill_in_missing_bands(img, bands):

    assert isinstance(bands,list)

    # Calculate additional bands
    added_bands = {
        # Name : [img_index, function]
        "NDVI" : [None, calculate_ndvi_band],
        "GNDVI": [None, calculate_gndvi_band]
    }

    for b in filter(lambda x: x not in ["R", "G", "B", "NIR"], CC.CH.keys_sorted_by_value()):
        if CC.CH[b] in bands:
            tmp = added_bands[b][1](img)
            tmp = np.reshape(tmp, (tmp.shape[0], tmp.shape[1], 1))
            img = np.concatenate((img, tmp), axis=2)
            added_bands[b][0] = img.shape[-1] - 1

    # Base bands (not calculated)
    base_bands = list(filter(lambda x: CC.CH.inv[x] in ["R", "G", "B", "NIR"], bands))
    bands = sorted(base_bands + [val[0] for val in added_bands.values() if val[0] is not None])
    img = img[:, :, bands]
    return img


def load_labels(csvfile, label_map, tifdir=None, to_dataframe=True):
    """
    Load image labels from file.

    Arguments
    ---------
    :param str csvfile: Filename of CSV file to use including path.
    :param dict[str,int] label_map: Label mapping to use (maps label_string -> bin_vector_index)

    Optional Arguments
    ------------------
    :param str tifdir: Path to directory containing the TIF files that are referenced by the CSV
    :param bool to_dataframe: Return a pandas DataFrame instead.  Index will be the filename and columns will be the
        labels with binary values.  Index and columns will both be sorted alphabetically.

    Returns
    -------
    :return:    dict which maps: TIF filename -> Binary vector of labels
                function which maps: Binary vector of labels -> Label text
    :rtype: (BiDict, function)|DataFrame
    """
    assert tifdir is not None or to_dataframe, "Either tifdir must be given or to_dataframe must be True."
    if to_dataframe:
        df = pd.read_csv(csvfile, index_col=0)
        df["tags"] = df["tags"].map(lambda x: x.split())
        for t in label_map:
            df[t] = df["tags"].map(lambda x: 1 if t in x else 0)
        df.drop(columns=["tags"], inplace=True)
        df.sort_index()
        df = df.reindex(label_map.keys_sorted_by_value(), axis=1)
        return df
    else:
        label_dict = dict()
        num_classes = len(set(label_map.values()))
        label_text = lambda bv: " ".join(sorted(list(label_map[i] for i in range(len(bv)) if bv[i] > 0.9)))

        # Import labels from CSV
        with open(csvfile) as f:
            for r in csv.DictReader(f):
                # Encode labels to binary (more than one element can be 1)
                bin_vec = np.zeros(num_classes, dtype="uint8")
                for l in r["tags"].split():
                    if l in label_map:
                        bin_vec[label_map[l]] = 1
                label_dict[tifdir + "/" + r["name"]] = bin_vec

        return label_dict, label_text


def load_validation_set(files, filenamemap, res, bands, preprocessing=None):
    """
    Loads a the validation set from file and applies any preprocessing.

    Arguments
    ---------
    :param iterable[str] files: Image filenames including path
    :param dict[str,ndarray] filenamemap: A dict mapping filename to binary vector representation of labels
    :param int res: resolution to resize images to (preferrably base-2)
    :param list[int] bands: Image bands to use (elements must be referenced in CC.CH)

    Optional Arguments
    ------------------
    :param function preprocessing: Preprocessing to be applied prior to any image augmentations (eg standardisation).

    Returns
    -------
    :returns: A tuple containing: \n
        X_valid, the validation inputs \n
        y_valid, the validation outputs
    :rtype: (ndarray,ndarray)

    """
    num_classes = filenamemap.values()[0].shape[0]
    X_valid = np.zeros((len(files), res, res, len(bands)))
    y_valid = np.zeros((len(files), num_classes))
    print("Loading validation set...")

    for i, f in enumerate(tqdm(files)):
        # Could potentially multi-thread this loop
        img = load_image(f, res, bands)
        if preprocessing is not None:
            img = preprocessing(img)
        X_valid[i, :, :, :] = img
        y_valid[i, :] = filenamemap[f]

    return X_valid, y_valid


def log_to_csv(lst, path):
    """
    Logs things to a CSV log file.  Will append to file rather than overwriting.

    Arguments
    ---------
    :param list|list[list] lst: list to log, or a list of lists with one list per row
    :param str path: path to file to write the log to
    """
    with open(path, "a") as csvfile:
        out = csv.writer(csvfile, delimiter=",", quoting=csv.QUOTE_NONNUMERIC)
        if type(lst[0]) is list:
            for l in lst:
                out.writerow(l)
        else:
            out.writerow(lst)
    return


def make_decisions(scores, thresholds=None, mutually_exclusive=[]):
    """
    Decision-making logic - given a score (typically between 0 and 1, though may be slightly < 0 or > 1), how do we
    determine whether a label should be present?

    Arguments
    ----
    :param numpy.ndarray scores: Scores ( 0-1 ) for labels, with samples along axis 0 and labels along axis 1
    :param numpy.ndarray,list[float] thresholds: Vector of thresholds for labels with the same label ordering as `scores`.
        Mutually exlusive entries should have NaN entries.
    :param list[list[int]] mutually_exclusive: list of lists where each list contains the label indices of a mutually
        exclusive group

    Returns
    ----
    :return: Binary predictions
    :rtype: numpy.ndarray
    """
    if thresholds is None:
        thresholds = [np.nan] * scores.shape[1]

    nan_thresh = set(i for i,t in enumerate(thresholds) if np.isnan(t))
    mut_ex = set()
    for l in mutually_exclusive:
        mut_ex |= set(l)
    assert mut_ex >= nan_thresh, "Labels have NaN thresholds only if they are in a mutually exclusive group"

    scores = scores.copy()
    # Of the mutually exclusive labels, simply pick the label with the highest score
    for label_map in mutually_exclusive:
        indices = sorted(list(label_map))
        scores = np.apply_along_axis(pick_one_prediction, 1, scores, indices=indices)

    # Apply thresholds to the other labels.
    for i,t in enumerate(thresholds):
        if not np.isnan(t):
            scores[:, i] = scores[:, i] > t
    return scores


# def _mosaic_label_tif_chip(tif_label):
#     tif = tif_label[0]
#     label = tif_label[1]
#     alpha_file = re.sub(r"\.TIF$", "_alpha.TIF", tif)
#     shp_file = re.sub(r"\.TIF$", ".shp", tif)
#     delete(alpha_file, shp_file)
#
#     # Add 5th band of zeros
#     os.system('gdalwarp -dstnodata 0 -dstalpha -of GTiff {} {}'.format(tif, alpha_file))
#
#     # Run polygonize
#     os.system("gdal_polygonize.py {} -b 5 -f 'ESRI Shapefile' {}".format(alpha_file, shp_file))
#     delete(alpha_file)
#
#     # Open a Shapefile, and get field names
#     source = ogr.Open(shp_file, update=True)
#     layer = source.GetLayer()
#     # layer_defn = layer.GetLayerDefn()
#     # field_names = [layer_defn.GetFieldDefn(i).GetName() for i in range(layer_defn.GetFieldCount())]
#
#     # Add a new field
#     # new_field = ogr.FieldDefn('Labels', ogr.OFTString)
#     layer.CreateField(ogr.FieldDefn('Labels', ogr.OFTString))
#     for i in layer:
#         # layer.SetFeature(i)
#         i.SetField("Labels", label)
#         layer.SetFeature(i)
#     del source  # close shapefile
#     return shp_file
#
#
# def mosaic(scenedir, cachedir = True, numworkers = 4, exclude = set()):
#     """
#     Mosiac together a scene of TIF chips and build fishnet overlay with feature labels.
#
#     Arguments
#     ---------
#     :param str scenedir: Directory path to the scene folder.
#
#     Optional Arguments
#     ------------------
#     :param str cachedir: Where to store cached projection settings.  Defaults to a **.cache** directory in the parent
#         directory of *scenedir*.  Set to *None* to disable cache.
#     :param bool verbose: GDAL command logging
#     :param int numworkers: Number of processes to spawn for multiprocessing GDAL stuff
#     :param set exclude: set of labels to exclude TIFs by
#     """
#     from multiprocessing import Pool
#     tif_files = glob.glob(scenedir+"/*.TIF")
#     rootdir = "/".join(scenedir.split("/")[0:-1])
#     shp_files = []
#     tif_labels = pd.read_csv(scenedir+"/labels.csv", index_col=0)
#     abbreviate = lambda tags : " ".join(CC.LABELS.ABBREV[t] for t in tags.split())
#     if cachedir and type(cachedir) is not str:
#         cachedir = create_directory(rootdir, "/.cache", ignore_existing=True)
#
#     tif_labels = map(lambda x : tif_labels.loc[x.split("/")[-1], 'tags'], tif_files)
#     abbrev_labels = map(abbreviate, tif_labels)
#     tif_labels = map(lambda x : set(x.split()), tif_labels)
#
#     # Multithread the bulk of the gdal stuff
#     _p = Pool(numworkers)
#     shp_files = _p.map(_mosaic_label_tif_chip, zip(tif_files, abbrev_labels))
#     del _p
#
#     scene_vector = scenedir + "/classified.shp"
#     scene_rastor = scenedir + "/classified.TIF"
#     scene_prj = scenedir + "/classified.prj"
#     delete(scene_vector, scene_rastor, scene_prj)
#
#     # make vector shapefile
#     w = shapefile.Writer()
#     for f in shp_files:
#       r = shapefile.Reader(f)
#       w._shapes.extend(r.shapes())
#       w.records.extend(r.records())
#     w.fields = list(r.fields)
#     w.save(scene_vector)
#
#     # Build virtual rastor
#     temp_rastor_list = scenedir + ".tmplist"
#
#     filtered_tifs = filter(lambda x : x[0].isdisjoint(exclude), zip(tif_labels, tif_files))
#     filtered_tifs = map(lambda x: x[1], filtered_tifs)
#
#     with open(temp_rastor_list, "w+") as f:
#         f.writelines(map(lambda x : x + "\n", filtered_tifs))
#     os.system("gdalbuildvrt -input_file_list {} {}".format(temp_rastor_list, scene_rastor))
#     delete(temp_rastor_list)
#
#     # Create projection file
#     with open(scene_prj, "w+") as prj:
#         prj.write(get_wkt_prj("32755", cachedir=cachedir))
#
#     # DriverName = "ESRI Shapefile"  # e.g.: GeoJSON, ESRI Shapefile
#
#     driver = ogr.GetDriverByName("ESRI Shapefile")
#     for f in shp_files:
#         if os.path.exists(f):
#             driver.DeleteDataSource(f)
#     return


def optimise_threshold(y_true, y, numit = 1000, end_sigma=0.025):
    """
    Optimise the prediction threshold for F-2 scores.

    Arguments
    ---------
    :param np.ndarray y_true: Ground truth labels (binary, 1-D)
    :param np.ndarray y: Model/Ensemble output (possibly after a softmax/sigmoid/regression step).  (float, 1-D)

    Optional Arguments
    ------------------
    :param int numit: Number of random-search iterations to do.
    :param float sigma: Std of the Gaussian distributions to draw from during random-search

    Returns
    -------
    :return: Threshold
    :rtype: float
    """
    f2 = lambda t: fbeta_score(y_true, y > t, 2)
    best_thresh = 0.2
    best_val = f2(best_thresh)
    decay = np.log10(0.25/end_sigma)/numit
    print('\nBeginning search...')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        for it in range(numit):
            sigma = 0.25*10**(-decay*numit)
            new_thresh = best_thresh + np.random.randn()*sigma
            new_val = f2(new_thresh)
            if new_val > best_val:
                print('it {:>5d} : val {:.16f} : thresh {:.16f}'.format(it, new_val,new_thresh))
                best_val = new_val
                best_thresh = new_thresh
            elif it % 100 == 0:
                print('it {:>5d} : val {:.16f} : thresh {:.16f}'.format(it, best_val, best_thresh))
    return float(best_thresh)


def print_warning(s):
    """Formats and prints a string as a warning"""
    print("\t\033[91m" + "[" + s.upper() + "]\033[0m")
    return


def read_file_to_list(filename):
    """Read each line in a file into a list of strings"""
    with open(filename, "r") as f:
        out = list(l.rstrip("\n") for l in f)
    return out

def save_results(fpath, y_pred, filenames, label_map):
    """
    Save predictions to file.

    Arguments
    ---------
    :param str fpath: path of file to save to
    :param ndarray y_pred: binary array of labels with shape (len(filenames), num_classes)
    :param iterable[str] filenames: TIF files (without path).
    :param dict[int,str] label_map: Mapping from binary vector index to label string.
    """
    assert y_pred.shape[0] == len(filenames)
    num_classes = y_pred.shape[1]
    with open(fpath, "w+") as f:
        for i, tif in enumerate(filenames):
            tags = []
            for j in range(num_classes):
                if y_pred[i, j] == 1:
                    tags.append(label_map[j])
                elif y_pred[i, j] == 0:
                    pass
                else:
                    raise CloudClassifyError("y_pred must only contain binary values, not {}".format(y_pred[i, j]))
            f.write("{},{}\n".format(tif, " ".join(tags)))
    return


def pick_one_prediction(vec, indices = None):
    """
    "Pick best" prediction.  Given a vector of floats, pick the largest element and set it to 1, and set all others to
    0.  Only count elements indexed by indices.

    Arguments
    ---------
    :param np.ndarray vec: Vector to apply to.

    Optional arguments
    :param np.ndarray indices: Indices of the elements of *vec* to consider.

    Returns
    -------
    :return: The new array
    :rtype: np.ndarray
    """

    v = vec.copy()
    if indices is None:
        indices = range(len(v))
    max_index = indices[np.argmax(v[indices])]
    v[indices] = 0
    v[max_index] = 1
    return v

def create_filename(f,ext=None):
    import re
    f = re.sub("\s+","_",str(f))
    f = re.sub("\W+","",f)
    if ext is not None:
        return "{}.{}".format(f,ext)
    else:
        return f

def categorical_crossentropy(y_true, y_pred, eps=1e-5):
    y_pred = np.clip(y_pred, eps, 1-eps)
    return -np.sum(y_true*np.log(y_pred))/y_true.shape[0]

def binary_crossentropy(y_true, y_pred, eps=1e-5):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred))