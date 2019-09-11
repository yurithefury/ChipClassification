"""
Because Keras is a steaming pile of shit and doesn't let you dynamically set which device to use, the hack around it
is to modify environment variables before importing ANY keras crap.  This means anything to do with keras has to live in
this file, which has to be explicitly imported are calling utils.gpu_setup().
"""

import io

import keras
import time
import keras.backend as K
from . import *
from keras.models import load_model
from keras.layers import *
from keras.utils import plot_model, multi_gpu_model

from collections import defaultdict, Iterable, OrderedDict



def soft_sample_f2_loss(y_true, y_score):
    num = 5*K.batch_dot(y_true, y_score)
    denom = 4*K.sum(y_true, axis=-1) + K.sum(y_score, axis=-1)
    return -K.mean(num/denom)


def log_soft_sample_f2_loss(y_true, y_score):
    num = 5*K.batch_dot(y_true, y_score)
    denom = 4*K.sum(y_true, axis=-1) + K.sum(y_score, axis=-1)
    return -K.log(K.mean(num/denom))


def balanced_accuracy_score(ytrue, ypred):
    return K.mean(K.sum(ytrue*ypred, axis=0) / (K.epsilon() + K.sum(ytrue, axis=0)))


class CSVLogger(keras.callbacks.Callback):
    """Callback that streams epoch results to a csv file.

    Supports all values that can be represented as a string,
    including 1D iterables such as np.ndarray.

    # Example

    ```python
    csv_logger = CSVLogger('training.log')
    model.fit(X_train, Y_train, callbacks=[csv_logger])
    ```

    # Arguments
        filename: filename of the csv file, e.g. 'run/log.csv'.
        separator: string used to separate elements in the csv file.
        append: True: append if file exists (useful for continuing
            training). False: overwrite existing file,
    """

    def __init__(self, filename, separator=',', append=False):
        self.sep = separator
        self.filename = filename
        self.append = append
        self.writer = None
        self.keys = None
        self.append_header = True
        self.file_flags = ''
        self._open_args = {'newline': '\n'}
        super(CSVLogger, self).__init__()

    def on_train_begin(self, logs=None):
        if self.append:
            if os.path.exists(self.filename):
                with open(self.filename, 'r' + self.file_flags) as f:
                    self.append_header = not bool(len(f.readline()))
            mode = 'a'
        else:
            mode = 'w'
        self.csv_file = io.open(self.filename,
                                mode + self.file_flags,
                                **self._open_args)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, Iterable) and not is_zero_dim_ndarray:
                return '"[%s]"' % (', '.join(map(str, k)))
            else:
                return k

        if self.keys is None:
            self.keys = sorted(logs.keys())

        if self.model.stop_training:
            # We set NA so that csv parsers do not fail for this last epoch.
            logs = dict([(k, logs[k] if k in logs else 'NA') for k in self.keys])

        if not self.writer:
            class CustomDialect(csv.excel):
                delimiter = self.sep
            fieldnames = ['epoch'] + self.keys

            self.writer = csv.DictWriter(self.csv_file,
                                         fieldnames=fieldnames,
                                         dialect=CustomDialect)
            if self.append_header:
                self.writer.writeheader()

        row_dict = OrderedDict({'epoch': epoch+1})
        row_dict.update((key, handle_value(logs[key])) for key in self.keys)
        self.writer.writerow(row_dict)
        self.csv_file.flush()

    def on_train_end(self, logs=None):
        self.csv_file.close()
        self.writer = None


class ModelCheckpoint(keras.callbacks.ModelCheckpoint):
    def __init__(self, target_model, filepath, **kwargs):
        super().__init__(filepath, **kwargs)
        self.target_model = target_model

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.target_model.save_weights(filepath, overwrite=True)
                        else:
                            self.target_model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.target_model.save_weights(filepath, overwrite=True)
                else:
                    self.target_model.save(filepath, overwrite=True)


# Keras callback to log stuff to a csv at the end of every epoch
class ExtraMetrics(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.start_time = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()


    def on_epoch_end(self, epoch, logs=None):
        logs['time'] = time.time() - self.epoch_start_time


class HDF5Matrix:
    """Representation of HDF5 dataset to be used instead of a Numpy array.  Very similar to the Keras version, with
    additions and bugfixes.

    # Example

    ```python
        x_data = HDF5Matrix('input/file.hdf5', 'data')
        model.predict(x_data)
    ```

    Providing `start` and `end` allows use of a slice of the dataset.

    Optionally, a normalizer function (or lambda) can be given. This will
    be called on every slice of data retrieved.

    # Arguments
        datapath: string, path to a HDF5 file
        dataset: string, name of the HDF5 dataset in the file specified
            in datapath
        start_sample: int, start of desired slice of the specified dataset
        end_sample: int, end of desired slice of the specified dataset
        normalizer: function to be called on data when retrieved
        restrict: dict, mapping axis index to index, slice or list.  Data will be restricted to these the indexed data.
            E.g. If the HDF5 dataset is of shape (200,4,4,3) and `restrict = {3: [1,2]} then the new shape will be
            (200,4,4,2), similar to slicing a numpy array by [:, :, :, [1,2]].  For index 0 use start_sample, end_sample.

    # Returns
        An array-like HDF5 dataset.
    """

    refs = defaultdict(int)

    def __init__(self, datapath, dataset, start=0, end=None, normalizer=None, restrict=dict()):

        if h5py is None:
            raise ImportError('The use of HDF5Matrix requires '
                              'HDF5 and h5py installed.')

        if datapath not in list(self.refs.keys()):
            f = h5py.File(datapath, mode='r')
            self.refs[datapath] = f
        else:
            f = self.refs[datapath]
        self.data = f[dataset]
        self.start = start
        if end is None:
            self.end = self.data.shape[0]
        else:
            self.end = end
        self.normalizer = normalizer
        if self.normalizer is not None:
            first_val = self.normalizer(self.data[0:1])
        else:
            first_val = self.data[0:1]
        self._base_shape = first_val.shape[1:]
        self._base_dtype = first_val.dtype

        self._indexed_shape = list(self.data.shape)
        self._axis_index = [np.s_[:]] * (len(self.data.shape))

        if 0 in restrict:
            raise IndexError("Restricting 0 index not allowed.")

        for k,v in restrict.items():
            self._indexed_shape[k] = len(v)
            self._axis_index[k] = v

        self._indexed_shape = tuple(self._indexed_shape)
        self._axis_index = tuple(self._axis_index)


    def __len__(self):
        return self.end - self.start


    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop = key.start, key.stop
            if start is None:
                start = 0
            if stop is None:
                stop = self.shape[0]
            if stop + self.start <= self.end:
                idx = slice(start + self.start, stop + self.start)
            else:
                raise IndexError
        elif isinstance(key, (int, np.integer)):
            if key + self.start < self.end:
                idx = key + self.start
            else:
                raise IndexError
        elif isinstance(key, np.ndarray):
            if np.max(key) + self.start < self.end:
                idx = (self.start + key).tolist()
            else:
                raise IndexError
        else:
            # Assume list/iterable
            if max(key) + self.start < self.end:
                idx = [x + self.start for x in key]
            else:
                raise IndexError

        if self.normalizer is not None:
            return self.normalizer(self.data[idx])[self._axis_index]
        else:
            return self.data[idx][self._axis_index]


    @property
    def shape(self):
        """Gets a numpy-style shape tuple giving the dataset dimensions.

        # Returns
            A numpy-style shape tuple.
        """
        return (self.end - self.start,) + self._indexed_shape[1:]

    @property
    def dtype(self):
        """Gets the datatype of the dataset.

        # Returns
            A numpy dtype string.
        """
        return self._base_dtype

    @property
    def ndim(self):
        """Gets the number of dimensions (rank) of the dataset.

        # Returns
            An integer denoting the number of dimensions (rank) of the dataset.
        """
        return self.data.ndim

    @property
    def size(self):
        """Gets the total dataset size (number of elements).

        # Returns
            An integer denoting the number of elements in the dataset.
        """
        return np.prod(self._indexed_shape)



# Build model
def build_model_default(input_tensor, classes, output_activation, filters=32, kernel=(4,4)):

    x = BatchNormalization()(input_tensor)
    x = Conv2D(filters, kernel_size=kernel, padding="same", activation="relu")(x)
    x = Conv2D(filters, kernel_size=kernel, activation="relu")(x)
    x = MaxPooling2D(pool_size=kernel)(x)
    x = Dropout(0.25)(x)

    filters *= 2
    kernel = tuple(k//2 for k in kernel)

    x = Conv2D(filters, kernel_size=kernel, padding="same", activation="relu")(x)
    x = Conv2D(filters, kernel_size=kernel, activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    filters *= 2

    x = Conv2D(filters, kernel_size=kernel, padding="same", activation="relu")(x)
    x = Conv2D(filters, kernel_size=kernel, activation="relu")(x)
    x = MaxPooling2D(pool_size=kernel)(x)
    x = Dropout(0.25)(x)

    filters  *= 2

    x = Conv2D(filters, kernel_size=kernel, padding="same", activation="relu")(x)
    x = Conv2D(filters, kernel_size=kernel, activation="relu")(x)
    x = MaxPooling2D(pool_size=kernel)(x)
    x = Dropout(0.25)(x)

    filters *= 2

    x = Flatten()(x)
    x = Dense(filters, activation="relu")(x)
    x = Dropout(0.25)(x)
    output = Dense(classes, activation=output_activation)(x)
    return output

# Build model
def build_model_sugarbyte(input_tensor, classes, output_activation, filters=32, kernel=(4,4), dilation=1, kernel_shrink=1,
                          depth = 4):
    assert depth > kernel_shrink and depth > 0
    x = input_tensor
    for block_idx in range(depth):
        if kernel_shrink == block_idx:
            kernel = tuple(k // 2 for k in kernel)
        if block_idx > depth//2 - 1:
            dilation = 1
        u = Conv2D(filters, kernel_size=kernel, padding="same", activation=None, use_bias=False, dilation_rate=dilation)(x)
        u = BatchNormalization()(u)
        u = ReLU()(u)
        u = Conv2D(filters, kernel_size=kernel, padding="same", activation=None, use_bias=False, dilation_rate=dilation)(u)
        u = BatchNormalization()(u)
        u = ReLU()(u)
        x = MaxPooling2D(pool_size=kernel, padding='same')(u)
        filters *= 2

    x = Flatten()(x)
    x = Dense(filters, activation=None)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    output = Dense(classes, activation=output_activation)(x)
    return output


# Build model
def build_model_shallow(input_tensor, classes, output_activation):
    x = BatchNormalization()(input_tensor)
    x = Conv2D(64, kernel_size=(16, 16), padding="same", activation="relu")(x)
    x = Conv2D(64, kernel_size=(16, 16), activation="relu")(x)
    x = MaxPooling2D(pool_size=(4, 4))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(128, kernel_size=(8, 8), padding="same", activation="relu")(x)
    x = Conv2D(128, kernel_size=(8, 8), activation="relu")(x)
    x = MaxPooling2D(pool_size=(4, 4))(x)
    x = Dropout(0.25)(x)

    x = Flatten()(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.25)(x)
    output = Dense(classes, activation=output_activation)(x)
    return output

def build_model_super_shallow(input_tensor, classes, output_activation):
    x = BatchNormalization()(input_tensor)
    x = Conv2D(16, kernel_size=(16, 16), strides=(2,2), padding="same", activation="relu")(x)
    x = Conv2D(16, kernel_size=(16, 16), activation="relu")(x)
    x = MaxPooling2D(pool_size=(4, 4))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(32, kernel_size=(8, 8), padding="same", activation="relu")(x)
    x = Conv2D(32, kernel_size=(8, 8), activation="relu")(x)
    x = MaxPooling2D(pool_size=(4, 4))(x)
    x = Dropout(0.25)(x)

    x = Flatten()(x)
    x = Dense(16, activation="relu")(x)
    x = Dropout(0.25)(x)
    output = Dense(classes, activation=output_activation)(x)
    return output


def build_model_densenet(input_tensor, classes, output_activation):
    densenet_model = keras.applications.DenseNet201(input_tensor=input_tensor, classes=classes,
                                 weights=None, include_top=False, pooling='avg')
    out = densenet_model.output
    out = Dense(classes, activation=output_activation)(out)
    return out


def build_model_resnet(input_tensor, classes, output_activation):
    resnet_model = keras.applications.ResNet50(input_tensor=input_tensor, classes=classes,
                            weights=None, include_top=False, pooling='avg')
    out = resnet_model.output
    out = Dense(classes, activation=output_activation)(out)
    return out


# Give model a name here.
_ARCHITECTURES = {
    'super_shallow' : build_model_shallow,
    'shallow' : build_model_shallow,
    'sugarbyte' : build_model_sugarbyte,
    'default' : build_model_default,
    'densenet' : build_model_densenet,
    'resnet' : build_model_resnet
}
