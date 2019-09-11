"""
These classes are used to organise the hyperparameters for hyperparameter tuning.
"""
import json
import h5py

class BaseParams():
    def __init__(self,
                 batch_size=None,
                 kernel_size=None,
                 filters=None,
                 epochs=None,
                 train_h5=None,
                 validate_h5=None,
                 test_h5=None,
                 directory=None):

        self.batch_size = int(batch_size) if batch_size is not None else None
        self.kernel_size = int(kernel_size) if kernel_size is not None else None
        self.filters = int(filters) if filters is not None else None
        self.epochs = int(epochs) if epochs is not None else None

        self._int_attr = ['batch_size', 'kernel_size', 'filters', 'epochs']

        self.train_h5 = train_h5
        self.validate_h5 = validate_h5
        self.test_h5 = test_h5

        self._hdf5_dataset_attr = ['train_h5', 'validate_h5', 'test_h5']

        self._float_attr = []

        self.directory = directory

    def validate(self, assertion=False, ignore=None):
        bad = dict()
        for int_property in self._int_attr:
            x = getattr(self, int_property)
            if type(x) is not int:
                bad[int_property] = "must be integer"

        for float_property in self._float_attr:
            x = getattr(self, float_property)
            if type(x) is not float:
                bad[float_property] = "must be float"

        for name in self._hdf5_dataset_attr:
            h5 = getattr(self, name)
            if h5 is None:
                bad[name] = "is None."
            else:
                if not isinstance(h5, h5py.Dataset):
                    bad[name] = "{} is not a h5py.Dataset".format(str(h5))

        if ignore is None:
            ignore = []
        elif type(ignore) is str:
            ignore = [ignore]

        for name in ignore:
            if name in bad:
                del bad[name]

        if assertion and len(bad) > 0:
            msg = "validation errors:\n"
            msg += "\n".join("{}:{}".format(n, m) for n, m in bad.items())
            raise Exception(msg)

        return bad

    @classmethod
    def from_json(cls, data):
        if isinstance(data, str):
            try:
                d = json.loads(data)
            except json.JSONDecodeError:
                # data is a path
                with open(data, 'r') as f:
                    d = json.load(f)

        else:  # assume data is a file pointer
            d = json.load(data)

        try:
            return cls(**d)
        except TypeError:
            raise Exception('bad JSON.')

    def to_json(self, path=None):
        d = self.__dict__.copy()
        for h5 in self._hdf5_dataset_attr:
            if h5 in d:
                del d[h5]

        other_ignores = ('directory')

        for k in list(d.keys()):
            if k.startswith('_') or k in other_ignores:
                del d[k]

        if path is not None:
            with open(path, 'w') as f:
                json.dump(d, f, indent='\t')
            return path
        else:
            return json.dumps(d)


class Params(BaseParams):
    def __init__(self, lr_alpha=None, lr_beta=None, **kwargs):
        super().__init__(**kwargs)
        self.lr_alpha = float(lr_alpha) if lr_alpha is not None else None
        self.lr_beta = float(lr_beta) if lr_beta is not None else None
        # For checking and input validation, add floats to the _float_attr list
        self._float_attr.extend(['lr_alpha', 'lr_beta'])


class SugarbyteParams(BaseParams):
    def __init__(self, kernel_shrink=None, depth=None, dilation=None, **kwargs):
        super().__init__(**kwargs)
        self.kernel_shrink = kernel_shrink
        self.depth = depth
        self.dilation = dilation
        # For checking and input validation, add ints to the _int_attr list
        self._int_attr.extend(['kernel_shrink', 'depth', 'dilation'])

