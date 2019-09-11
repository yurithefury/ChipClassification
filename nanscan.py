import h5py
import sys
import numpy as np
from tqdm import trange

def discover_datasets(h5, _datasets=tuple()):
    if isinstance(h5, h5py.File) or isinstance(h5, h5py.Group):
        for item in h5:
            _datasets = _datasets + discover_datasets(h5[item])
        return _datasets
    else:
        return (h5.name,)


def scan_dataset(d):
    bad_idxs = []
    if not np.issubdtype(d.dtype, np.number):
        return None
    for i in trange(d.shape[0], leave=False):
        if np.any(np.isnan(d[i])):
            bad_idxs.append(i)
    ndims = len(d.shape)
    bad_idxs = map(lambda x : "[ " +", ".join([str(x)] + [":"]*(ndims-1)) + " ]", bad_idxs)
    return list(bad_idxs)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("usage: nanscan.py HDF5_FILE\nRecursively scans a HDF5 file for NaN values.")
        quit(1)
    h5 = h5py.File(sys.argv[1], 'r')
    datasets = discover_datasets(h5)
    dataset_name_text_width = max(map(len, datasets)) + 4
    bad_idxs = dict()
    for d in datasets:
        print(d + " " * (dataset_name_text_width - len(d)), end='')
        bad_idxs = scan_dataset(h5[d])
        if bad_idxs is None:
            print("N/A - non-numeric datatype.")
        elif len(bad_idxs) == 0:
            print("ok.")
        else:
            print("Found bad indices at:")
            for idx in bad_idxs:
                print(" " * (dataset_name_text_width + 2) + idx)
