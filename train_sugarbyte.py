import time
from utils import *
from yu import *
import glob
import sys


def soft_fbeta(y_true, y_score, beta=2):
    return np.mean((1 + beta ** 2) * np.sum(y_true * y_score, axis=1) / (
                beta * beta * np.sum(y_true, axis=1) + np.sum(y_score, axis=1)))


def train(params, gpus=None, small_dataset=False, load=None):
    start_time = time.time()
    master_device, num_gpus = gpu_setup(gpus)
    import utils.cnn
    import tensorflow as tf

    label_map = CC.LABELS.CATEGORICAL.CLOUD
    print("Training on labels:\n" + "\n".join(label_map.keys_sorted_by_value()) + "\n")

    # band_indices is an array of integers which specifiy which bands are used as inputs.  For example,
    # if you simply want to use all bands and you have 4 bands in your HDF5, set this to [0,1,2,3].
    band_indices = sorted(CC.CH.values())

    # likewise, label_indices is an array of integers specifying which labels you are actually using from you
    # HDF5 file.  If you just want to use all labels, you can set label_indices = slice(None)
    label_indices = sorted(label_map.values())

    # Hyperparameters
    num_classes = len(label_indices)
    batch_size = params.batch_size
    augment = get_augmentation_func(x_axis=1, y_axis=2)
    epochs = params.epochs
    epoch_start_idx = 0
    # Use only a small dataset when debugging
    if small_dataset:
        sample_end_index = 1024
    else:
        sample_end_index = None

    # Load data
    h5f = params.train_h5.file
    y_train = h5f["labels"][:sample_end_index, label_indices]
    h5_filename = h5f.filename
    h5_dataset_name = params.train_h5.name
    h5f.close()
    x_train = utils.cnn.HDF5Matrix(h5_filename, h5_dataset_name, end=sample_end_index, restrict={3: band_indices},
                                   normalizer=augment)

    h5f = params.validate_h5.file
    y_validate = h5f["labels"][:sample_end_index, label_indices]
    h5_filename = h5f.filename
    h5_dataset_name = params.validate_h5.name
    h5f.close()
    x_validate = utils.cnn.HDF5Matrix(h5_filename, h5_dataset_name, end=sample_end_index, restrict={3: band_indices})

    res = x_train.shape[1:3]

    # Weight classes -- some BBBAAAAAAD keras documentation here - class_weights parameter in model.fit assumes a
    # multi-class problem, not a multi-label one.  Since Keras simply turns class_weights into sample_weights, we can
    # do this manually.
    class_counts = np.sum(y_train, axis=0)
    num_samples = y_train.shape[0]
    # sum of class weights across classes is the number of classes, just like with the `default` weighting of 1
    class_weights = (num_samples - class_counts) * num_classes / (num_samples * (num_classes - 1))
    sample_weights = np.sum(y_train * class_weights, axis=1)
    print(' Class counts: ' + " ".join("{:4d}".format(int(x)) for x in class_counts))
    print('Class weights: ' + " ".join("{:.2f}".format(x) for x in class_weights))
    print(f'Sample weight range: ({sample_weights.min():.3f}, {sample_weights.max():.3f})')

    # Directories and files
    pth = lambda x: os.path.join(params.directory, x)
    best_model_file = pth("model_best.hdf5")
    history_file = pth("history.csv")

    # Build/load model
    with tf.device(master_device):
        if load is not None: # If continuing from previous checkpoint
            print("Loading model weights from: {}".format(load))
            model_template = utils.cnn.load_model(load, compile=False)
            epoch_start_idx = int(os.path.basename(load).split(".")[0].split("_")[-1])
            h = pd.read_csv(history_file, index_col=0)
            best_val_loss = h['val_loss'].min()
            starting_lr = h.tail(1)['lr'].values[0]

        else:
            # Here we define the model architecture using a function that must be defined in utils.cnn
            input_layer = utils.cnn.Input((res[0], res[1], len(band_indices)))
            model_template = utils.cnn.build_model_sugarbyte(input_layer, num_classes, "softmax",
                                                             filters=params.filters,
                                                             kernel=(params.kernel_size, params.kernel_size),
                                                             depth=params.depth,
                                                             dilation=params.dilation,
                                                             kernel_shrink=params.kernel_shrink
                                                             )
            model_template = utils.cnn.keras.Model(inputs=input_layer, outputs=model_template)
            best_val_loss = np.inf
            starting_lr = 0.1

    if num_gpus > 1:
        model = utils.cnn.multi_gpu_model(model_template, gpus=num_gpus)
    else:
        model = model_template

    # Here we set the loss function and optimiser.  Note that utils.cnn.balanced_accuracy_score is a custom metric, so
    # see the source code on how to create a custom loss/metric function in Keras
    opt = utils.cnn.keras.optimizers.Adam(lr=starting_lr)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=[utils.cnn.balanced_accuracy_score])

    try:
        utils.cnn.plot_model(model_template, show_layer_names=True, show_shapes=True)
    except ImportError:
        pass

    lr_schedule = utils.cnn.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                                              factor=0.5,
                                                              patience=8,
                                                              min_lr=0.00005,
                                                              min_delta=0.0001,
                                                              mode='min')
    # Note that some Callbacks are imported from utils.cnn.keras and some from utils.cnn, the latter being custom
    # ones (often slight variations of Keras callbacks which fix bugs)
    epoch_checkpoints = utils.cnn.ModelCheckpoint(model_template, pth("model_{epoch:04d}.hdf5"), verbose=1)
    best_checkpoint = utils.cnn.ModelCheckpoint(model_template, best_model_file, verbose=1, save_best_only=True,
                                                mode="min", monitor="val_loss")
    best_checkpoint.best = best_val_loss
    history_logger = utils.cnn.CSVLogger(history_file, append=(load is not None))
    timer = utils.cnn.ExtraMetrics()

    # Train the model
    model.fit(x_train, y_train,
              validation_data=(x_validate, y_validate),
              initial_epoch=epoch_start_idx,
              epochs=epochs,
              batch_size=batch_size,
              shuffle="batch",
              sample_weight=sample_weights,
              callbacks=[lr_schedule, epoch_checkpoints, best_checkpoint, timer, history_logger]
              )

    print("\nTotal time for this job: {:,d} seconds".format(int(time.time() - start_time)))

    # The train() function returns the best validation loss and accuracy.  This values are written to a file and used
    # in Bayesian optimisation.
    h = pd.read_csv(history_logger.filename, index_col=0)
    best_val_loss = h['val_loss'].min()
    best_acc = h['val_balanced_accuracy_score'].max()
    return best_val_loss, best_acc


def locate_previous_epochs(directory):
    """
    Determine the last epoch by finding a previous checkpoint file in a model directory.  Used for continuing training.
    """
    last_epoch = -1
    last_checkpoint = None
    if not os.path.exists(os.path.join(directory, "history.csv")):
        return None
    for f in glob.glob(f"{directory}/model_*.hdf5"):
        num = f.split('_')[-1].rstrip('.hdf5')
        try:
            num = int(num)
            if num > last_epoch:
                last_epoch = num
                last_checkpoint = f
        except ValueError:
            continue

    return last_checkpoint


if __name__ == '__main__':
    import h5py

    h5_train = h5py.File('data/proc/cloud_train.hdf5', 'r')
    h5_validate = h5py.File('data/proc/cloud_val.hdf5', 'r')

    if len(sys.argv) > 1:
        # This is for Bayesian optimisation - the single argument is the job directory where parameters for this
        # job are given, in a file called "params.json"
        pth = sys.argv[1]
        last_checkpoint = locate_previous_epochs(pth)
        params = bo.SugarbyteParams.from_json(os.path.join(pth, "params.json"))
        params.train_h5 = h5_train['images']
        params.validate_h5 = h5_validate['images']
        params.directory = pth
        params.validate(assertion=True, ignore=['test_h5'])
        loss, acc = train(params, load=last_checkpoint)

    else:  # default debug args (train a single model)
        print("{} : (debugging)".format(__file__))
        debugdir = "data/models/debug"
        # last_checkpoint = locate_previous_epochs(debugdir)
        last_checkpoint = None
        # shutil.rmtree(debugdir, ignore_errors=True)
        os.makedirs(debugdir, exist_ok=True)

        # Here
        params = bo.SugarbyteParams(
            batch_size=8,
            directory=debugdir,
            kernel_size=3,
            kernel_shrink=-1,
            dilation=1,
            depth=3,
            filters=8,
            epochs=7,
            train_h5=h5_train['images'],
            validate_h5=h5_validate['images'],
            test_h5=None
        )

        params.validate()
        # loss = train(params, small_dataset=True)
        loss, acc = train(params, load=last_checkpoint, small_dataset=True)

    np.savetxt(os.path.join(params.directory, "result.txt"), np.array([loss]))
    np.savetxt(os.path.join(params.directory, "acc.txt"), np.array([acc]))
