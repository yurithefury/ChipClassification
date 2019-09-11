echo "PYTHON=$(which python)"
echo "KERAS_BACKEND=${KERAS_BACKEND-[not set]}"

export TF_CPP_MIN_VLOG_LEVEL=1
cat <<EOF | python -
import tensorflow as tf
print(f'TENSORFLOW_VERSION={tf.VERSION}')
from keras import backend as K
from time import sleep
gpus = K.tensorflow_backend._get_available_gpus()
sleep(2)
if len(gpus) > 0:
    print("\nGPU devices found:")
    print("\n".join(gpus))
else:
    print("\nNo GPU devices found.")
EOF
