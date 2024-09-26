# Installing TensorFlow 2.10 with Conda

Adapted from https://gretel.ai/blog/install-tensorflow-with-cuda-cdnn-and-gpu-support-in-4-easy-steps

To install TensorFlow 2.10 with GPU support using Conda, follow these steps:

1. Create a new Conda environment:

    ```shell
    conda create -n tf2_10 python=3.9
    conda activate tf2_10
    conda install -c conda-forge cudatoolkit=11.2.2 cudnn=8.1.0
 
    mkdir -p $CONDA_PREFIX/etc/conda/activate.d
    echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
    ```

2. and sign back in via SSH or close and re-open your terminal window. Reactivate your conda session and install Tensorflow.

    ```shell
    conda activate tf2_10
    python3 -m pip install tensorflow==2.10 numpy==1.24
    ```

4. Verify that TensorFlow 2.10 is installed correctly with GPU support:

    ```shell
    python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
    ```

    This command should output `2.10` if TensorFlow is installed successfully.


Remember to activate the Conda environment (`conda activate tf2_10`) every time you want to use TensorFlow 2.10.
