FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3

# Tiberius only: TensorFlow, Tiberius and its Python packages. The tools of
# the Paludamentum evidence pipeline (miniprot, HISAT2, StringTie,
# TransDecoder, ...) are in Paludamentum's own image.
#
# Build from the root of a Tiberius checkout; the image holds exactly that
# checkout (COPY, see .dockerignore), tagged with its version:
#   docker build -t gaiusaugustus/tiberius:<version> .

USER root

# Record the Python packages of the NGC base image. The NGC TensorFlow build
# (2.17.0+nv25.2, CUDA 12.8, cuDNN 9, all SMs) is the only one in this image that
# runs on Blackwell GPUs (sm_120, e.g. RTX PRO 6000). The check at the end of
# this file fails the build if a later pip install replaced it or added PyPI
# CUDA/cuDNN wheels next to it.
RUN python3 -c "import importlib.metadata as m; print('\n'.join(sorted(d.metadata['Name'].lower().replace('_', '-') + '==' + d.version for d in m.distributions())))" > /opt/ngc-base-packages.txt

ENV TF_USE_LEGACY_KERAS=0
RUN python3 -m pip install --no-cache-dir "keras==3.15.1"

# Not `pip install .[from_source]`: hidten[tensorflow] requires
# tensorflow[and-cuda], and the and-cuda extra of the NGC wheel pins PyPI
# CUDA 12.3 / cuDNN 8.9 / ptxas 12.3. These get installed on top of the NGC
# CUDA 12.8 stack and crash Tiberius on Blackwell GPUs. So the TF-dependent
# packages are installed without dependencies and the rest explicitly.
# numpy, pandas, requests, rich and pyyaml are the versions of the NGC base.
RUN python3 -m pip install --no-cache-dir --no-deps \
        "bricks2marble==0.1.1" \
        "hidten==1.0.0" && \
    python3 -m pip install --no-cache-dir \
        "numpy==1.26.4" \
        "pandas==2.2.3" \
        "requests==2.32.3" \
        "rich==13.9.4" \
        "pyyaml==6.0.2" \
        "pydantic==2.13.5" \
        "biopython==1.88" \
        "packaging==26.3"

COPY . /opt/Tiberius
RUN cd /opt/Tiberius && \
    python3 -m pip install --no-cache-dir --no-deps . && \
    chmod +x tiberius.py tiberius/*.py && \
    mkdir -p model_weights && chmod -R 777 model_weights

ENV PATH=${PATH}:/opt/Tiberius/tiberius/
ENV PATH=${PATH}:/opt/Tiberius/

# Fail the build if the NGC TensorFlow/CUDA stack was modified (see top of file)
# or Tiberius does not start
RUN python3 -c "import importlib.metadata as m, sys; \
base = set(open('/opt/ngc-base-packages.txt').read().split()); \
now = {d.metadata['Name'].lower().replace('_', '-') + '==' + d.version for d in m.distributions()}; \
added = sorted(p for p in now - base if p.startswith(('nvidia-', 'tensorflow'))); \
tf = m.version('tensorflow'); \
sys.exit(f'NGC TensorFlow stack modified: tensorflow=={tf}, added/changed: {added}' if '+nv' not in tf or added else 0)" && \
    python3 -c "import bricks2marble, hidten, Bio, yaml, rich, requests, packaging" && \
    tiberius.py --help > /dev/null && \
    python3 -c "import importlib.metadata as m, tensorflow as tf; b = tf.sysconfig.get_build_info(); print('tiberius', m.version('tiberius'), 'bricks2marble', m.version('bricks2marble'), 'hidten', m.version('hidten'), 'keras', m.version('keras'), 'TensorFlow', tf.__version__, 'CUDA', b['cuda_version'], 'cuDNN', b['cudnn_version'])"


USER ${NB_UID}
