FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/tf-cu113.2-10.py310:latest

RUN apt-get update && apt-get install -y git

RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends \
    time && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade \
    pip setuptools wheel \
    "hatchling>=1.26" \
    "packaging>=24.0"

RUN pip install "pyfamsa<0.6.0"

RUN pip install pyBigWig bio scikit-learn biopython bcbio-gff requests

ENV PYTHONPATH="/opt/learnMSA/:${PYTHONPATH}"

RUN cd /opt && \
    git clone https://github.com/Gaius-Augustus/Tiberius && \
    cd  Tiberius && \
    pip install . && \
    chmod +x tiberius.py && \
    chmod +x tiberius/*py

RUN mkdir -p /opt/Tiberius/model_weights && chmod -R 777 /opt/Tiberius/model_weights

ENV PATH=${PATH}:/opt/Tiberius/tiberius/
ENV PATH=${PATH}:/opt/Tiberius/

USER ${NB_UID}