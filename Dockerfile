FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3

USER root

ENV TF_USE_LEGACY_KERAS=0
RUN python3 -m pip install --upgrade "keras>=3,<4"

RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends \
      time \
      git \
      libgsl-dev \
      libboost-all-dev \
      libsuitesparse-dev \
      liblpsolve55-dev \
      libsqlite3-dev \
      libmysql++-dev \
      libboost-iostreams-dev \
      zlib1g-dev \
      libbamtools-dev \
      samtools \
      libhts-dev \
      cdbfasta \
      diamond-aligner \
      libfile-which-perl \
      libparallel-forkmanager-perl \
      libyaml-perl \
      libdbd-mysql-perl \
      wget \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade \
    pip setuptools wheel \
    "hatchling>=1.26" \
    "packaging>=24.0"


RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
 && rm -rf /var/lib/apt/lists/*

RUN apt update && \
    apt install --yes --no-install-recommends hisat2  && \
    apt clean all

RUN python3 -m pip install "pyfamsa<0.6.0"
RUN python3 -m pip install pyyaml pyBigWig bio scikit-learn biopython bcbio-gff requests

RUN cd /opt && \
	git clone https://github.com/Gaius-Augustus/Augustus/ && \
	cd Augustus && \
	make clean && \
	make && \
	make install
ENV PATH=${PATH}:/opt/Augustus/bin/


RUN cd        /opt      && \
    git      clone        https://github.com/Gaius-Augustus/Tiberius && \
    cd Tiberius && \
    python3 -m pip install .[from_source] && \
    chmod +x tiberius.py && \
    chmod +x tiberius/scripts/* && \
    chmod +x tiberius/*py

RUN mkdir -p /opt/Tiberius/model_weights && chmod -R 777 /opt/Tiberius/model_weights

#COPY tensorflow*2.17*.whl /tmp/

#RUN  python3 -m pip install --upgrade \
#        nvidia-cudnn-cu12~=9.0 \
#        nvidia-nccl-cu12 \
#        nvidia-cuda-runtime-cu12~=12.8.0 \
#        nvidia-cuda-nvcc-cu12~=12.8.0 \
#        nvidia-cusparse-cu12 \
#        nvidia-cufft-cu12 \
#        nvidia-cusolver-cu12 \
#        /tmp/tensorflow-*whl

#RUN rm /tmp/*

ENV PATH=${PATH}:/opt/Tiberius/tiberius/
ENV PATH=${PATH}:/opt/Tiberius/
ENV PATH=${PATH}:/opt/Tiberius/tiberius/scripts

RUN apt update && \
    apt install -yq bamtools && \
    apt clean all



RUN python3 -m pip install --upgrade \
    pip setuptools wheel \
    "hatchling>=1.26" \
    "packaging>=24.0"



RUN  cd /opt && \
     git clone https://github.com/TransDecoder/TransDecoder
ENV PATH=${PATH}:/opt/TransDecoder/util

RUN  cd /opt && \
     git clone https://github.com/tomasbruna/miniprothint
ENV PATH=${PATH}:/opt/miniprothint

RUN cd /opt && \
    git clone https://github.com/lh3/miniprot && \
    cd miniprot && make
ENV PATH=${PATH}:/opt/miniprot

RUN  cd /opt && \
     git clone https://github.com/tomasbruna/miniprot-boundary-scorer && \
     cd miniprot-boundary-scorer && make
ENV PATH=${PATH}:/opt/miniprot-boundary-scorer


#RUN cd /opt && \
#        wget -O hisat2.zip https://cloud.biohpc.swmed.edu/index.php/s/oTtGWbWjaxsQ2Ho/download && \
#        unzip hisat2.zip

#ENV PATH=${PATH}:/opt/hisat2-2.2.1/


RUN cd /opt && \
    wget https://github.com/lh3/minimap2/releases/download/v2.30/minimap2-2.30_x64-linux.tar.bz2 && \
    tar -jxvf minimap2-2.30_x64-linux.tar.bz2
ENV PATH=${PATH}:/opt/minimap2-2.30_x64-linux

RUN cd /opt && \
    wget https://github.com/gpertea/stringtie/releases/download/v3.0.3/stringtie-3.0.3.Linux_x86_64.tar.gz && \
    tar xzf stringtie-3.0.3.Linux_x86_64.tar.gz
ENV PATH=${PATH}:/opt/stringtie-3.0.3.Linux_x86_64

RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends \
      perl \
      libdb-dev \
      zlib1g-dev \
      curl \
      cpanminus \
      ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install cpanm + Perl modules (URI includes URI::Escape)
RUN cd /opt && \
    curl -fsSL https://cpanmin.us | perl - App::cpanminus && \
    cpanm --notest DB_File && \
    cpanm --notest URI

# Download + unpack TransDecoder
RUN cd /opt && \
    wget -q https://github.com/TransDecoder/TransDecoder/archive/refs/tags/TransDecoder-v5.7.1.tar.gz && \
    tar -xzf TransDecoder-v5.7.1.tar.gz && \
    rm -f TransDecoder-v5.7.1.tar.gz

ENV PATH="/opt/TransDecoder-TransDecoder-v5.7.1:/opt/TransDecoder-TransDecoder-v5.7.1/util:${PATH}"

RUN cd /opt && \
    wget https://github.com/bbuchfink/diamond/releases/download/v2.1.16/diamond-linux64.tar.gz && \
    tar xzf diamond-linux64.tar.gz
ENV PATH=${PATH}:/opt/diamond/

RUN cd /opt && \
    mkdir bedtools && \
    cd bedtools && \
    wget https://github.com/arq5x/bedtools2/releases/download/v2.30.0/bedtools.static.binary && \
    mv bedtools.static.binary bedtools && \
    chmod a+x bedtools
ENV PATH=${PATH}:/opt/bedtools/

RUN cd /opt && \
    wget https://ftp-trace.ncbi.nlm.nih.gov/sra/sdk/3.3.0/sratoolkit.3.3.0-ubuntu64.tar.gz && \
    tar xzf sratoolkit.3.3.0-ubuntu64.tar.gz
ENV PATH=${PATH}:/opt/sratoolkit.3.3.0-ubuntu64/bin/

RUN cd /opt && \
    wget -q https://github.com/shenwei356/seqkit/releases/download/v2.11.0/seqkit_linux_amd64.tar.gz && \
    tar xzf seqkit_linux_amd64.tar.gz && \
    mv seqkit /usr/local/bin/seqkit && \
    chmod +x /usr/local/bin/seqkit && \
    rm seqkit_linux_amd64.tar.gz

RUN cd /opt && \
	wget -q  https://github.com/gpertea/gffread/releases/download/v0.12.7/gffread-0.12.7.Linux_x86_64.tar.gz && \
	tar xzf gffread-0.12.7.Linux_x86_64.tar.gz
ENV PATH=${PATH}:/opt/gffread-0.12.7.Linux_x86_64

RUN python3 -m pip install plotly

RUN cd /opt && \
    rm *tar.gz
ENV PATH=/usr/local/bin/:$PATH


USER ${NB_UID}




