FROM ubuntu:18.04
RUN apt-get update && \
    apt-get install -y autoconf automake gcc gfortran g++ zlib1g-dev make cmake wget make git swig pkg-config python3 python-pip python3-pip libffi-dev libssl-dev gsl-bin libgsl-dev libblas-dev libfftw3-dev && apt-get clean all

RUN mkdir /build/
#RUN cd /build && wget https://www.python.org/ftp/python/3.6.1/Python-3.6.1.tgz \
#  && tar xvzf Python-3.6.1.tgz && cd /build/Python-3.6.1 \
#  && ./configure && make -j4 && make install && make clean && rm /build/Python-3.6.1.tgz
RUN pip3 install --upgrade pip
RUN cd /build && wget http://www.mpich.org/static/downloads/3.2/mpich-3.2.tar.gz \
  && tar xvzf mpich-3.2.tar.gz && cd /build/mpich-3.2 \
  && ./configure && make -j4 && make install && make clean && rm /build/mpich-3.2.tar.gz
RUN cd /build && wget https://bitbucket.org/mpi4py/mpi4py/downloads/mpi4py-2.0.0.tar.gz \
  && tar xvzf mpi4py-2.0.0.tar.gz
RUN cd /build/mpi4py-2.0.0 && python3 setup.py build && python3 setup.py install && rm -rf /build/
RUN rm -f /usr/bin/python && ln -s /usr/bin/python3 /usr/bin/python
RUN pip3 install numpy==1.16
#RUN pip3 install numpy
COPY requirements.txt /tmp/
RUN pip3 install -r /tmp/requirements.txt
RUN pip3 install joblib
RUN pip3 install pandas
RUN pip3 install pyfits
RUN pip3 install statsmodels
RUN git clone https://github.com/esheldon/healpix_util.git
RUN cd healpix_util && python3 setup.py build && python3 setup.py install && rm -rf /build/
RUN git clone https://j-dr:"Ursa&Aleph0"@github.com/j-dr/redshift-wg.git
RUN pip3 install https://bitbucket.org/yymao/helpers/get/master.zip
RUN pip3 install h5py
RUN git clone https://github.com/astropy/halotools.git
RUN pip3 install cython
RUN cd halotools && python3 setup.py build && python3 setup.py install && rm -rf /build/
RUN pip3 install corrfunc
RUN apt-get update && apt-get install apt-transport-https
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys D6BC243565B2087BC3F897C9277A7293F59E4889 && echo "deb http://miktex.org/download/ubuntu xenial universe" | tee /etc/apt/sources.list.d/miktex.list && apt-get update && apt-get --assume-yes install miktex
RUN pip3 install jupyter ipython
RUN pip3 install seaborn
ARG CSBUSTT=124
RUN git clone https://github.com/j-dr/pyaddgals.git
RUN cd pyaddgals && python3 setup.py build && python3 setup.py install && rm -rf /build/
RUN git clone https://github.com/j-dr/pixLC.git
RUN cd pixLC && python3 setup.py build && python3 setup.py install && rm -rf /build/
RUN python3 -c "exec(\"from fast3tree import fast3tree \\nimport numpy as np \\ntree = fast3tree(np.zeros((10,3),dtype=np.float32))\\ntree = fast3tree(np.zeros((10,3),dtype=np.float64))\")"
RUN    git clone https://github.com/erykoff/redmapper && cd redmapper && pip install -r requirements.txt --no-cache-dir && \
       cd .. && git clone https://github.com/LSSTDESC/healsparse.git && cd healsparse && python setup.py install && \
       cd ../redmapper && python setup.py install

#CMD ls /usr/local/lib/python3.5/dist-packages/fast3tree/
ENV XDG_CACHE_HOME=/srv/cache
RUN mkdir -p $XDG_CACHE_HOME/astropy
ENV XDG_CONFIG_HOME=/srv/config
RUN mkdir -p $XDG_CONFIG_HOME/astropy
RUN python -c "import astropy"
RUN /sbin/ldconfig
