FROM ubuntu:15.10
RUN apt-get update && apt-get install -y autoconf automake gcc g++ make gfortran wget && apt-get clean all

RUN mkdir /build/
RUN cd /build && wget https://www.python.org/ftp/python/3.6.1/Python-3.6.1.tgz \
  && tar xvzf Python-2.7.13.tgz && cd /build/Python-3.6.1 \
  && ./configure && make -j4 && make install && make clean && rm /build/Python-3.6.1.tgz
RUN cd /build && wget http://www.mpich.org/static/downloads/3.2/mpich-3.2.tar.gz \
  && tar xvzf mpich-3.2.tar.gz && cd /build/mpich-3.2 \
  && ./configure && make -j4 && make install && make clean && rm /build/mpich-3.2.tar.gz
RUN cd /build && wget https://bitbucket.org/mpi4py/mpi4py/downloads/mpi4py-2.0.0.tar.gz \
  && tar xvzf mpi4py-2.0.0.tar.gz
RUN cd /build/mpi4py-2.0.0 && python setup.py build && python setup.py install && rm -rf /build/
RUN /sbin/ldconfig
RUN apt-get update && 
    apt-get install -y wget make git \
                       swig python-numpy libgsl2 gsl-bin pkg-config \
                       python-pip

RUN git clone git@github.com:j-dr/pyaddgals.git
RUN git clone git@github.com:j-dr/pixLC.git

RUN cd pyaddgals && pip install --no-cache-dir -r requirements.txt
RUN cd pixLC && pip install --no-cache-dir -r requirements.txt
