FROM ubuntu:18.04

# Prepare distribution
RUN apt-get update -q \
    && apt-get -y upgrade

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends apt-utils

# CPP deps
RUN DEBIAN_FRONTEND=noninteractive \
       apt-get install -qy \
         libboost-all-dev \
         libcomedi-dev \
         libfftw3-dev \
         libgmp-dev \
         libgsl-dev \
         liblog4cpp5-dev \
         libqwt-qt5-dev \
         libqt5opengl5-dev \
         libsdl1.2-dev \
         libzmq3-dev \
         gir1.2-gobject \
         gir1.2-gtk-3.0 \
         gir1.2-freedesktop \
         gir1.2-pango \
         qtbase5-dev \
         --no-install-recommends

# Py3 deps
RUN DEBIAN_FRONTEND=noninteractive \
       apt-get install -qy \
         python3-dev \
         python3-click \
         python3-click-plugins \
         python3-mako \
         python3-gi \
         python3-gi-cairo \
         python3-lxml \
         python3-numpy \
         python3-pyqt5 \
         python3-yaml \
         python3-zmq \
         python3-sphinx \
         python3-six \
         python3-scipy \
         --no-install-recommends

# Build deps
RUN mv /sbin/sysctl /sbin/sysctl.orig \
    && ln -sf /bin/true /sbin/sysctl \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
       --no-install-recommends \
       build-essential \
       ccache \
       cmake \
       doxygen \
       swig \
       git \
       g++ \
    && apt-get clean \
    && rm -f /sbin/sysctl \
    && mv /sbin/sysctl.orig /sbin/sysctl

RUN rm -rf /var/lib/apt/*

RUN useradd -ms /bin/bash orbit

RUN git clone --recursive  https://github.com/gnuradio/gnuradio.git && cd gnuradio/ && git checkout v3.8.0.0 && mkdir build && cd build/ && cmake ../ && make -j 4 && make install && ldconfig

USER orbit
WORKDIR /home/orbit

ENTRYPOINT ["/bin/bash"]
