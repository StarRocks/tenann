FROM mirrors.tencent.com/starrocks/dev-env:branch-3.1-tq
ENV TENANN_THIRDPARTY /var/local/thirdparty
ENV TENANN_GCC_HOME /opt/gcc/usr
ENV TENANN_HOME /root/tenann
ENV FC /opt/rh/devtoolset-10/root/bin/gfortran
ADD / /root/tenann
COPY thirdparty/ ${TENANN_THIRDPARTY}/

RUN yum install -y centos-release-scl perl-App-cpanminus && yum install -y devtoolset-10-gcc-gfortran 
RUN /var/local/thirdparty/download-thirdparty.sh
RUN /var/local/thirdparty/build-thirdparty.sh
RUN rm -rf /root/tenann
RUN cp /opt/rh/devtoolset-10/root/lib/gcc/x86_64-redhat-linux/10/libgfortran.a /usr/local/lib/

WORKDIR /root