FROM mirrors.tencent.com/starrocks/dev-env:branch-3.1-tq
ENV TENANN_THIRDPARTY /var/local/thirdparty
ENV TENANN_GCC_HOME /opt/gcc/usr
ENV TENANN_HOME /root/tenann
ADD / /root/tenann
COPY thirdparty/ ${TENANN_THIRDPARTY}/

RUN yum install -y gcc-gfortran libgfortran-static 
RUN /var/local/thirdparty/download-thirdparty.sh
RUN /var/local/thirdparty/build-thirdparty.sh
RUN rm -rf /root/tenann
RUN cp /usr/lib/gcc/x86_64-redhat-linux/4.8.2/libgfortran.a /usr/local/lib

WORKDIR /root