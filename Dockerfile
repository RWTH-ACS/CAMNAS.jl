# Build Dpsim
FROM sogno/dpsim:dev AS DPSIM_BUILDER

RUN mkdir /app
WORKDIR /app

# Clone DPSIM without history
RUN git clone --depth 1 --branch v1.2.1 https://github.com/sogno-platform/dpsim.git

WORKDIR /app/dpsim/
RUN mkdir -p build && cd build && \
    cmake \
      -DCIMPP=OFF \
      -DCGMES_Build=ON \
      -Dwith_villas=OFF \
      -DPSIM_BUILD_DOC=OFF \
      -DPSIM_BUILD_EXAMPLES=ON \
      .. && \
    make -j$(nproc)


# Fetch cim grid data
WORKDIR /app/
RUN git clone https://github.com/n-eiling/cim-grid-data.git
# Note: The dpsim import path is relative and not absolute to build (likely broken)
RUN mkdir -p /app/dpsim/build/dpsim/examples/cxx/build/_deps/cim-data-src/WSCC-09/WSCC-09_RX
RUN cp -R /app/cim-grid-data/WSCC-09/WSCC-09_RX/* /app/dpsim/build/dpsim/examples/cxx/build/_deps/cim-data-src/WSCC-09/WSCC-09_RX


# Build Camnas
# Install build tools
RUN dnf update && \
    dnf install -y \
        wget \
        tar \
        git \
        gcc \
        g++ \
        make

# Install Julia
ARG JULIA_MAJOR=1.12
ARG JULIA_VERSION=1.12.1

RUN wget -q https://julialang-s3.julialang.org/bin/linux/x64/${JULIA_MAJOR}/julia-${JULIA_VERSION}-linux-x86_64.tar.gz && \
    tar -C /usr/local -xzf julia-${JULIA_VERSION}-linux-x86_64.tar.gz && \
    ln -s /usr/local/julia-${JULIA_VERSION}/bin/julia /usr/local/bin/julia && \
    rm julia-${JULIA_VERSION}-linux-x86_64.tar.gz

ENV JULIA_BINDIR=/usr/local/julia-${JULIA_VERSION}/bin
ENV PATH=/usr/local/julia-${JULIA_VERSION}/bin:$PATH

# Workdir
WORKDIR /app/dpsim/dpsim/src/SolverPlugins/CAMNAS.jl