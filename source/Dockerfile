FROM python:3.11-slim-bullseye

RUN sed -i 's/main/main contrib non-free/g' /etc/apt/sources.list

RUN apt-get update && apt-get install -y \
    gcc \
    libc-dev \
    libtestu01-0-dev \
    dieharder \
    make \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir numpy cython setuptools
RUN pip install --no-cache-dir matplotlib scipy statsmodels z3-solver

WORKDIR /app
COPY setup.py alt_setup.py crypto_setup.py ./
COPY *.pyx ./
RUN python3 setup.py build_ext --inplace

COPY alternatives/ ./alternatives/
RUN python3 alt_setup.py build_ext --inplace

COPY crypto/ ./crypto/
RUN python3 crypto_setup.py build_ext --inplace

COPY . .

RUN gcc -o test_from_pipe test_from_pipe.c -I/usr/include/testu01 -ltestu01 -lm
RUN gcc -o test_from_pipe_BigCrush test_from_pipe_BigCrush.c -I/usr/include/testu01 -ltestu01 -lm

CMD ["/bin/bash"]