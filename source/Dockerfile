FROM python:3.9-slim-bullseye

RUN apt-get update && apt-get install -y \
    gcc \
    libc-dev \
    libtestu01-0-dev \
    dieharder \
    make \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir numpy cython setuptools

WORKDIR /app
COPY . .

RUN python3 setup.py build_ext --inplace

RUN gcc -o test_from_pipe test_from_pipe.c -ltestu01 -lprobdist -mylib -lm

CMD ["/bin/bash"]