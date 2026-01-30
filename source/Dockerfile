FROM python:3.9-slim-bullseye

RUN sed -i 's/main/main contrib non-free/g' /etc/apt/sources.list

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

RUN gcc -o test_from_pipe test_from_pipe.c -I/usr/include/testu01 -ltestu01 -lm

CMD ["/bin/bash"]