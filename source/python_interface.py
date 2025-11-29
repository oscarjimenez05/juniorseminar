#!/usr/bin/env python3
import sys
import struct
import time
import argparse

import generators as gens

maximum = 2 ** 32 - 1
seed = 3817035023
chunk_size = 8192
debug = 0


def output(next_seed, expected):
    """
    Outputs numbers to stdout
    :return: the next seed
    """
    numbers = gens.pcg64(next_seed, expected, maximum+1)
    if len(numbers) != expected:
        print(f"[WARN] Expected {expected}, got {len(numbers)}", file=sys.stderr)
        raise SystemExit(1)

    sys.stdout.buffer.write(struct.pack('<{}I'.format(expected), *numbers))

    if debug:
        for num in numbers:
            print(num, file=sys.stderr)

    sys.stdout.flush()
    return int(numbers[-1])


def pipe():
    """
    Main function to run the PIPE interface for Dieharder.
    """
    print(f"--- Dieharder Interface Initialized ---", file=sys.stderr)
    print(f"Range = [0, {maximum}]", file=sys.stderr)
    print(f"Chunk size = {chunk_size}", file=sys.stderr)
    print(f"Starting stream...", file=sys.stderr)

    chunks_sent = 0
    start_time = time.time()
    next_seed = seed

    try:
        while True:
            next_seed = output(next_seed, chunk_size)

            chunks_sent += 1
            if chunks_sent % 2000 == 0:
                elapsed = time.time() - start_time
                rate = (chunks_sent * chunk_size) / elapsed
                print(f"[INFO] Sent {chunks_sent:,} chunks "
                      f"({chunks_sent * chunk_size:,} numbers) "
                      f"at {rate:,.0f} numbers/sec", file=sys.stderr)
                sys.stderr.flush()

    except BrokenPipeError:
        print("\n--- Stream closed by Dieharder. Exiting gracefully. ---", file=sys.stderr)
        sys.stderr.flush()
    except KeyboardInterrupt:
        print("\n--- Stream interrupted by user. Exiting. ---", file=sys.stderr)
        sys.stderr.flush()


def file():
    """
    Generate a fixed number of random 32-bit unsigned integers and write to stdout as binary.
    """
    total_numbers = 200_000_000

    print(f"--- Dieharder Fixed Output Interface Initialized ---", file=sys.stderr)
    print(f"Range = [0, {maximum}]", file=sys.stderr)
    print(f"Chunk size = {chunk_size}", file=sys.stderr)
    print(f"Total numbers to generate = {total_numbers:,}", file=sys.stderr)
    print(f"Starting stream...", file=sys.stderr)

    chunks_sent = 0
    numbers_sent = 0
    start_time = time.time()
    next_seed = seed

    try:
        while numbers_sent < total_numbers:
            remaining = total_numbers - numbers_sent
            current_chunk = min(chunk_size, remaining)

            next_seed = output(next_seed, current_chunk)

            chunks_sent += 1
            numbers_sent += current_chunk

            if chunks_sent % 500 == 0:
                elapsed = time.time() - start_time
                rate = numbers_sent / elapsed
                print(f"[INFO] {round(100*numbers_sent/total_numbers, 3)}% "
                      f"Sent {numbers_sent:,}/{total_numbers:,} numbers "
                      f"({rate:,.0f} nums/sec)", file=sys.stderr)
                sys.stderr.flush()

        sys.stdout.flush()
        print(f"--- Completed {numbers_sent:,} numbers. ---", file=sys.stderr)

    except BrokenPipeError:
        print("\n--- Stream closed early. Exiting gracefully. ---", file=sys.stderr)
    except KeyboardInterrupt:
        print("\n--- Interrupted by user. Exiting. ---", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Dieharder Interface.")
    parser.add_argument("mode", help="(f)ile or (p)ipe.")
    args = parser.parse_args()
    if args.mode == 'f':
        file()
    elif args.mode == 'p':
        pipe()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
