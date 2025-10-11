#!/usr/bin/env python3
import sys
import struct
import time

import c_lcg_lh as c


def main():
    """
    Generate a fixed number of random 32-bit unsigned integers and write to stdout as binary.
    """
    chunk_size = 4096
    #total_numbers = 200_000_000
    total_numbers = 5000000
    seed = 123456789
    maximum = 2 ** 32 - 1

    print(f"--- Dieharder Fixed Output Interface Initialized ---", file=sys.stderr)
    print(f"Range = [0, {maximum}]", file=sys.stderr)
    print(f"Chunk size = {chunk_size}", file=sys.stderr)
    print(f"Total numbers to generate = {total_numbers:,}", file=sys.stderr)
    print(f"Starting stream...", file=sys.stderr)

    chunks_sent = 0
    numbers_sent = 0
    start_time = time.time()

    try:
        while numbers_sent < total_numbers:
            remaining = total_numbers - numbers_sent
            current_chunk = min(chunk_size, remaining)
            numbers = c.g_lcg_lh64(seed, current_chunk, 0, maximum, step=0)

            if len(numbers) != current_chunk:
                print(f"[WARN] Expected {current_chunk}, got {len(numbers)}", file=sys.stderr)
                break

            sys.stdout.buffer.write(struct.pack('<{}I'.format(current_chunk), *numbers))
            seed = int(numbers[-1])

            chunks_sent += 1
            numbers_sent += current_chunk

            if chunks_sent % 500 == 0:
                elapsed = time.time() - start_time
                rate = numbers_sent / elapsed
                print(f"[INFO] Sent {numbers_sent:,}/{total_numbers:,} numbers "
                      f"({rate:,.0f} nums/sec)", file=sys.stderr)
                sys.stderr.flush()

        sys.stdout.flush()
        print(f"--- Completed {numbers_sent:,} numbers. ---", file=sys.stderr)

    except BrokenPipeError:
        print("\n--- Stream closed early. Exiting gracefully. ---", file=sys.stderr)
    except KeyboardInterrupt:
        print("\n--- Interrupted by user. Exiting. ---", file=sys.stderr)


if __name__ == "__main__":
    main()
