#!/usr/bin/env python3
import sys
import struct
import time

import c_lcg_lh as c


def main():
    """
    Main function to run the interface for Dieharder.
    """
    chunk_size = 4096
    seed = 123456789
    maximum = 2 ** 32 - 1
    print(f"--- Dieharder Interface Initialized ---", file=sys.stderr)
    print(f"Range = [0, {maximum}]", file=sys.stderr)
    print(f"Chunk size = {chunk_size}", file=sys.stderr)
    print(f"Starting stream...", file=sys.stderr)

    chunks_sent = 0
    start_time = time.time()

    try:
        while True:
            numbers = c.g_lcg_lh64(seed, chunk_size, 0, maximum)

            if len(numbers) != chunk_size:
                print(f"[WARN] Expected {chunk_size}, got {len(numbers)}", file=sys.stderr)
                break


            sys.stdout.buffer.write(struct.pack('<{}I'.format(chunk_size), *numbers))
            sys.stdout.flush()
            seed = int(numbers[-1])

            chunks_sent += 1
            if chunks_sent % 500 == 0:
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


if __name__ == "__main__":
    main()
