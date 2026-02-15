#!/usr/bin/env python3
import sys
import struct
import time
import argparse

import crypto_lh as crypto

maximum = 2 ** 32 - 1
chunk_size = 8192
w = 14

generator = None
debug = False


def output(expected):
    """
    Outputs numbers to stdout
    :return: the next seed
    """
    numbers = generator.generate_chunk(expected, debug)

    if len(numbers) != expected:
        print(f"[WARN] Expected {expected}, got {len(numbers)}", file=sys.stderr)
        raise SystemExit(1)

    sys.stdout.buffer.write(struct.pack('<{}I'.format(expected), *numbers))

    if debug:
        for num in numbers:
            print(num, file=sys.stderr)
    sys.stdout.flush()


def pipe():
    """
    Main function to run the PIPE interface for testing.
    """
    print(f"--- Testing Interface Initialized ---", file=sys.stderr)
    print(f"Range = [0, {maximum}]", file=sys.stderr)
    print(f"Chunk size = {chunk_size}", file=sys.stderr)
    print(f"Starting stream...", file=sys.stderr)

    chunks_sent = 0
    start_time = time.time()

    try:
        while True:
            output(chunk_size)

            chunks_sent += 1
            if chunks_sent % 4000 == 0:
                elapsed = time.time() - start_time
                rate = (chunks_sent * chunk_size) / elapsed
                print(f"[INFO] Sent {chunks_sent:,} chunks "
                      f"({chunks_sent * chunk_size:,} numbers) "
                      f"at {rate:,.0f} numbers/sec", file=sys.stderr)
                sys.stderr.flush()

    except BrokenPipeError:
        print("\n--- Stream closed by Tester. Exiting gracefully. ---", file=sys.stderr)
        sys.stderr.flush()
    except KeyboardInterrupt:
        print("\n--- Stream interrupted by user. Exiting. ---", file=sys.stderr)
        sys.stderr.flush()



def main():
    parser = argparse.ArgumentParser(description=" Crypto Testing Interface.")
    parser.add_argument("seed", type=int, help="seed")
    parser.add_argument("--delta", type=int, help="delta")
    parser.add_argument("--debug", action="store_true", help="enable debug mode")

    args = parser.parse_args()

    global generator, debug
    debug = args.debug

    generator = crypto.CryptoLehmer(STATES, w, args.delta, 0, maximum)

    pipe()


if __name__ == "__main__":
    main()
