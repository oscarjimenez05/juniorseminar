import time
import math
from z3 import *


def z3_xorshift64_step(x):
    """
    Symbolic implementation of Xorshift64.
    x: Z3 BitVec(64)
    """
    # x ^= x << 13
    x = x ^ (x << 13)
    # x ^= x >> 7
    x = x ^ LShR(x, 7)
    # x ^= x << 17
    x = x ^ (x << 17)
    return x


def z3_generate_lehmer(start_seed, w):
    """
    Symbolically generates a single Lehmer code from a seed.
    Matches the logic in xor_lh.pyx exactly.
    """
    state = start_seed
    window = []

    factorials = [math.factorial(w - i - 1) for i in range(w)]

    # Initialize/Fill window (assuming delta=w for a fresh chunk)
    for _ in range(w):
        state = z3_xorshift64_step(state)
        window.append(state)

    # cap at 64 bits
    lehmer_sum = BitVecVal(0, 64)

    for i in range(w):
        smaller_count = BitVecVal(0, 64)
        for j in range(i + 1, w):
            # uses ULT (Unsigned Less Than)
            is_smaller = If(ULT(window[j], window[i]), BitVecVal(1, 64), BitVecVal(0, 64))
            smaller_count += is_smaller

        lehmer_sum += smaller_count * factorials[i]

    return lehmer_sum


def real_xorshift64_step(x):
    x = (x ^ (x << 13)) & 0xFFFFFFFFFFFFFFFF
    x = (x ^ (x >> 7)) & 0xFFFFFFFFFFFFFFFF
    x = (x ^ (x << 17)) & 0xFFFFFFFFFFFFFFFF
    return x


def real_get_lehmer(seed, w):
    """Generates a real observation to test the solver against."""
    state = seed
    window = []
    factorials = [math.factorial(w - i - 1) for i in range(w)]

    for _ in range(w):
        state = real_xorshift64_step(state)
        window.append(state)

    lehmer = 0
    for i in range(w):
        smaller = 0
        for j in range(i + 1, w):
            if window[j] < window[i]:
                smaller += 1
        lehmer += smaller * factorials[i]

    # uint64 overflow wrapping
    return lehmer & 0xFFFFFFFFFFFFFFFF


def run_attack():
    W_SIZE = 6
    SECRET_SEED = 2366022249

    print(f"--- Attacking XorLehmer (w={W_SIZE}) ---")
    print(f"Target Secret Seed: {SECRET_SEED}")

    observed_lehmer = real_get_lehmer(SECRET_SEED, W_SIZE)
    print(f"Attacker observes Lehmer Code: {observed_lehmer}")

    print("Building Z3 equations...")
    solver = Solver()

    unknown_seed = BitVec('seed', 64)

    symbolic_output = z3_generate_lehmer(unknown_seed, W_SIZE)

    solver.add(symbolic_output == observed_lehmer)

    print("Running solver (this reverses the hash)...")
    start = time.time()
    result = solver.check()
    duration = time.time() - start

    if result == sat:
        print(f"\n[!] BROKEN in {duration:.4f} seconds!")
        model = solver.model()
        found_seed = model[unknown_seed].as_long()
        print(f"Z3 calculated seed: {found_seed}")

        if found_seed == SECRET_SEED:
            print(">> SUCCESS: Exact seed recovery.")
        else:
            # not a match, but state collision
            print(">> SUCCESS: Found a valid seed collision (Pre-image attack successful).")
            print(f"(Actual seed was {SECRET_SEED}, found valid alternative {found_seed})")
    else:
        print("FAILED: Could not solve. (This shouldn't happen for Xorshift)")


if __name__ == "__main__":
    run_attack()