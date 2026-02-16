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


def z3_solve_sequence(observed_sequence, w, delta, minimum, maximum):
    """
    Attacks the generator using a SEQUENCE of outputs.
    Each new observation adds a new layer of constraints, eliminating collisions.
    """
    print(f"--- Setting up Z3 Solver for {len(observed_sequence)} outputs ---")

    R = math.factorial(w)
    r_range = maximum - minimum + 1
    thresh = R - (R % r_range)

    solver = Solver()

    unknown_seed = BitVec('seed', 64)

    current_state = unknown_seed

    window = []
    for _ in range(w):
        current_state = z3_xorshift64_step(current_state)
        window.append(current_state)

    factorials = [math.factorial(w - i - 1) for i in range(w)]

    for step_idx, observed_val in enumerate(observed_sequence):

        if step_idx > 0:
            new_window = window[delta:]
            for _ in range(delta):
                current_state = z3_xorshift64_step(current_state)
                new_window.append(current_state)

            window = new_window

        lehmer_sum = BitVecVal(0, 64)
        for i in range(w):
            smaller_count = BitVecVal(0, 64)
            for j in range(i + 1, w):
                # uses ULT (Unsigned Less Than)
                is_smaller = If(ULT(window[j], window[i]), BitVecVal(1, 64), BitVecVal(0, 64))
                smaller_count += is_smaller
            lehmer_sum += smaller_count * factorials[i]

        solver.add(ULT(lehmer_sum, thresh))
        target_mod = observed_val - minimum
        if r_range >= math.factorial(w):
            solver.add(lehmer_sum == target_mod)
        else:
            solver.add(URem(lehmer_sum, r_range) == target_mod)

    print("Running solver...")
    start = time.time()
    result = solver.check()
    duration = time.time() - start

    if result == sat:
        print(f"BROKEN in {duration:.4f}s")
        model = solver.model()
        found_seed = model[unknown_seed].as_long()
        print(f"Recovered Seed: {found_seed}")
        return found_seed
    else:
        print("UNSAT")
        print("(This might mean the sequence had a 'rejected' gap we didn't account for,")
        print(" or the generator is secure... but for XorLehmer, it's likely a gap.)")
        return None



def real_xorshift64_step(x):
    x = (x ^ (x << 13)) & 0xFFFFFFFFFFFFFFFF
    x = (x ^ (x >> 7)) & 0xFFFFFFFFFFFFFFFF
    x = (x ^ (x << 17)) & 0xFFFFFFFFFFFFFFFF
    return x


def get_real_sequence(seed, w, delta, minimum, maximum, count):
    """Generates a list of 'count' outputs from the real generator."""
    state = seed
    outputs = []

    R = math.factorial(w)
    r_range = maximum - minimum + 1
    thresh = R - (R % r_range)

    window = []
    for _ in range(w):
        state = real_xorshift64_step(state)
        window.append(state)

    while len(outputs) < count:
        lehmer = 0
        factorials = [math.factorial(w - i - 1) for i in range(w)]
        for i in range(w):
            smaller = 0
            for j in range(i + 1, w):
                if window[j] < window[i]:
                    smaller += 1
            lehmer += smaller * factorials[i]

        if lehmer < thresh:
            val = (lehmer % r_range) + minimum
            outputs.append(val)

        # ideally, we should simulate the gap if a reject happens
        # a real attack script would try assuming skip=0, then skip=1 etc.
        new_window = window[delta:]
        for _ in range(delta):
            state = real_xorshift64_step(state)
            new_window.append(state)
        window = new_window

    return outputs


if __name__ == "__main__":
    W = 6
    DELTA = 6
    MIN = 0
    MAX = 719
    SECRET = 2366022249
    SEQ_LEN = 3  # observe SEQ_LEN nums

    print(f"Secret Seed: {SECRET}")
    sequence = get_real_sequence(SECRET, W, DELTA, MIN, MAX, SEQ_LEN)
    print(f"Observed Sequence: {sequence}")

    print(f"Z3 got sequence: {get_real_sequence(z3_solve_sequence(sequence, W, DELTA, MIN, MAX), W, DELTA, MIN, MAX, SEQ_LEN)}")