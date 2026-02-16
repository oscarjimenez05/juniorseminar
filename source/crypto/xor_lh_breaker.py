import time
import math
from z3 import *


# --- Decode Lehmer to Permutation Constraints ---
def lehmer_to_permutation(lehmer_code, w):
    """
    Decodes a Lehmer integer into a list of indices representing the
    relative rank of each element in the window.

    Returns a list 'perm' where perm[i] is the rank of window[i].
    Example: if window is [10, 5, 20], ranks are [1, 0, 2].
    Meaning: window[1] < window[0] < window[2].
    """
    factoradic = []
    temp_code = lehmer_code
    for i in range(w):
        fact = math.factorial(w - 1 - i)
        factoradic.append(temp_code // fact)
        temp_code %= fact

    # convert factoradic to permutation (Lehmer code logic)
    available_ranks = list(range(w))
    ranks = [0] * w

    for i in range(w):
        # The number 'c_i' means x_i is the (c_i)-th smallest of the remaining numbers
        pick_index = factoradic[i]
        ranks[i] = available_ranks.pop(pick_index)

    return ranks


def generate_constraints_from_ranks(window_vars, ranks):
    """
    Converts ranks into Z3 strict inequality constraints.
    If ranks are [1, 0, 2] -> window[1] < window[0] < window[2]
    """
    constraints = []

    # invert ranks to find which index has rank 0, rank 1, etc.
    # sorted_indices[0] is the index of the smallest element
    # sorted_indices[1] is the index of the 2nd smallest
    sorted_indices = [0] * len(ranks)
    for index, rank in enumerate(ranks):
        sorted_indices[rank] = index

    # add chain constraints: smallest < 2nd Smallest < 3rd ...
    for k in range(len(sorted_indices) - 1):
        idx_A = sorted_indices[k]
        idx_B = sorted_indices[k + 1]
        constraints.append(ULT(window_vars[idx_A], window_vars[idx_B]))

    return constraints



def z3_xorshift64_step(x):
    x = x ^ (x << 13)
    x = x ^ LShR(x, 7)
    x = x ^ (x << 17)
    return x


def z3_solve_fast(observed_sequence, w, delta, minimum, maximum):
    print(f"--- Setting up Optimized Solver (Sequence Length: {len(observed_sequence)}) ---")

    solver = Solver()

    unknown_seed = BitVec('seed', 64)

    current_state = unknown_seed

    window = []
    for _ in range(w):
        current_state = z3_xorshift64_step(current_state)
        window.append(current_state)

    for step_idx, observed_val in enumerate(observed_sequence):
        if step_idx > 0:
            new_window = window[delta:]
            for _ in range(delta):
                current_state = z3_xorshift64_step(current_state)
                new_window.append(current_state)

            window = new_window

        # 2. Decode the Observation
        # Assuming FULL RANGE observation (Lehmer code is visible)
        # If observed_val was modulo'd, this specific optimization needs branching.
        # But for MAX=719, this is exact.
        lehmer_val = observed_val - minimum

        ranks = lehmer_to_permutation(lehmer_val, w)

        fast_constraints = generate_constraints_from_ranks(window, ranks)
        solver.add(fast_constraints)

    print("Running solver...")
    start = time.time()
    result = solver.check()
    duration = time.time() - start

    if result == sat:
        print(f"BROKEN in {duration:.4f}s")
        model = solver.model()
        found = model[unknown_seed].as_long()
        print(f"Recovered Seed: {found}")
        return found
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
    state = seed
    outputs = []

    R = math.factorial(w)

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

        if lehmer < R:
            outputs.append(lehmer)

        new_window = window[delta:]
        for _ in range(delta):
            state = real_xorshift64_step(state)
            new_window.append(state)
        window = new_window

    return outputs


if __name__ == "__main__":
    W = 6
    DELTA = 1
    MIN = 0
    MAX = 719
    SECRET = 123456789123456789
    SEQ_LEN = 7

    print(f"Secret Seed: {SECRET}")
    sequence = get_real_sequence(SECRET, W, DELTA, MIN, MAX, SEQ_LEN)
    print(f"Observed Sequence: {sequence}")

    print(f"Z3 got sequence: {get_real_sequence(z3_solve_fast(sequence, W, DELTA, MIN, MAX), W, DELTA, MIN, MAX, SEQ_LEN)}")