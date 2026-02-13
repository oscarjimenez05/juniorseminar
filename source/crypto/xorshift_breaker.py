import time
from z3 import *


def xorshift64_z3(x):
    """
    Uses Z3's Logical Shift Right
    """
    x = x ^ LShR(x, 13)
    x = x ^ (x << 7)
    x = x ^ LShR(x, 17)
    return x


def run_cryptanalysis():
    print("--- Starting Z3 ---")

    true_seed = 6283551937245425110

    # run it forward normally to get the output the attacker sees.
    temp = true_seed ^ (true_seed >> 13)
    temp = temp ^ ((temp << 7) & 0xFFFFFFFFFFFFFFFF) # need mask to keep number within 64 bits in python
    temp = temp ^ (temp >> 17)
    observed_output = temp

    print(f"Attacker observed output: {observed_output}")
    print("Handing equations to Z3...\n")

    solver = Solver()

    # define unknown 64 bit variable for z3
    unknown_seed = BitVec('unknown_seed', 64)

    # running the unknown variable through known algorithm
    z3_output = xorshift64_z3(unknown_seed)

    # add constraint output of those equations must equal observed number
    solver.add(z3_output == observed_output)

    # solve
    start_time = time.time()
    result = solver.check()
    end_time = time.time()

    # 5. CHECK THE RESULTS
    if result == sat:
        print("Z3 found a satisfying state!")
        model = solver.model()
        guessed_seed = model[unknown_seed].as_long()

        print(f"Z3 guessed the seed is: {guessed_seed}")
        if guessed_seed == true_seed:
            print("MATCH: attacker recovered hidden state.")
    else:
        print("SECURE: Z3 could not find a solution.")

    print(f"Time taken: {end_time - start_time:.5f} seconds")


if __name__ == "__main__":
    run_cryptanalysis()