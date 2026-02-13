import numpy as np


def rotate_left(val, r_bits):
    """64-bit circular left shift using NumPy."""
    return (val << np.uint64(r_bits)) | (val >> np.uint64(64 - r_bits))


def mix_arx(g1, g2, g3, g4, r):
    """
    We test the variable rotation 'r' on the first step,
    using ChaCha constants for the rest to see how 'r' affects.
    """
    g1 = g1 + g2
    g4 = g4 ^ g1
    g4 = rotate_left(g4, r)

    g3 = g3 + g4
    g2 = g2 ^ g3
    g2 = rotate_left(g2, 12)

    g1 = g1 + g2
    g4 = g4 ^ g1
    g4 = rotate_left(g4, 8)

    g3 = g3 + g4
    g2 = g2 ^ g3
    g2 = rotate_left(g2, 7)

    return g1 ^ g2 ^ g3 ^ g4


def count_set_bits(n):
    """Counts the number of 1s in the binary representation."""
    count = np.zeros_like(n, dtype=np.uint8)
    temp = n.copy()
    while np.any(temp):
        count += (temp & 1).astype(np.uint8)
        temp >>= np.uint64(1)
    return count


def analyze_avalanche(n_samples=5000000):
    print("Generating random base states...")
    # generate N random 64-bit states for each generator
    rng = np.random.default_rng()
    g1_base = rng.integers(0, 2 ** 64, n_samples, dtype=np.uint64)
    g2_base = rng.integers(0, 2 ** 64, n_samples, dtype=np.uint64)
    g3_base = rng.integers(0, 2 ** 64, n_samples, dtype=np.uint64)
    g4_base = rng.integers(0, 2 ** 64, n_samples, dtype=np.uint64)

    # flip exactly one bit in the input.
    g1_flipped = g1_base ^ np.uint64(1)

    print(f"Testing all 63 possible rotation constants over {n_samples} samples...\n")
    print("Rot | Avg Flipped Bits | Distance from Ideal (32.0)")
    print("-" * 50)

    best_rot = 0
    best_dist = 64.0

    # test every possible rotation constant
    for r in range(1, 64):
        # compute baseline output
        out_base = mix_arx(g1_base, g2_base, g3_base, g4_base, r)

        # compute output with the flipped input bit
        out_flipped = mix_arx(g1_flipped, g2_base, g3_base, g4_base, r)

        # XOR to find which bits changed, then count them
        diff = out_base ^ out_flipped
        flipped_counts = count_set_bits(diff)

        # calculate the average across all samples
        avg_flipped = np.mean(flipped_counts)
        dist_from_ideal = abs(32.0 - avg_flipped)

        print(f"{r:3d} | {avg_flipped:16.4f} | {dist_from_ideal:.4f}")

        if dist_from_ideal < best_dist:
            best_dist = dist_from_ideal
            best_rot = r

    print("-" * 50)
    print(f"Optimal Rotation Constant: {best_rot} (Off by {best_dist:.4f} bits)")


if __name__ == "__main__":
    analyze_avalanche()