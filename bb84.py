import time
import numpy as np
from numpy import zeros, array
from numpy.random import randint, choice, shuffle
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

alice = bob = eve = None  # Instances of the three agents
backend = AerSimulator()  # Backend instance for simulation
qubit = None  # Qubit to be sent
n = 8  # Final key length
delta = 1  # Redundancy factor
b = (4 + delta) * n  # Key length with redundancy

COMPUTATIONAL = 0  # Computational basis index
HADAMARD = 1  # Hadamard basis index
MISMATCH = -1  # Value to indicate mismatch in bases
SHOTS = 1  # Number of shots for simulation
WITH_EVE = False  # Whether to include Eve in the simulation
TIMEOUT = 2  # Timeout for retrying in case of insufficient matching bases

ALICE = 0  # Index of Alice
BOB = 1  # Index of Bob
EVE = 2  # Index of Eve


class Alice:
    def __init__(self):
        print(f"\n[{type(self).__name__}] Generating {b} random bits for the key...")
        print(f"[{type(self).__name__}] Generating {b} random bases for the qubits...")
        self.bases = choice([COMPUTATIONAL, HADAMARD], b)  # Random bases for the qubits
        self.key = randint(2, size=b)  # Random bits for the key

    def print_key(self):
        print(f"\nKey:\t\t\t\t{self.key}\n")

    def print_bases(self):
        print(
            f"{type(self).__name__}'s bases:\t\t\t["
            + " ".join(["C" if b == 0 else "H" for b in self.bases])
            + "]"
        )

    def send(self, i):
        global qubit
        classical_bits_count = 2 if WITH_EVE else 1

        qubit = QuantumCircuit(1, classical_bits_count)

        if self.key[i] == 1:
            qubit.x(0)  # Apply NOT gate if bit is 1
        if self.bases[i] == HADAMARD:
            qubit.h(0)  # Apply Hadamard gate if basis is 1 (Hadamard basis)


class Bob:
    def __init__(self):
        print(f"[{type(self).__name__}] Generating {b} random bases for the qubits...")
        self.bases = choice([COMPUTATIONAL, HADAMARD], b)  # Random bases for the qubits
        self.received_key = zeros(b, dtype=int)  # Key received from Alice
        self.matching_key = zeros(b, dtype=int)  # Key bits of matching bases
        self.matching_bases = []  # Indices of matching bases

    def print_bases(self):
        print(
            f"{type(self).__name__}'s bases:\t\t\t["
            + " ".join(["C" if b == 0 else "H" for b in self.bases])
            + "]"
        )

    def receive(self, i):
        if self.bases[i] == HADAMARD:
            qubit.h(0)  # Change basis if necessary

        qubit.measure(0, BOB - 1)

    def check_bases_match(self):
        for i in range(b):
            if alice.bases[i] == self.bases[i]:
                self.matching_bases.append(i)  # Keep track of matching bases
                self.matching_key[i] = self.received_key[i]
            else:
                self.matching_key[i] = MISMATCH

        matching_bases_count = len(self.matching_bases)

        if matching_bases_count < 2 * n:  # Check if there are enough matching bases
            print(
                f"\x1b[1D]\n\nMatching bases count ({matching_bases_count}) is less than {2 * n}."
                + f" Will retry in {TIMEOUT} seconds..."
            )
            time.sleep(TIMEOUT)
            print("\033[H\033[J", end="")

            main()
            exit(1)

        self.matching_bases = array(self.matching_bases)

        print(
            f"\x1b[1D]\n\nMatching bases count: \t\t{matching_bases_count} ≥ {2 * n}"
            + f"\nMatching bases indices: \t{self.matching_bases}"
            + f"\n\nBits kept by {type(self).__name__}: \t\t[{' '.join([str(b) if b != MISMATCH else '-' for b in self.matching_key])}]"
        )


class Eve:
    def __init__(self):
        print(f"[{type(self).__name__}] Generating random {b} bases for the qubits...")

        self.bases = choice([COMPUTATIONAL, HADAMARD], b)  # Random bases for the qubits
        self.intercepted_key = zeros(b, dtype=int)

    def print_bases(self):
        print(
            f"{type(self).__name__}'s bases:\t\t\t["
            + " ".join(["C" if b == 0 else "H" for b in self.bases])
            + "]"
        )

    def print_intercepted_key(self):
        print(
            f"\x1b[1D]\nKey intercepted by {type(self).__name__}: \t{self.intercepted_key}",
            end="",
        )

    def intercept(self, i):
        if self.bases[i] == HADAMARD:
            qubit.h(0)  # Basis change if necessary

        qubit.measure(0, EVE - 1)


def print_parameters():
    np.set_printoptions(linewidth=300)
    print("Parameters:")
    print(f"  * Key length (n): {n} bits")
    print(f"  * Redundancy factor (δ): {delta}")
    print(f"  * Extended key length (b): {b} bits")


def init_agents():
    global alice, bob, eve

    alice = Alice()
    bob = Bob()
    eve = Eve() if WITH_EVE else None

    alice.print_key()
    alice.print_bases()

    if WITH_EVE:
        eve.print_bases()

    bob.print_bases()
    
    print("\nKey received by Bob: \t\t[", end="")


def choose_and_split_indices():
    # 2 * n random indices are selected among those of the matching bases
    work_indices = bob.matching_bases.copy()
    shuffle(work_indices)
    discarded_indices = work_indices[2 * n :]
    work_indices = work_indices[: 2 * n]

    if len(discarded_indices) > 0:
        print(f"\nIndices of discarded bits: \t{discarded_indices}")

    # Valid indices are split into two sets: one for the key...
    key_indices = work_indices[:n]
    key_indices.sort()
    print(f"Indices of key bits: \t\t{key_indices}")

    # ...and one for the check bits
    check_indices = work_indices[n:]
    check_indices.sort()
    print(f"Indices of check bits: \t\t{check_indices}\n")

    return key_indices, check_indices, discarded_indices


def check_intrusion(key_indices, check_indices):
    intrusion_detected = False
    mismatched_indices = []
    key = ""

    for i in range(n):
        # If no intrusion is being detected, the key is built
        if not intrusion_detected:
            key += str(bob.matching_key[key_indices[i]])
        elif key is not None:
            key = None

        # Check if the bits revealed by Alice and Bob are different
        if (
            bob.matching_key[check_indices[i]]
            != alice.key[check_indices[i]]
        ):
            intrusion_detected = True
            mismatched_indices.append(check_indices[i])

    mismatched_indices.sort()

    return intrusion_detected, array(mismatched_indices), key


def check_false_negative(key_indices, discarded_indices):
    # Verify if it was a false negative: since the choice of the check bits is random,
    # this could have led to not selecting bits that could have revealed the intrusion

    mismatched_indices = []

    # Check if the bits that do not match are among the key bits...
    for i in range(n):
        if bob.matching_key[key_indices[i]] != alice.key[key_indices[i]]:
            mismatched_indices.append(key_indices[i])

    # ...or among the discarded/extra bits
    for i in range(len(discarded_indices)):
        if bob.matching_key[discarded_indices[i]] != alice.key[discarded_indices[i]]:
            mismatched_indices.append(discarded_indices[i])

    mismatched_indices = array(mismatched_indices)

    if len(mismatched_indices) > 0:
        print(
            f"False negative! Bits in {mismatched_indices} do not match, but were not used for the check (not selected or used for the key)"
        )


def simulate(i):
    transpiled = transpile(qubit, backend)
    counts = backend.run(transpiled, shots=SHOTS).result().get_counts()
    measured_bits = max(counts, key=counts.get)[::-1]

    bob.received_key[i] = int(measured_bits[BOB - 1])
    
    print(bob.received_key[i], end=" ", flush=True)

    if WITH_EVE:
        eve.intercepted_key[i] = int(measured_bits[EVE - 1])

    if i == b - 1:
        print("\x1b[1D]", end="")


def main():
    print_parameters()

    init_agents()

    # Key exchange phase
    for i in range(b):
        alice.send(i)  # Alice sends the qubit to Bob in the chosen basis

        if WITH_EVE:  # If Eve is present, she intercepts the qubit
            eve.intercept(i)

        bob.receive(i)  # Bob measures the qubit in the chosen basis

        simulate(i)  # Simulation of the quantum circuit

    if WITH_EVE:
        eve.print_intercepted_key()

    # Bases comparison phase
    bob.check_bases_match()

    # If enough bases match, the key is split into key and check bits
    key_indices, check_indices, discarded_indices = choose_and_split_indices()

    intrusion_detected, mismatched_indices, key = check_intrusion(
        key_indices, check_indices
    )

    if not intrusion_detected:
        print("No mismatched bits found within check indices. Key exchange successful!")
        print(f"Key ({len(key)} bits):\t\t\t{key}")

        check_false_negative(key_indices, discarded_indices)
    else:
        print(f"Intrusion detected! Bits at indices {mismatched_indices} do not match.")


if __name__ == "__main__":
    main()
