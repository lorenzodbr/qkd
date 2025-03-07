import time
import numpy as np
from numpy import pi, zeros
from numpy.random import randint
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import AerSimulator

alice = bob = charlie = eve = None  # Instances of the three agents
backend = AerSimulator()  # Backend instance for simulation
qubits = None  # Qubits to be sent
n = 8  # Final key length
delta = 1  # Redundancy factor
b = (4 + delta) * n  # Number of qubits to be sent

ALICE = 0  # Qubit index for Alice
BOB = 1  # Qubit index for Bob
EVE = 2  # Qubit index for Eve
THRESHOLD = 0.1  # Threshold for Bell's inequality
CHSH_LIMIT = 2  # Maximum value of violation of Bell's inequality
SHOTS = 1  # Number of shots for simulation
WITH_EVE = True  # Whether to include Eve in the simulation
TIMEOUT = 2  # Timeout for retrying in case of insufficient matching bases


class Alice:
    angles = [0, pi / 4, pi / 2]  # Possible angles for rotation

    def __init__(self):
        print(
            f"\n[{type(self).__name__}] Generating {b} random bases for the qubits..."
        )
        self.bases = randint(len(self.angles), size=b)  # Random bases for the qubits
        self.received_bits = zeros(b, dtype=int)
        self.key = ""

    def print_key(self):
        print(f"\n{type(self).__name__}'s key:\t\t {self.key}")

    def print_received_bits(self):
        print(f"\n{type(self).__name__}'s received bits:\t {self.received_bits}")

    def print_bases(self):
        print(f"\n{type(self).__name__}'s bases:\t\t {self.bases}")

    def receive(self, i):
        qubits[i].ry(
            self.angles[self.bases[i]], ALICE
        )  # Rotate the qubit by the randomly chosen angle
        qubits[i].measure(ALICE, ALICE)  # Measure the qubit after rotation


class Bob:
    angles = [pi / 4, pi / 2, 3 * pi / 4]  # Possible angles for rotation

    def __init__(self):
        print(f"[{type(self).__name__}] Generating {b} random bases for the qubits...")
        self.bases = randint(len(self.angles), size=b)  # Random bases for the qubits
        self.received_bits = zeros(b, dtype=int)
        self.key = ""

    def print_key(self):
        print(f"{type(self).__name__}'s key:\t\t {self.key}")

    def print_received_bits(self):
        print(f"\n{type(self).__name__}'s received bits:\t {self.received_bits}")

    def print_bases(self):
        print(f"{type(self).__name__}'s bases:\t\t {self.bases}")

    def receive(self, i):
        qubits[i].ry(
            self.angles[self.bases[i]], BOB
        )  # Rotate the qubit by the randomly chosen angle
        qubits[i].measure(BOB, BOB)  # Measure the qubit after rotation


class Charlie:
    def __init__(self):
        print(f"[{type(self).__name__}] Preparing qubit for transmission...")

    def send(self):
        qubits_count = 3 if WITH_EVE else 2
        qr = QuantumRegister(qubits_count, "qr")  # Quantum register to store the qubits
        cr = ClassicalRegister(
            qubits_count, "cr"
        )  # Classical register to store measurement results
        qubit = QuantumCircuit(qr, cr)

        qubit.h(
            qr[ALICE]
        )  # Apply Hadamard gate to Alice's qubit to create superposition
        qubit.cx(qr[ALICE], qr[BOB])  # Apply CNOT gate to Bob's qubit to entangle it
        if WITH_EVE:
            qubit.cx(
                qr[ALICE], qr[EVE]
            )  # Apply CNOT gate to Eve's qubit to entangle it if is present

        qubits.append(qubit)


class Eve:
    possible_angles = [0, pi / 4, pi / 2]  # Possible angles for rotation

    def __init__(self):
        print(f"[{type(self).__name__}] Generating {b} random bases for the qubits...")

        self.bases = randint(
            len(self.possible_angles), size=b
        )  # Random bases for the qubits
        self.intercepted_bits = zeros(b, dtype=int)
        self.key = ""

    def print_key(self):
        print(f"{type(self).__name__}'s key:\t\t {self.key}")

    def print_intercepted_bits(self):
        print(f"{type(self).__name__}'s intercepted bits:\t {self.intercepted_bits}")

    def print_bases(self):
        print(f"{type(self).__name__}'s bases:\t\t {self.bases}")

    def intercept(self, i):
        qubits[i].ry(self.possible_angles[self.bases[i]], EVE)
        qubits[i].measure(EVE, EVE)


def print_parameters():
    np.set_printoptions(linewidth=300)
    print("Parameters:")
    print(f"  * Key length (n): {n} bits")
    print(f"  * Redundancy factor (delta): {delta}")
    print(f"  * Number of qubits to be sent (b): {b}")


def init_agents():
    global alice, bob, charlie, eve, qubits

    alice = Alice()
    bob = Bob()
    charlie = Charlie()
    eve = Eve() if WITH_EVE else None
    qubits = []

    alice.print_bases()
    if WITH_EVE:
        eve.print_bases()
    bob.print_bases()


def simulate():
    transpiled = transpile(qubits, backend)
    counts = backend.run(transpiled, shots=SHOTS).result().get_counts()

    for i in range(b):
        # Get the most probable state and set the received bit accordingly (due to little-endian encoding)
        measured_states = max(counts[i], key=counts[i].get)[::-1]

        alice.received_bits[i] = int(measured_states[ALICE])
        bob.received_bits[i] = int(measured_states[BOB])
        if WITH_EVE:
            eve.intercepted_bits[i] = int(measured_states[EVE])

    alice.print_received_bits()


def calc_CHSH():
    values = {  # Expected values for the nine possible combinations of bases
        (a, b): 0 for a in range(len(Alice.angles)) for b in range(len(Bob.angles))
    }
    counts = {  # Number of occurrences for the nine possible combinations of bases
        (a, b): 0 for a in range(len(Alice.angles)) for b in range(len(Bob.angles))
    }

    for i in range(b):
        alice_basis = alice.bases[i]
        bob_basis = bob.bases[i]

        # Map 0 to -1 and 1 to 1
        alice_measure = -1 + 2 * alice.received_bits[i]
        bob_measure = -1 + 2 * bob.received_bits[i]

        # If they collapsed to the same result add 1, otherwise add -1
        values[(alice_basis, bob_basis)] += alice_measure * bob_measure
        counts[(alice_basis, bob_basis)] += 1  # Record the occurrence of the bases

    for pair in values:
        if counts[pair] > 0:
            values[pair] /= counts[pair]  # Normalize the values

    # Calculate the CHSH value according to the formula
    S = abs(values[(0, 0)] - values[(0, 2)] + values[(2, 0)] + values[(2, 2)])

    print(f"\nCHSH value (S):\t\t {S:.3f}")

    return S


def calc_key():
    i = j = 0

    # Get n bits from received bits where bases match
    while i < b and j < n:
        if Alice.angles[alice.bases[i]] == Bob.angles[bob.bases[i]]:
            alice.key += str(alice.received_bits[i])
            bob.key += str(bob.received_bits[i])
            if WITH_EVE:
                eve.key += str(eve.intercepted_bits[i])

            j += 1
        i += 1

    if j < n:  # Retry if insufficient matching bases
        print(
            f"\nError: Insufficient number of matching bases ({j} < {n}). Will retry in {TIMEOUT} seconds..."
        )
        time.sleep(TIMEOUT)
        print("\033[H\033[J", end="")

        main()
        exit(1)

    alice.print_key()
    bob.print_key()
    if WITH_EVE:
        eve.print_key()


def main():
    print_parameters()

    init_agents()

    # Key exchange phase
    for i in range(b):
        charlie.send()

        alice.receive(i)
        bob.receive(i)
        if WITH_EVE:
            eve.intercept(i)

    simulate()  # Simulation of quantum circuits

    bob.print_received_bits()
    if WITH_EVE:
        eve.print_intercepted_bits()

    calc_key()

    S = calc_CHSH()

    if S - CHSH_LIMIT < THRESHOLD:
        print(
            "\nIntrusion detected! Bell's inequality is not violated, hence key communication compromised."
        )


if __name__ == "__main__":
    main()
