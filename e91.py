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
delta = 0  # Redundancy factor
b = (4 + delta) * n  # Number of qubits to be sent

ALICE = 0  # Qubit index for Alice
BOB = 1  # Qubit index for Bob
EVE = 2  # Qubit index for Eve
THRESHOLD = 0.1  # Threshold for Bell's inequality
CHSH_LIMIT = 2  # Maximum value of violation of Bell's inequality
SHOTS = 1  # Number of shots for simulation
WITH_EVE = False  # Whether to include Eve in the simulation
TIMEOUT = 2  # Timeout for retrying in case of insufficient matching bases


class Alice:
    angles = [0, pi / 4, pi / 2]

    def __init__(self):
        print(
            f"\n[{type(self).__name__}] Generating {b} random bases for the qubits..."
        )
        self.bases = randint(len(self.angles), size=b)
        self.received_bits = zeros(b, dtype=int)
        self.key = ""

    def get_key(self):
        return self.key

    def print_key(self):
        print(f"\n{type(self).__name__}'s key:\t\t {alice.get_key()}")

    def append_key(self, key):
        self.key += str(key)

    def get_received_bits(self):
        return self.received_bits

    def print_received_bits(self):
        print(f"\n{type(self).__name__}'s received bits:\t [", end="")

    def get_received_bit(self, i):
        return self.received_bits[i]

    def set_received_bit(self, i, bit):
        self.received_bits[i] = bit

    def get_bases(self):
        return self.bases

    def print_bases(self):
        print(f"\n{type(self).__name__}'s bases:\t\t {self.bases}")

    def receive(self, i):
        qubits.ry(self.angles[self.bases[i]], ALICE)
        qubits.measure(ALICE, ALICE)


class Bob:
    angles = [pi / 4, pi / 2, 3 * pi / 4]

    def __init__(self):
        print(f"[{type(self).__name__}] Generating {b} random bases for the qubits...")
        self.bases = randint(len(self.angles), size=b)
        self.received_bits = zeros(b, dtype=int)
        self.key = ""

    def get_key(self):
        return self.key

    def print_key(self):
        print(f"{type(self).__name__}'s key:\t\t {bob.get_key()}")

    def append_key(self, key):
        self.key += str(key)

    def get_received_bits(self):
        return self.received_bits

    def print_received_bits(self):
        print(
            f"\x1b[1D]\n{type(self).__name__}'s received bits:\t {bob.get_received_bits()}"
        )

    def get_received_bit(self, i):
        return self.received_bits[i]

    def set_received_bit(self, i, bit):
        self.received_bits[i] = bit

    def get_bases(self):
        return self.bases

    def print_bases(self):
        print(f"{type(self).__name__}'s bases:\t\t {self.bases}")

    def receive(self, i):
        qubits.ry(self.angles[self.bases[i]], BOB)
        qubits.measure(BOB, BOB)


class Charlie:
    def __init__(self):
        print(f"[{type(self).__name__}] Preparing qubit for transmission...")

    def prepare_qubit(self):
        global qubits

        qubits_count = 3 if WITH_EVE else 2
        qr = QuantumRegister(qubits_count, "qr")
        cr = ClassicalRegister(qubits_count, "cr")
        qubits = QuantumCircuit(qr, cr)

        qubits.h(qr[ALICE])
        qubits.cx(qr[ALICE], qr[BOB])
        if WITH_EVE:
            qubits.cx(qr[BOB], qr[EVE])


class Eve:
    possible_angles = [0, pi / 4, pi / 2]

    def __init__(self):
        print(f"[{type(self).__name__}] Generating {b} random bases for the qubits...")

        self.bases = randint(len(self.possible_angles), size=b)
        self.intercepted_bits = zeros(b, dtype=int)
        self.key = ""

    def get_key(self):
        return self.key

    def print_key(self):
        print(f"{type(self).__name__}'s key:\t\t {eve.get_key()}")

    def append_key(self, key):
        self.key += str(key)

    def get_intercepted_bits(self):
        return self.intercepted_bits

    def print_intercepted_bits(self):
        print(
            f"{type(self).__name__}'s intercepted bits:\t {eve.get_intercepted_bits()}"
        )

    def get_intercepted_bit(self, i):
        return self.intercepted_bits[i]

    def set_intercepted_bit(self, i, bit):
        self.intercepted_bits[i] = bit

    def get_bases(self):
        return self.bases

    def print_bases(self):
        print(f"{type(self).__name__}'s bases:\t\t {self.bases}")

    def intercept(self, i):
        qubits.ry(self.possible_angles[self.bases[i]], EVE)
        qubits.measure(EVE, EVE)


def print_parameters():
    np.set_printoptions(linewidth=300)
    print("Parameters:")
    print(f"  * Key length (n): {n} bits")
    print(f"  * Redundancy factor (delta): {delta}")
    print(f"  * Number of qubits to be sent (b): {b}")


def init_agents():
    global alice, bob, charlie, eve

    alice = Alice()
    bob = Bob()
    charlie = Charlie()
    eve = Eve() if WITH_EVE else None

    alice.print_bases()
    if WITH_EVE:
        eve.print_bases()
    bob.print_bases()


def simulate(i):
    transpiled = transpile(qubits, backend)
    counts = backend.run(transpiled, shots=SHOTS).result().get_counts()

    measured_states = max(counts, key=counts.get)[::-1]
    alice.set_received_bit(i, int(measured_states[ALICE]))
    bob.set_received_bit(i, int(measured_states[BOB]))
    if WITH_EVE:
        eve.set_intercepted_bit(i, int(measured_states[EVE]))

    print(measured_states[ALICE], end=" ", flush=True)


def calc_CHSH():
    values = {
        (a, b): 0 for a in range(len(Alice.angles)) for b in range(len(Bob.angles))
    }
    counts = {
        (a, b): 0 for a in range(len(Alice.angles)) for b in range(len(Bob.angles))
    }

    for i in range(b):
        alice_basis = alice.get_bases()[i]
        bob_basis = bob.get_bases()[i]

        alice_measure = -1 if alice.get_received_bit(i) == 0 else 1
        bob_measure = -1 if bob.get_received_bit(i) == 0 else 1

        values[(alice_basis, bob_basis)] += alice_measure * bob_measure
        counts[(alice_basis, bob_basis)] += 1

    for pair in values:
        if counts[pair] > 0:
            values[pair] /= counts[pair]
        else:
            values[pair] = 0

    S = abs(values[(0, 0)] - values[(0, 2)] + values[(2, 0)] + values[(2, 2)])

    print(f"\nCHSH value (S):\t\t {S:.3f}")

    return S


def calc_key():
    i = j = 0

    while i < b and j < n:
        if alice.angles[alice.get_bases()[i]] == bob.angles[bob.get_bases()[i]]:
            alice.append_key(alice.get_received_bit(i))
            bob.append_key(bob.get_received_bit(i))
            if WITH_EVE:
                eve.append_key(eve.get_intercepted_bits()[i])

            j += 1

        i += 1

    if j < n:
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

    alice.print_received_bits()

    for i in range(b):
        charlie.prepare_qubit()

        alice.receive(i)
        bob.receive(i)
        if WITH_EVE:
            eve.intercept(i)

        simulate(i)

    bob.print_received_bits()
    if WITH_EVE:
        eve.print_intercepted_bits()

    calc_key()

    S = calc_CHSH()

    if S - CHSH_LIMIT < THRESHOLD:
        print(
            "\nIntrusion detected! Bell's inequality is not violated, hence key is compromised."
        )


if __name__ == "__main__":
    main()
