import numpy as np
from numpy import pi, zeros
from numpy.random import randint
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import AerSimulator

alice = bob = eve = None  # Instances of the three agents
backend = AerSimulator()  # Backend instance for simulation
qubits = None  # Qubit to be sent
n = 16  # Final key length
delta = 2  # Redundancy factor
b = (4 + delta) * n  # Number of qubits to be sent

counts = []

ALICE = 0
BOB = 1
EVE = 2


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
        print(f"\nAlice's key:\t\t {alice.get_key()}")

    def append_key(self, key):
        self.key += str(key)

    def get_received_bits(self):
        return self.received_bits

    def print_received_bits(self):
        print("\nAlice's received bits:\t [", end="")

    def get_received_bit(self, i):
        return self.received_bits[i]

    def set_received_bit(self, i, bit):
        self.received_bits[i] = bit

    def get_bases(self):
        return self.bases

    def print_bases(self):
        print(f"\nAlice's bases:\t\t {self.bases}")

    def receive(self, i):
        global qubits

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
        print(f"Bob's key:\t\t {bob.get_key()}")

    def append_key(self, key):
        self.key += str(key)

    def get_received_bits(self):
        return self.received_bits

    def print_received_bits(self):
        print(f"\x1b[1D]\nBob's received bits:\t {bob.get_received_bits()}")

    def get_received_bit(self, i):
        return self.received_bits[i]

    def set_received_bit(self, i, bit):
        self.received_bits[i] = bit

    def get_bases(self):
        return self.bases

    def print_bases(self):
        print(f"Bob's bases:\t\t {self.bases}")

    def receive(self, i):
        qubits.ry(self.angles[self.bases[i]], BOB)
        qubits.measure(BOB, BOB)


class Charlie:
    def __init__(self, with_eve=False):
        print(f"[{type(self).__name__}] Preparing qubit for transmission...")
        self.with_eve = with_eve

    def prepare_qubit(self):
        global qubits

        qubits_count = 3 if self.with_eve else 2
        qr = QuantumRegister(qubits_count, "qr")  # alice(q[0]), bob(q[1]), eve(q[2])
        cr = ClassicalRegister(qubits_count, "cr")
        qubits = QuantumCircuit(qr, cr)

        # Bell state on qubits
        qubits.h(qr[ALICE])
        qubits.cx(qr[ALICE], qr[BOB])
        if self.with_eve:
            qubits.cx(qr[BOB], qr[EVE])


class Eve:
    possible_angles = [pi / 4]

    def __init__(self):
        print(f"[{type(self).__name__}] Generating {b} random bases for the qubits...")

        self.bases = randint(len(self.possible_angles), size=b)
        self.intercepted_bits = zeros(b, dtype=int)

    def get_intercepted_bits(self):
        return self.intercepted_bits

    def print_intercepted_bits(self):
        print(f"Eve's intercepted bits:\t {eve.get_intercepted_bits()}")

    def set_intercepted_bit(self, i, bit):
        self.intercepted_bits[i] = bit

    def get_bases(self):
        return self.bases

    def print_bases(self):
        print(f"Eve's bases:\t\t {self.bases}")

    def intercept(self, i):
        qubits.ry(self.possible_angles[self.bases[i]], EVE)
        qubits.measure(EVE, EVE)


def print_parameters():
    np.set_printoptions(linewidth=300)
    print("Parameters:")
    print(f"  * Key length (n): {n} bits")
    print(f"  * Redundancy factor (delta): {delta}")
    print(f"  * Number of qubits to be sent (b): {b}")


def init_agents(with_eve=False):
    alice = Alice()
    bob = Bob()
    charlie = Charlie(with_eve)
    eve = Eve() if with_eve else None

    alice.print_bases()
    if with_eve:
        eve.print_bases()
    bob.print_bases()

    return alice, bob, charlie, eve


def measure(i):
    transpiled = transpile(qubits, backend)
    c = backend.run(transpiled, shots=8192).result().get_counts()

    counts.append(c)

    measured_states = max(c, key=c.get)[::-1]
    alice.set_received_bit(i, int(measured_states[ALICE]))
    bob.set_received_bit(i, int(measured_states[BOB]))
    if eve:
        eve.set_intercepted_bit(i, int(measured_states[EVE]))

    print(measured_states[ALICE], end=" ", flush=True)


def calc_CHSH():
    CHSH = []

    for i in range(0, len(counts), 4):
        theta_dict = counts[i : i + 4]
        zz = theta_dict[0]
        zx = theta_dict[1]
        xz = theta_dict[2]
        xx = theta_dict[3]

        num_shots = sum(xx[y] for y in xx)

        chsh1 = 0

        for element in zz:
            parity = (-1) ** (int(element[0]) + int(element[1]))
            chsh1 += parity * zz[element]

        for element in zx:
            parity = (-1) ** (int(element[0]) + int(element[1]))
            chsh1 -= parity * zx[element]

        for element in xz:
            parity = (-1) ** (int(element[0]) + int(element[1]))
            chsh1 += parity * xz[element]

        for element in xx:
            parity = (-1) ** (int(element[0]) + int(element[1]))
            chsh1 += parity * xx[element]

        CHSH.append(abs(chsh1 / num_shots))

    S = max(CHSH)
    print(f"\nCHSH value (S):\t\t {S:.3f}")

    return S


def calc_key():
    i = j = 0

    while i < b and j < n:
        if alice.angles[alice.get_bases()[i]] == bob.angles[bob.get_bases()[i]]:
            alice.append_key(alice.get_received_bit(i))
            bob.append_key(bob.get_received_bit(i))
            j += 1

        i += 1

    if j < n:
        print(f"\nError: Insufficient number of matching bases ({j} < {n})")
        exit(1)

    alice.print_key()
    bob.print_key()


def main(with_eve=False):
    print_parameters()

    global alice, bob, eve
    alice, bob, charlie, eve = init_agents(with_eve)

    alice.print_received_bits()

    for i in range(b):
        charlie.prepare_qubit()

        alice.receive(i)
        bob.receive(i)
        if with_eve:
            eve.intercept(i)

        measure(i)

    bob.print_received_bits()
    if with_eve:
        eve.print_intercepted_bits()

    calc_key()

    S = calc_CHSH()

    if S - 2 < 0.1:
        print("\nIntrusion detected! Bell's inequality not violated.")


if __name__ == "__main__":
    # Set `with_eve` to `True` to include Eve in the simulation
    main(with_eve=False)
