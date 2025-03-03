from matplotlib import pyplot as plt
import numpy as np
from numpy import pi, zeros
from numpy.random import randint
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import circuit_drawer

alice = bob = eve = None  # Instances of the three agents
backend = AerSimulator()  # Backend instance for simulation
qubits = None  # Qubit to be sent
n = 10  # Final key length
delta = 1  # Redundancy factor
b = (4 + delta) * n  # Number of qubits to be sent

ALICE = 0
BOB = 1
EVE = 2


class Alice:
    possible_angles = [0, -pi / 4, -pi / 2]

    def __init__(self):
        print(
            f"\n[{type(self).__name__}] Generating {b} random bases for the qubits..."
        )
        self.bases = randint(len(self.possible_angles), size=b)
        self.received_bits = zeros(b, dtype=int)
        self.key = ""

    def get_key(self):
        return self.key

    def append_key(self, key):
        self.key += str(key)

    def get_received_key(self):
        return self.received_bits

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

        qubits.ry(self.possible_angles[self.bases[i]], ALICE)
        qubits.measure(ALICE, ALICE)


class Bob:
    possible_angles = [-pi / 4, -pi / 2, -3 * pi / 4]

    def __init__(self):
        print(f"[{type(self).__name__}] Generating {b} random bases for the qubits...")
        self.bases = randint(len(self.possible_angles), size=b)
        self.received_bits = zeros(b, dtype=int)
        self.key = ""

    def get_key(self):
        return self.key

    def append_key(self, key):
        self.key += str(key)

    def get_received_key(self):
        return self.received_bits

    def get_received_bit(self, i):
        return self.received_bits[i]

    def set_received_bit(self, i, bit):
        self.received_bits[i] = bit

    def get_bases(self):
        return self.bases

    def print_bases(self):
        print(f"Bob's bases:\t\t {self.bases}")

    def receive(self, i):
        qubits.ry(self.possible_angles[self.bases[i]], BOB)
        qubits.measure(BOB, BOB)


class Charlie:
    def __init__(self, with_eve=False):
        print(f"[{type(self).__name__}] Preparing qubit for transmission...")
        self.with_eve = with_eve

    def prepare_qubit(self):
        global qubits

        qubits_count = 3 if self.with_eve else 2
        qr = QuantumRegister(qubits_count, "qr")  # alice(q[0]), bob(q[1]), eve(q[2])
        cr = ClassicalRegister(qubits_count, "cr")  # classical bit = hold measurements
        qubits = QuantumCircuit(qr, cr)

        # Bell state on qubits
        qubits.h(qr[ALICE])
        qubits.cx(qr[ALICE], qr[BOB])  # 0 (Alice), 1 (Bob)
        if self.with_eve:
            qubits.cx(qr[BOB], qr[EVE])



class Eve:
    possible_angles = [-pi / 4]

    def __init__(self):
        print(f"[{type(self).__name__}] Generating {b} random bases for the qubits...")

        self.bases = randint(len(self.possible_angles), size=b)
        self.intercepted_bits = zeros(b, dtype=int)

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
    counts = backend.run(transpiled).result().get_counts()

    measured_states = max(counts, key=counts.get)

    measured_states = measured_states[::-1]
    alice.set_received_bit(i, int(measured_states[ALICE]))
    bob.set_received_bit(i, int(measured_states[BOB]))
    if eve:
        eve.set_intercepted_bit(i, int(measured_states[EVE]))

    print(measured_states[ALICE], end=" ", flush=True)


def calc_CHSH():
    correlations = {(a, b): 0 for a in range(3) for b in range(3)}
    counts = {(a, b): 0 for a in range(3) for b in range(3)}

    for i in range(len(alice.get_bases())):
        a_base = alice.get_bases()[i]
        b_base = bob.get_bases()[i]

        a_val = 1 - 2 * alice.get_received_bit(i)
        b_val = 1 - 2 * bob.get_received_bit(i)
        correlations[(a_base, b_base)] += a_val * b_val
        counts[(a_base, b_base)] += 1

    for key in correlations:
        if counts[key] > 0:
            correlations[key] /= counts[key]

    S = abs(
        correlations[(0, 0)]
        - correlations[(0, 2)]
        + correlations[(2, 0)]
        + correlations[(2, 2)]
    )
    return S


def calc_key():
    i = j = 0

    while i < b and j < n:
        if alice.get_bases()[i] == bob.get_bases()[i]:
            alice.append_key(alice.get_received_bit(i))
            bob.append_key(bob.get_received_bit(i))
            j += 1

        i += 1

    if j < n:
        print("\nError: Insufficient number of matching bases.")
        exit(1)


def main(with_eve=False):
    print_parameters()

    global alice, bob, eve
    alice, bob, charlie, eve = init_agents(with_eve)

    print("\nAlice's received bits:\t [", end="")

    for i in range(b):
        charlie.prepare_qubit()

        # alice.receive(i)
        
        qubits.ry(alice.possible_angles[alice.get_bases()[i]], ALICE)

        # bob.receive(i)
        
        qubits.ry(bob.possible_angles[bob.get_bases()[i]], BOB)

        # if with_eve:
        #     eve.intercept(i)
        
        if with_eve:
            qubits.ry(eve.possible_angles[eve.get_bases()[i]], EVE)
        
        
        qubits.measure(list(range(qubits.num_clbits)), list(range(qubits.num_clbits))) 

        measure(i)

    print(f"\x1b[1D]\nBob's received bits:\t {bob.get_received_key()}")

    # plt.figure(figsize=(12, 8))
    # circuit_drawer(qubits, output="mpl")
    # plt.title("Example E91 Circuitt")
    # plt.show()

    S = calc_CHSH()
    intrusion_detected = S <= 2

    calc_key()

    print(f"\nAlice's key:\t\t {alice.get_key()}")
    print(f"Bob's key:\t\t {bob.get_key()}")

    print(f"\nCHSH value:\t\t {S:.3f}")

    if intrusion_detected:
        print("\nIntrusion detected! Bell's inequality not violated.")


if __name__ == "__main__":
    # Set `with_eve` to `True` to include Eve in the simulation
    main(with_eve=False)
