import numpy as np
from numpy.random import randint
from qiskit import QuantumCircuit, transpile
from qiskit_aer import QasmSimulator

alice = bob = eve = None  # Istanze degli agenti
simulator = QasmSimulator()  # Simulatore per la misura dei qubit
qubit = None  # Qubit da inviare
n = 8  # Lunghezza della chiave (finale)
delta = 1  # Fattore di ridondanza
extended_key_length = (4 + delta) * n  # Lunghezza della chiave con ridondanza


class Alice:
    def __init__(self):
        print(
            f"\n[{type(self).__name__}] Generazione di {extended_key_length} bit casuali per la chiave..."
        )
        print(
            f"[{type(self).__name__}] Generazione di {extended_key_length} basi casuali..."
        )
        self.bases = randint(
            2, size=extended_key_length
        )  # 0: Computazionale, 1: Hadamard
        self.key = randint(2, size=extended_key_length)  # Bit casuali per la chiave

    def get_key(self):
        return self.key

    def get_bases(self):
        return self.bases

    def encode_bit(self, bit, base):
        """Codifica un bit di Alice in base scelta."""

        global qubit

        qubit = QuantumCircuit(1, 1)

        if bit == 1:
            qubit.x(0)  # Applica NOT se il bit è 1 (default è 0)
        if base == 1:
            qubit.h(0)  # Applica Hadamard se in base Hadamard

    def send(self, i):
        """Invia un qubit codificato a Bob."""

        self.encode_bit(self.key[i], self.bases[i])


class Bob:
    def __init__(self):
        print(
            f"[{type(self).__name__}] Generazione di {extended_key_length} basi casuali..."
        )
        self.bases = randint(
            2, size=extended_key_length
        )  # 0: Computazionale, 1: Hadamard
        self.received_key = np.zeros(extended_key_length, dtype=int)
        self.matching_key = np.zeros(extended_key_length, dtype=int)
        self.matching_bases = []

    def get_bases(self):
        return self.bases

    def get_received_key(self):
        return self.received_key

    def get_matching_key(self):
        return self.matching_key

    def get_matching_bases(self):
        return self.matching_bases

    def measure(self, base, simulator):
        """Bob misura il qubit nella sua base scelta."""

        global qubit

        if base == 1:
            qubit.h(0)  # Cambia base se necessario

        qubit.measure(0, 0)  # Misura il qubit

        # Simulazione
        transpiled_circuit = transpile(qubit, simulator)
        counts = simulator.run(transpiled_circuit).result().get_counts()
        measured_bit = max(counts, key=counts.get)

        print(measured_bit, end=" ", flush=True)

        return measured_bit

    def receive(self, i):
        self.received_key[i] = self.measure(self.bases[i], simulator)

    def check_bases_match(self):
        for i in range(extended_key_length):
            if alice.get_bases()[i] == self.bases[i]:
                self.matching_bases.append(i)
                self.matching_key[i] = self.received_key[i]
            else:
                self.matching_key[i] = -1  # Indica che la base non corrisponde

        if len(self.matching_bases) < 2 * n:
            print(
                f"\n\nLe basi che corrispondono ({len(self.matching_bases)}) sono minori di {2 * n}. La comunicazione viene interrotta."
            )

            exit(1)
        else:
            print(
                f"\x1b[1D]\nChiave selezionata da Bob: \t[{' '.join([str(b) if b > -1 else '-' for b in self.matching_key])}]"
            )

            self.matching_bases = np.array(self.matching_bases)

            print(
                f"\nBasi corrispondenti ({len(self.matching_bases)}): \t{self.matching_bases}"
            )


class Eve:
    global qubit

    def __init__(self):
        print(
            f"[{type(self).__name__}] Generazione di {extended_key_length} basi casuali..."
        )

        self.bases = randint(
            2, size=extended_key_length
        )  # 0: Computazionale, 1: Hadamard
        self.intercepted_key = np.zeros(extended_key_length, dtype=int)

    def get_bases(self):
        return self.bases

    def get_intercepted_key(self):
        return self.intercepted_key

    def measure(self, base, simulator):
        """Eve misura il qubit nel suo caso base (interferenza)."""

        if base == 1:
            qubit.h(0)  # Cambia base se necessario

        qubit.measure(0, 0)  # Misura il qubit

        # Intercettazione del qubit e misura
        transpiled_circuit = transpile(qubit, simulator)
        counts = simulator.run(transpiled_circuit).result().get_counts()
        measured_bit = max(counts, key=counts.get)

        return measured_bit

    def intercept(self, i):
        self.intercepted_key[i] = self.measure(self.bases[i], simulator)

    def print_intercepted_key(self):
        print(f"\x1b[1D]\nChiave intercettata da Eve: \t{self.intercepted_key}", end="")


def print_parameters():
    np.set_printoptions(linewidth=300)
    print("Parametri:")
    print(f"  * Lunghezza della chiave: {n} bit")
    print(f"  * Fattore di ridondanza: {delta}")
    print(f"  * Lunghezza della chiave estesa: {extended_key_length} bit")


def generate_key(with_eve=False):
    """Genera i bit casuali e le basi per Alice, Bob e opzionalmente Eve."""

    alice = Alice()
    eve = Eve() if with_eve else None
    bob = Bob()

    print(f"\nChiave:\t\t\t\t{alice.get_key()}\n")
    print(
        "Basi di Alice:\t\t\t["
        + " ".join(["C" if b == 0 else "H" for b in alice.get_bases()])
        + "]"
    )
    if with_eve:
        print(
            "Basi di Eve:\t\t\t["
            + " ".join(["C" if b == 0 else "H" for b in eve.get_bases()])
            + "]"
        )
    print(
        "Basi di Bob:\t\t\t["
        + " ".join(["C" if b == 0 else "H" for b in bob.get_bases()])
        + "]"
    )

    return alice, bob, eve


def split_valid_indices():
    # Vengono selezionati 2 * n indici casuali tra quelli delle basi corrispondenti
    work_indices = np.random.choice(bob.get_matching_bases(), 2 * n, replace=False)
    np.random.shuffle(work_indices)

    # Gli indici validi vengono divisi in due gruppi: uno per la chiave...
    key_indices = work_indices[:n]
    key_indices.sort()
    print(f"Indici per la chiave: \t\t{key_indices}")

    # ...l'altro per il controllo
    check_indices = work_indices[n:]
    check_indices.sort()
    print(f"Indici per il controllo: \t{check_indices}\n")

    return key_indices, check_indices


def check_intrusion(key_indices, check_indices):
    intrusion_detected = False
    mismatched_indices = []
    key = ""

    for i in range(n):
        # Se non sta venendo rilevata alcuna intrusione, si costruisce la chiave
        if not intrusion_detected:
            key += str(bob.get_matching_key()[key_indices[i]])
        else:
            key = None

        # Controllo se i bit resi noti da Alice e Bob sono diversi
        if (
            bob.get_matching_key()[check_indices[i]]
            != alice.get_key()[check_indices[i]]
        ):
            intrusion_detected = True
            mismatched_indices.append(check_indices[i])

    return intrusion_detected, np.array(mismatched_indices), key


def check_false_negative(mismatched_indices, key_indices):
    # Verifica se si è trattato di un falso negativo: poiché la scelta dei bit di controllo è casuale,
    # questa potrebbe aver portato a non selezionare bit che avrebbero potuto rivelare l'intrusione

    mismatched_indices = []
    for i in range(n):
        if bob.get_matching_key()[key_indices[i]] != alice.get_key()[key_indices[i]]:
            mismatched_indices.append(key_indices[i])
            
    mismatched_indices = np.array(mismatched_indices)

    if len(mismatched_indices) > 0:
        print(
            f"Falso negativo! I bit negli indici {mismatched_indices} non corrispondono, ma non sono stati selezionati per il controllo."
        )


def main(with_eve=False):
    print_parameters()

    global alice, bob, eve
    alice, bob, eve = generate_key(with_eve)

    print("\nChiave ricevuta da Bob: \t[", end="")

    # Fase di scambio della chiave
    for i in range(extended_key_length):
        alice.send(i)

        # Se c'è Eve, intercetta e misura il qubit
        if with_eve:
            eve.intercept(i)

        # Bob misura il qubit nella base scelta
        bob.receive(i)

    if with_eve:
        eve.print_intercepted_key()

    # Fase di confronto delle basi
    bob.check_bases_match()

    key_indices, check_indices = split_valid_indices()

    intrusion_detected, mismatched_indices, key = check_intrusion(
        key_indices, check_indices
    )

    if not intrusion_detected:
        print(f"Chiave ({len(key)} bit):\t\t\t{key}")

        check_false_negative(mismatched_indices, key_indices)
    else:
        mismatched_indices.sort()
        print(
            f"Intrusione rilevata! Gli indici dei bit non corrispondenti sono: {mismatched_indices}"
        )


if __name__ == "__main__":
    # Impostare il flag `with_eve` a True per includere Eve, False per escluderla
    main(with_eve=True)
