import numpy as np
from numpy.random import randint
from qiskit import QuantumCircuit, transpile
from qiskit_aer import QasmSimulator

class Alice:
    def __init__(self, bits_count):
        self.bases = randint(2, size=bits_count)  # 0: Computazionale, 1: Hadamard
        self.key_bits = randint(2, size=bits_count)  # Bit casuali per la chiave

    def get_key(self):
        return self.key_bits

    def get_bases(self):
        return self.bases

    def encode_bit(self, bit, base):
        """Codifica un bit di Alice in base scelta."""
        circuit = QuantumCircuit(1, 1)  # 1 qubit, 1 bit classico
        if bit == 1:
            circuit.x(0)  # Applica NOT se il bit è 1 (default è 0)
        if base == 1:
            circuit.h(0)  # Applica Hadamard se in base Hadamard
        return circuit

class Bob:
    def __init__(self, bits_count):
        self.bases = randint(2, size=bits_count)  # 0: Computazionale, 1: Hadamard

    def get_bases(self):
        return self.bases

    def measure(self, circuit, base, simulator):
        """Bob misura il qubit nella sua base scelta."""
        if base == 1:
            circuit.h(0)  # Cambia base se necessario

        circuit.measure(0, 0)  # Misura il qubit

        # Simulazione
        transpiled_circuit = transpile(circuit, simulator)
        result = simulator.run(transpiled_circuit).result()
        measured_bit = int(next(iter(result.get_counts())))
        return measured_bit


class Eve:
    def __init__(self, bits_count):
        self.bases = randint(2, size=bits_count)  # 0: Computazionale, 1: Hadamard

    def get_bases(self):
        return self.bases

    def measure(self, circuit, base, simulator):
        """Eve misura il qubit nel suo caso base (interferenza)."""
        if base == 1:
            circuit.h(0)  # Cambia base se necessario

        circuit.measure(0, 0)  # Misura il qubit

        # Intercettazione del qubit e misura
        transpiled_circuit = transpile(circuit, simulator)
        result = simulator.run(transpiled_circuit).result()
        measured_bit = int(next(iter(result.get_counts())))
        return measured_bit


def generate_key(bits_count, with_eve=False):
    """Genera i bit casuali e le basi per Alice, Bob e opzionalmente Eve."""

    alice = Alice(bits_count)
    bob = Bob(bits_count)
    eve = Eve(bits_count) if with_eve else None

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


def main(with_eve=False):
    np.set_printoptions(linewidth=300)
    
    n = 8  # Lunghezza della chiave (finale)
    delta = 1  # Fattore di ridondanza
    extended_key_length = (
        4 + delta
    ) * n  # Lunghezza della chiave con ridondanza
    
    print("Parametri:")
    print(f"  * Lunghezza della chiave: {n} bit")
    print(f"  * Fattore di ridondanza: {delta}")
    print(f"  * Lunghezza della chiave estesa: {extended_key_length} bit")

    alice, bob, eve = generate_key(extended_key_length, with_eve)

    simulator = QasmSimulator()

    bob_received_key = np.zeros(extended_key_length, dtype=int)
    bob_matching_key = np.zeros(extended_key_length, dtype=int)
    eve_intercepted_key = np.zeros(extended_key_length, dtype=int)
    matching_bases = []

    # Fase di scambio
    for i in range(extended_key_length):
        circuit = alice.encode_bit(alice.get_key()[i], alice.get_bases()[i])

        # Se c'è Eve, intercetta e misura il qubit
        if with_eve:
            eve_intercepted_key[i] = eve.measure(circuit, eve.get_bases()[i], simulator)

        # Bob misura il qubit nel suo caso base
        bob_received_key[i] = bob.measure(circuit, bob.get_bases()[i], simulator)

        # Se Bob e Alice usano la stessa base, confrontano i risultati
        if alice.get_bases()[i] == bob.get_bases()[i]:
            matching_bases.append(i)
            bob_matching_key[i] = bob_received_key[i]
        else:
            bob_matching_key[i] = -1  # Indica che la base non corrisponde

    print(f"\nChiave ricevuta da Bob: \t{bob_received_key}")
    print(
        f"Chiave valida di Bob: \t\t[{' '.join([str(b) if b > -1 else '-' for b in bob_matching_key])}]"
    )
    if with_eve:
        print(f"Chiave intercettata da Eve: \t{eve_intercepted_key}")

    # Per poter proseguire devono esserci almeno 2 * n basi corrispondenti
    if len(matching_bases) < 2 * n:
        print(
            f"\nLe basi che corrispondono ({len(matching_bases)}) sono minori di {2 * n}. La comunicazione viene interrotta."
        )
        return
    else:
        print(f"\nBasi corrispondenti: \t\t{matching_bases}")

    # Vengono selezionati 2 * n indici casuali tra quelli delle basi corrispondenti
    matching_bases = np.array(matching_bases)
    work_indices = np.random.choice(matching_bases, 2 * n, replace=False)
    np.random.shuffle(work_indices)

    # Gli indici validi vengono divisi in due gruppi: uno per la chiave...
    key_indices = work_indices[:n]
    key_indices.sort()
    print(f"Indici per la chiave: \t\t{key_indices}")

    # ...l'altro per il controllo
    check_indices = work_indices[n:]
    check_indices.sort()
    print(f"Indici per il controllo: \t{check_indices}\n")

    # Salvataggio degli indici non corrispondenti tra quelli di controllo
    mismatched_indices = []

    intrusion_detected = False
    key = ""

    for i in range(n):
        # Se non sta venendo rilevata alcuna intrusione, si costruisce la chiave
        if not intrusion_detected:
            key += str(bob_matching_key[key_indices[i]])

        # Controllo se i bit resi noti da Alice e Bob sono diversi
        if bob_matching_key[check_indices[i]] != alice.get_key()[check_indices[i]]:
            intrusion_detected = True
            mismatched_indices.append(check_indices[i])

    if not intrusion_detected:
        print(f"Chiave ({len(key)} bit):\t\t\t{key}")

        # Verifica se si è trattato di un falso negativo: dato che la scelta dei bit di controllo è
        # casuale, questa potrebbe aver portato a non selezionare bit che avrebbero rivelato l'intrusione
        mismatched_indices.clear()
        for i in range(n):
            if bob_matching_key[key_indices[i]] != alice.get_key()[key_indices[i]]:
                mismatched_indices.append(key_indices[i])
        
        if len(mismatched_indices) > 0:
            print(f"Falso negativo! I bit negli indici {mismatched_indices} non corrispondono, ma non sono stati selezionati per il controllo.")
    else:
        mismatched_indices.sort()
        print(
            f"Intrusione rilevata! Gli indici dei bit non corrispondenti sono: {mismatched_indices}"
        )


if __name__ == "__main__":
    # Impostare il flag `with_eve` a True per includere Eve, False per escluderla
    main(with_eve=False)