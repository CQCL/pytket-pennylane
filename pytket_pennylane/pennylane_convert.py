from typing import List, OrderedDict, cast
from numpy import pi as PI
from pytket.circuit import OpType, QubitRegister, BitRegister, Circuit
from pennylane.operation import Operation

# TODO add all pennylane operations https://pennylane.readthedocs.io/en/stable/introduction/operations.html
PYTKET_OPERATION_MAP = {
    "Hadamard": OpType.H,
    "PauliX": OpType.X,
    "PauliY": OpType.Y,
    "PauliZ": OpType.Z,
    "S": OpType.S,
    "T": OpType.T,
    "RX": OpType.Rx,
    "RY": OpType.Ry,
    "RZ": OpType.Rz,
    "CNOT": OpType.CX,
    "CY": OpType.CY,
    "CZ": OpType.CZ,
    "SWAP": OpType.SWAP,
    "U1": OpType.U1,
    "U2": OpType.U2,
    "U3": OpType.U3,
    "CRZ": OpType.CRz,
    "Toffoli": OpType.CCX,
    "CSWAP": OpType.CSWAP,
}

PYTKET_OPERATION_INVERSES_MAP = {k + ".inv": v for k, v in PYTKET_OPERATION_MAP.items()}
OPERATION_MAP = {**PYTKET_OPERATION_MAP, **PYTKET_OPERATION_INVERSES_MAP}


def apply_operations(
    operations: List[Operation], wire_map: OrderedDict, qreg: QubitRegister
) -> List[Circuit]:
    """Apply the circuit operations.

    This method serves as an auxiliary method to :meth:`~.PytketDevice.apply`.

    Args:
        operations (List[pennylane.Operation]): operations to be applied

    Returns:
        list[Circuit]: a list of tket circuit objects that
            specify the corresponding operations
    """
    circuits = []

    for operation in operations:
        # Apply the circuit operations
        device_wires = operation.wires.map(wire_map)
        par = cast(List[float], operation.parameters)
        operation = operation.name

        mapped_operation = OPERATION_MAP[operation]

        # self.qubit_unitary_check(operation, par, device_wires)

        qregs = [qreg[i] for i in device_wires.labels]

        invert = operation.endswith(".inv")

        new_c = Circuit()
        for q in qreg:
            new_c.add_qubit(q)

        new_c.add_gate(mapped_operation, [p / PI for p in par], qregs)
        if invert:
            new_c = new_c.dagger()

        circuits.append(new_c)

    return circuits


def pennylane_to_tk(
    operations: List[Operation],
    wire_map: OrderedDict,
    qreg: QubitRegister,
    creg: BitRegister,
    measure=False,
) -> Circuit:
    applied_operations = apply_operations(operations, wire_map, qreg)
    circ = Circuit("temp")
    circ.add_q_register(qreg)
    circ.add_c_register(creg)
    for circuit in applied_operations:
        circ.append(circuit)

    if measure:
        # Add measurements if they are needed
        for qr, cr in zip(qreg, creg):
            circ.Measure(qr, cr)

    return circ
