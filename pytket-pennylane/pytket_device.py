#creating device class to build Pennylane plugin
import numpy as np
import pennylane as qml
from pennylane import QubitDevice, DeviceError
from pytket.backends.qulacs import QulacsBackend
from pytket.circuit import OpType
#from pytket.circuit import add_q_register, add_c_register
from pytket import Circuit
#from _version import __version__

from qiskit.circuit.measure import measure
#from qiskit.compiler import assemble, transpile
from qiskit.converters import circuit_to_dag, dag_to_circuit


PYTKET_OPERATION_MAP = {
    "Hadamard" : OpType.H,
    "PauliX" : OpType.X,
    "PauliY" : OpType.Y,
    "PauliZ" : OpType.Z,
    "S" : OpType.S,
    "T" : OpType.T,
    "RX" : OpType.Rx, 
    "RY" : OpType.Ry,
    "RZ" : OpType.Rz,
    "CNOT" : OpType.CX,
    "CY" : OpType.CY,
    "CZ" : OpType.CZ,
    "SWAP" : OpType.SWAP,
    "U1" : OpType.U1,
    "U2" : OpType.U2,
    "U3" : OpType.U3,
    "CRZ" : OpType.CRz,
    "Toffoli" : OpType.CCX,
    "CSWAP" : OpType.CSWAP,
    "QubitUnitary" : OpType.Unitary2qBox
}

PYTKET_OPERATION_INVERSES_MAP = {k + ".inv": v for k, v in PYTKET_OPERATION_MAP.items()}

class pytketDevice(QubitDevice):
    """MyDevice docstring"""
    name = 'pytket-pennylane plugin'
    short_name = 'pytket.mydevice'
    pennylane_requires = '>=0.13.0'
    #version = '0.13.0'
    #plugin_version = __version__
    author = 'KN'

    _operation_map = {**PYTKET_OPERATION_MAP, **PYTKET_OPERATION_INVERSES_MAP}
    operations = set(_operation_map.keys())
    observables = {"PauliX", "PauliY", "PauliZ"}
    _capabilities = {"model": "qubit"}

    #qml->qiskit->tket

    def __init__(self, wires, shots, backend=QulacsBackend()):
        super().__init__(wires=wires, shots=shots)
        self.tket_backend = backend

    def reset(self):
        # Reset only internal data, not the options that are determined on
        # device creation
        self._circuit = Circuit(name="temp")
        self._reg = self._circuit.add_q_register("q", self.num_wires)
        self._creg = self._circuit.add_c_register("c", self.num_wires)
        self._state = None  # statevector of a simulator backend

    def apply(self, operations, **kwargs):
        rotations = kwargs.get("rotations", []) 

        applied_operations = self.apply_operations(operations)

        # Rotating the state for measurement in the computational basis
        rotation_circuits = self.apply_operations(rotations)
        applied_operations.extend(rotation_circuits)

        for circuit in applied_operations:
            self._circuit += circuit

    def apply_operations(self, operations):
        """Apply the circuit operations.

        This method serves as an auxiliary method to :meth:`~.QiskitDevice.apply`.

        Args:
            operations (List[pennylane.Operation]): operations to be applied

        Returns:
            list[QuantumCircuit]: a list of quantum circuit objects that
                specify the corresponding operations
        """
        circuits = []

        for operation in operations:
            # Apply the circuit operations
            device_wires = self.map_wires(operation.wires)
            par = operation.parameters
            operation = operation.name

            mapped_operation = self._operation_map[operation]

            self.qubit_unitary_check(operation, par, device_wires)

            qregs = [self._reg[i] for i in device_wires.labels]

            if operation.split(".inv")[0] in ("QubitUnitary"):
                # Need to revert the order of the quantum registers used in
                # circuit such that it matches the PennyLane ordering
                qregs = list(reversed(qregs))

            ## will pytket.utils.Graph.as_nx() work here?
            ## pytket.circuit.Circuit "Encapsulates a quantum circuit using a DAG representation."?
            dag = circuit_to_dag(QuantumCircuit(self._reg, self._creg, name=""))
            gate = mapped_operation(*par)

            if operation.endswith(".inv"):
                gate = gate.inverse()

            ## need to apply inverse gate to dag circuit
            dag.apply_operation_back(gate, qargs=qregs)
            circuit = dag_to_circuit(dag)
            circuits.append(circuit)

        return circuits

    @staticmethod
    def qubit_unitary_check(operation, par, wires):
        """Input check for the the QubitUnitary operation."""
        if operation == "QubitUnitary":
            if len(par[0]) != 2 ** len(wires):
                raise ValueError(
                    "Unitary matrix must be of shape (2**wires,\
                        2**wires)."
                )

    def analytic_probability(self, wires=None):
        if self._state is None:
            return None

        prob = self.marginal_prob(np.abs(self._state) ** 2, wires)
        return prob

dev = qml.device(short_name = 'pytket.mydevice', wires=2, name = 'pytket-pennylane plugin')
@qml.qnode(dev)
def my_quantum_function(x, y):
    qml.RZ(x, wires=0)
    qml.CNOT(wires=[0,1])
    qml.RY(y, wires=1)
    return qml.expval(qml.PauliZ(1))
