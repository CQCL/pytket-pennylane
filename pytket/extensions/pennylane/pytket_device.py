# creating device class to build Pennylane plugin
from typing import Any, Dict, Optional

import numpy as np
from pytket.backends.backend import Backend
from pytket.passes import BasePass
from pennylane import QubitDevice

# from pytket.extensions.qulacs import QulacsBackend
from pytket.extensions.qiskit import AerStateBackend, AerBackend
from pytket.circuit import OpType, Circuit

from .pennylane_convert import (
    OPERATION_MAP,
    pennylane_to_tk,
)


class PytketDevice(QubitDevice):
    """PytketDevice allows pytket backends and compilation to be used as Pennylane devices."""

    name = "pytket-pennylane plugin"
    short_name = "pytket.pytketdevice"
    pennylane_requires = ">=0.14.0"
    version = "0.14.0"
    plugin_version = "0.1.0"
    author = "KN"

    _operation_map = OPERATION_MAP
    operations = set(_operation_map.keys())
    observables = {"PauliX", "PauliY", "PauliZ", "Identity", "Hadamard"}

    def __init__(
        self,
        wires,
        tket_backend: Backend = AerBackend(),
        compilation_pass: Optional[BasePass] = None,
        shots=8192,
    ):
        if not (tket_backend.supports_shots or tket_backend.supports_state):
            raise ValueError("pytket Backend must support shots or state.")
        self.tket_backend = tket_backend
        self.compilation_pass = (
            self.tket_backend.default_compilation_pass()
            if compilation_pass is None
            else compilation_pass
        )
        super().__init__(
            wires=wires, shots=shots, analytic=self.tket_backend.supports_state
        )

    def capabilities(self):
        cap_dic: Dict[str, Any] = super().capabilities().copy()
        cap_dic.update(
            {
                "supports_finite_shots": self.tket_backend.supports_shots,
                "returns_state": self.tket_backend.supports_state,
                "supports_inverse_operations": True,
            }
        )
        return cap_dic

    def reset(self):
        # Reset only internal data, not the options that are determined on
        # device creation
        self._circuit = Circuit(name="temp")
        self._reg = self._circuit.add_q_register("q", self.num_wires)
        self._creg = self._circuit.add_c_register("c", self.num_wires)
        self._state = None  # statevector of a simulator backend
        super().reset()

    def apply(self, operations, **kwargs):
        rotations = kwargs.get("rotations", [])

        self._circuit = pennylane_to_tk(
            operations + rotations,
            self._wire_map,
            self._reg,
            self._creg,
            measure=(not self.tket_backend.supports_state),
        )
        # These operations need to run for all devices
        compiled_c = self.compile(self._circuit)
        self.run(compiled_c)

    def compile(self, circuit: Circuit):
        """Compile the quantum circuit to target the provided compile_backend.
        If compile_backend is None, then the target is simply the
        backend.
        """
        compile_c = circuit.copy()
        self.compilation_pass.apply(compile_c)
        return compile_c

    def run(self, compiled_c: Circuit):
        """Run the compiled circuit, and query the result."""
        shots = None
        if self.tket_backend.supports_shots:
            shots = self.shots
            if compiled_c.n_gates_of_type(OpType.Measure) == 0:
                compiled_c.measure_all()
        handle = self.tket_backend.process_circuit(compiled_c, n_shots=shots)
        self._backres = self.tket_backend.get_result(handle)

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
        if self.state is None:
            return None
        prob = self.marginal_prob(np.abs(self.state) ** 2, wires)
        return prob

    def generate_samples(self):
        if self.tket_backend.supports_shots:
            self._samples = np.asarray(self._backres.get_shots(self._creg), dtype=int)
            return self._samples
        else:
            return super().generate_samples()

    @property
    def state(self):
        if self.tket_backend.supports_state:
            if self._state is None:
                self._state = self._backres.get_state(self._reg)
            return self._state
        raise AttributeError("Device does not support state.")
