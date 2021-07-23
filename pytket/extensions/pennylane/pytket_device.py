from typing import Any, Dict, Optional

import numpy as np
from pytket.backends.backend import Backend
from pytket.passes import BasePass
from pennylane import QubitDevice

from pytket.extensions.qiskit import AerStateBackend
from pytket.extensions.pennylane import __extension_version__
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
    plugin_version = __extension_version__
    author = "KN"

    _operation_map = OPERATION_MAP
    operations = set(_operation_map.keys())
    observables = {"PauliX", "PauliY", "PauliZ", "Identity", "Hadamard"}

    def __init__(
        self,
        wires: int,
        shots: Optional[int] = None,
        pytket_backend: Backend = AerStateBackend(),
        optimisation_level: Optional[int] = None,
        compilation_pass: Optional[BasePass] = None,
    ):
        """Construct a device that use a Pytket Backend and compilation to
        execute circuits.

        :param wires: Number of wires
        :type wires: int
        :param shots: Number of shots to use (only relevant for sampling backends), defaults to None
        :type shots: Optional[int], optional
        :param pytket_backend: Pytket Backend class to use, defaults to AerStateBackend()
            to facilitate automated pennylane testing of this backend
        :type pytket_backend: Backend, optional
        :param optimisation_level: Backend default compilation optimisation level, ignored if `compilation_pass` is set,
         defaults to None
        :type optimisation_level: int, optional
        :param compilation_pass: Pytket compiler pass with which to compile circuits, defaults to None
        :type compilation_pass: Optional[BasePass], optional
        :raises ValueError: If the Backend does not support shots or state results
        """

        if not (pytket_backend.supports_shots or pytket_backend.supports_state):
            raise ValueError("pytket Backend must support shots or state.")
        self.pytket_backend = pytket_backend
        if compilation_pass is None:
            if optimisation_level is None:
                self.compilation_pass = self.pytket_backend.default_compilation_pass()
            else:
                self.compilation_pass = self.pytket_backend.default_compilation_pass(
                    optimisation_level
                )
        else:
            self.compilation_pass = compilation_pass
        super().__init__(wires=wires, shots=shots)

    def capabilities(self):
        cap_dic: Dict[str, Any] = super().capabilities().copy()
        cap_dic.update(
            {
                "supports_finite_shots": self.pytket_backend.supports_shots,
                "returns_state": self.pytket_backend.supports_state,
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
        self._backres = None
        self._state = None  # statevector of a simulator backend
        self._samples = None
        super().reset()

    def apply(self, operations, **kwargs):
        rotations = kwargs.get("rotations", [])

        self._circuit = pennylane_to_tk(
            operations + rotations,
            self._wire_map,
            self._reg,
            self._creg,
            measure=(not self.pytket_backend.supports_state),
        )
        # These operations need to run for all devices
        compiled_c = self.compile(self._circuit)
        self.run(compiled_c)

    def compile(self, circuit: Circuit):
        compile_c = circuit.copy()
        self.compilation_pass.apply(compile_c)
        return compile_c

    def run(self, compiled_c: Circuit):
        """Run the compiled circuit, and query the result."""
        shots = None
        if self.pytket_backend.supports_shots:
            shots = self.shots
            if compiled_c.n_gates_of_type(OpType.Measure) == 0:
                compiled_c.measure_all()
        handle = self.pytket_backend.process_circuit(compiled_c, n_shots=shots)
        self._backres = self.pytket_backend.get_result(handle)

    def analytic_probability(self, wires=None):
        if self.state is None:
            return None
        prob = self.marginal_prob(np.abs(self.state) ** 2, wires)
        return prob

    def generate_samples(self):
        if self.pytket_backend.supports_shots:
            self._samples = np.asarray(self._backres.get_shots(self._creg), dtype=int)
            return self._samples
        else:
            return super().generate_samples()

    @property
    def state(self):
        if self.pytket_backend.supports_state:
            if self._state is None:
                self._state = self._backres.get_state(self._reg)
            return self._state
        raise AttributeError("Device does not support state.")
