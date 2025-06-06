# Copyright Quantinuum
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from collections.abc import Iterable
from typing import Any, cast

import numpy as np

from pennylane.devices import QubitDevice  # type: ignore
from pennylane.operation import Operation  # type: ignore
from pytket.backends.backend import Backend
from pytket.backends.backendresult import BackendResult  # noqa: TC001
from pytket.circuit import Circuit, OpType
from pytket.extensions.pennylane import __extension_version__
from pytket.extensions.qiskit import AerStateBackend
from pytket.passes import BasePass

from .pennylane_convert import (
    OPERATION_MAP,
    pennylane_to_tk,
)


class PytketDevice(QubitDevice):
    """PytketDevice allows pytket backends and compilation to be used as Pennylane
    devices."""

    name = "pytket-pennylane plugin"
    short_name = "pytket.pytketdevice"
    pennylane_requires = ">=0.40.0"
    version = "0.15.0"
    plugin_version = __extension_version__
    author = "KN"

    _operation_map = OPERATION_MAP
    operations = set(_operation_map.keys())  # noqa: RUF012
    observables = {"PauliX", "PauliY", "PauliZ", "Identity", "Hadamard", "Prod"}  # noqa: RUF012

    def __init__(
        self,
        wires: int,
        shots: int | None = None,
        pytket_backend: Backend = AerStateBackend(),  # noqa: B008
        optimisation_level: int | None = None,
        compilation_pass: BasePass | None = None,
    ):
        """Construct a device that use a Pytket Backend and compilation to
        execute circuits.

        :param wires: Number of wires
        :param shots: Number of shots to use (only relevant for sampling backends),
            defaults to None
        :param pytket_backend: Pytket Backend class to use, defaults to
            AerStateBackend() to facilitate automated pennylane testing of this backend
        :param optimisation_level: Backend default compilation optimisation level,
            ignored if `compilation_pass` is set, defaults to None
        :param compilation_pass: Pytket compiler pass with which to compile circuits,
            defaults to None
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

    def capabilities(self) -> dict[str, Any]:
        """See :py:meth:`pennylane.devices._qubit_device.QubitDevice.capabilities`"""
        cap_dic: dict[str, Any] = super().capabilities().copy()
        cap_dic.update(
            {
                "supports_finite_shots": self.pytket_backend.supports_shots,
                "returns_state": self.pytket_backend.supports_state,
                "supports_inverse_operations": True,
            }
        )
        return cap_dic

    def reset(self) -> None:
        # Reset only internal data, not the options that are determined on
        # device creation
        self._circuit = Circuit(name="temp")
        self._reg = self._circuit.add_q_register("q", self.num_wires)
        self._creg = self._circuit.add_c_register("c", self.num_wires)
        self._backres: BackendResult | None = None
        self._state: np.ndarray | None = None  # statevector of a simulator backend
        self._samples: np.ndarray | None = None
        super().reset()

    def apply(
        self, operations: list[Operation], rotations: list[Operation] | None = None
    ) -> None:
        """See :py:meth:`pennylane.devices._qubit_device.QubitDevice.apply`"""
        self._circuit = pennylane_to_tk(
            operations if rotations is None else operations + rotations,
            self._wire_map,
            self._reg,
            self._creg,
            measure=(not self.pytket_backend.supports_state),
        )
        # These operations need to run for all devices
        compiled_c = self.compile(self._circuit)
        self.run(compiled_c)

    def compile(self, circuit: Circuit) -> Circuit:
        compile_c = circuit.copy()
        self.compilation_pass.apply(compile_c)
        return compile_c

    def run(self, compiled_c: Circuit) -> None:
        """Run the compiled circuit, and query the result."""
        shots = None
        if self.pytket_backend.supports_shots:
            shots = self.shots
            if compiled_c.n_gates_of_type(OpType.Measure) == 0:
                compiled_c.measure_all()
        handle = self.pytket_backend.process_circuit(compiled_c, n_shots=shots)
        self._backres = self.pytket_backend.get_result(handle)

    def analytic_probability(
        self, wires: int | Iterable[int] | None = None
    ) -> np.ndarray:
        """See :py:meth:`pennylane.devices._qubit_device.QubitDevice.analytic_probability`"""
        prob = self.marginal_prob(np.abs(self.state) ** 2, wires)
        return cast("np.ndarray", prob)

    def generate_samples(self) -> np.ndarray:
        """See :py:meth:`pennylane.devices._qubit_device.QubitDevice.generate_samples`"""
        if self.pytket_backend.supports_shots:
            if self._backres is None:
                raise RuntimeError("Result does not exist.")
            self._samples = np.asarray(
                self._backres.get_shots(self._creg.to_list()), dtype=int
            )
            return self._samples
        return cast("np.ndarray", super().generate_samples())

    @property
    def state(self) -> np.ndarray:
        if self.pytket_backend.supports_state:
            if self._state is None:
                if self._backres is None:
                    raise RuntimeError("Result does not exist.")
                self._state = self._backres.get_state(self._reg.to_list())
            return self._state
        raise AttributeError("Device does not support state.")
