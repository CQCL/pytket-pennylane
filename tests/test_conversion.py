from collections import OrderedDict
from math import pi

import pennylane as qml  # type: ignore
from pennylane.operation import Operation  # type: ignore  # noqa: TC002

from pytket.circuit import BitRegister, Circuit, QubitRegister
from pytket.extensions.pennylane import pennylane_to_tk


def test_pennylane_to_tk() -> None:
    test_data: list[
        tuple[
            tuple[list[Operation], OrderedDict, QubitRegister, BitRegister, bool],
            Circuit,
        ]
    ] = [
        (
            ([], OrderedDict(), QubitRegister("q", 0), BitRegister("c", 0), False),
            Circuit(0, name="temp"),
        ),
        (
            ([], OrderedDict(), QubitRegister("a", 0), BitRegister("b", 0), True),
            Circuit(0, name="temp"),
        ),
        (
            (
                [qml.RZ(0.6, wires=[0])],
                OrderedDict([(0, 0)]),
                QubitRegister("q", 1),
                BitRegister("c", 1),
                True,
            ),
            Circuit(1, name="temp").Rz(0.6 / pi, 0).measure_all(),
        ),
        (
            (
                [qml.Hadamard(wires=[0]), qml.CNOT(wires=[0, 1])],
                OrderedDict([(0, 0), (1, 1)]),
                QubitRegister("q", 2),
                BitRegister("c", 0),
                False,
            ),
            Circuit(2, 0, name="temp").H(0).CX(0, 1),
        ),
        (
            (
                [qml.Hadamard(wires=[0]), qml.CNOT(wires=[0, 1])],
                OrderedDict([(0, 1), (1, 0)]),
                QubitRegister("q", 2),
                BitRegister("c", 2),
                True,
            ),
            Circuit(2, 2, name="temp").H(1).CX(1, 0).measure_all(),
        ),
    ]

    for (operations, wire_map, qreg, creg, measure), c in test_data:
        c1 = pennylane_to_tk(operations, wire_map, qreg, creg, measure)
        assert c == c1
