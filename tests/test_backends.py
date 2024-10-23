import numpy as np
import pennylane as qml  # type: ignore
import pytest

from pytket.backends.backend import Backend
from pytket.backends.backend_exceptions import CircuitNotValidError
from pytket.extensions.cirq import CirqStateSampleBackend
from pytket.extensions.qiskit import AerBackend, AerStateBackend
from pytket.passes import SynthesiseTK as sample_pass

TEST_BACKENDS: list[Backend] = [
    AerStateBackend(),
    AerBackend(),
    CirqStateSampleBackend(),
]


def my_quantum_function(x, y):  # type: ignore
    qml.RZ(x, wires=0)
    qml.RX(y, wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RY(y, wires=1)
    qml.adjoint(qml.T(wires=0))
    qml.S(wires=0)
    qml.U1(x, wires=0)
    qml.U2(x, y, wires=0)
    qml.U3(x, y, x, wires=0)
    qml.PauliY(wires=1)
    qml.PauliX(wires=1)
    qml.PauliZ(wires=1)
    qml.CZ(wires=[1, 0])
    qml.CY(wires=[1, 0])
    qml.CRZ(x, wires=[1, 0])
    qml.CSWAP(wires=[1, 0, 2])
    qml.SWAP(wires=[1, 0])
    qml.Toffoli(wires=[1, 0, 2])

    return qml.expval(qml.PauliZ(1) @ qml.PauliX(0) @ qml.PauliY(2))


@pytest.mark.parametrize("test_backend", TEST_BACKENDS)
def test_backends(test_backend: Backend) -> None:
    dev = qml.device(
        "pytket.pytketdevice", wires=3, pytket_backend=test_backend, shots=100000
    )

    assert str(dev.compilation_pass) == "<tket::SequencePass>"

    test_func = qml.qnode(dev)(my_quantum_function)

    assert np.isclose([test_func(0.6, 0.8)], [0.274], atol=0.01)


def test_invalid_fail() -> None:
    dev = qml.device(
        "pytket.pytketdevice",
        wires=3,
        pytket_backend=AerStateBackend(),
        compilation_pass=sample_pass(),
    )
    assert str(dev.compilation_pass) == "<tket::BasePass>"

    test_func = qml.qnode(dev)(my_quantum_function)
    with pytest.raises(CircuitNotValidError):
        assert np.isclose([test_func(0.3, 0.2)], [0.084], rtol=0.05)
