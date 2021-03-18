import platform
import numpy as np
import pytest
import pennylane as qml
from pytket.extensions.qiskit import AerStateBackend, AerBackend
from pytket.extensions.cirq import CirqStateSampleBackend
from pytket.extensions.projectq import ProjectQBackend

from pytket.passes import RebaseHQS as test_pass
from pytket.backends.backend_exceptions import CircuitNotValidError

if platform.system().lower() != "windows":
    from pytket.extensions.qulacs import QulacsBackend

def gen_decorated_func(dev):
    @qml.qnode(dev)
    def my_quantum_function(x, y):
        qml.RZ(x, wires=0)
        qml.RX(y, wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RY(y, wires=1)
        qml.T(wires=0).inv()
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
    
    return my_quantum_function


def test_backends():
    test_backends = [AerStateBackend(), AerBackend(), CirqStateSampleBackend(), ProjectQBackend()]
    if platform.system().lower() != "windows":
        test_backends.append(QulacsBackend())

    for back in test_backends:
        print(back)
        dev = qml.device(
            "pytket.pytketdevice",
            wires=3,
            tket_backend=back,
            shots=100000
        )

        assert dev.compilation_pass.get_config()["name"] == "SequencePass"
        
        test_func = gen_decorated_func(dev)

        assert np.isclose([test_func(0.6, 0.8)], [0.274], atol=0.01)

    # check invalid pass fails
    dev = qml.device(
        "pytket.pytketdevice",
        wires=3,
        tket_backend=AerStateBackend(),
        compilation_pass=test_pass()
    )
    assert dev.compilation_pass.get_config()["name"] == "RebaseHQS"

    test_func = gen_decorated_func(dev)
    with pytest.raises(CircuitNotValidError):
        assert np.isclose([test_func(0.3, 0.2)], [0.084], rtol=0.05)
