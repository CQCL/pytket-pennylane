import pennylane as qml

# print(qml.about())

# from pytket_pennylane import pytket_device

# from pennylane_qiskit import BasicAerDevice

# dev = qml.device("qiskit.aer", wires=2)
from pytket.extensions.qiskit import AerStateBackend
from pytket.passes import RebaseHQS as test_pass
dev = qml.device(
    "pytket.pytketdevice",
    wires=2,
    tket_backend=AerStateBackend(),
    # compilation_pass=AerStateBackend().default_compilation_pass(0),
    compilation_pass=test_pass()
)
print(dev.tket_backend)
print(dev.compilation_pass)
from pennylane.devices.tests import test_device

test_device("pytket.pytketdevice", pytest_args=["-x", "-s"])
