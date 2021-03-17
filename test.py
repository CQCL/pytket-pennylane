import pennylane as qml

# print(qml.about())

# from pytket_pennylane import pytket_device

# from pennylane_qiskit import BasicAerDevice

# dev = qml.device("qiskit.aer", wires=2)
from pytket.extensions.qiskit import AerStateBackend

dev = qml.device(
    "pytket.mydevice",
    wires=2,
    tket_backend=AerStateBackend(),
    compilation_pass=AerStateBackend().default_compilation_pass(0),
)
print(dev.tket_backend)
print(dev.compilation_pass)
from pennylane.devices.tests import test_device

test_device("pytket.mydevice", pytest_args=["-x", "-s"])
