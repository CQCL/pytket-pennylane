import pennylane as qml
print(qml.about())

# from pytket_pennylane import pytket_device

# from pennylane_qiskit import BasicAerDevice

dev = qml.device('qiskit.aer', wires=2)

dev = qml.device('pytket.mydevice', wires=2)

from pennylane.devices.tests import test_device
test_device("pytket.mydevice", pytest_args=["-x", "-k", "measurements"])
