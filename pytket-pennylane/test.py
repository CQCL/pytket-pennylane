import pennylane as qml
print(qml.about())

# from pytket_pennylane import pytket_device

# from pennylane_qiskit import BasicAerDevice

#import pennylane as qml
#dev = qml.device('qiskit.aer', wires=2)

#import pennylane as qml
#dev = qml.device('pytket.mydevice', wires=2)

from pennylane.devices.tests import test_device
test_device("pytket.mydevice")
