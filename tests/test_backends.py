import pennylane as qml

# print(qml.about())

# from pytket_pennylane import pytket_device

# from pennylane_qiskit import BasicAerDevice

# dev = qml.device("qiskit.aer", wires=2)
from pytket.extensions.qiskit import AerStateBackend
from pytket.passes import RebaseHQS as test_pass



def test_backends():
    dev = qml.device(
        "pytket.pytketdevice",
        wires=2,
        tket_backend=AerStateBackend(),
        compilation_pass=AerStateBackend().default_compilation_pass(0),
        # compilation_pass=test_pass()
    )

    @qml.qnode(dev)
    def x_rotations(params):
        qml.RX(params[0], wires=0)
        qml.RX(params[1], wires=1)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0))
    print(dev.tket_backend.__extension_name__)
    print(dev.compilation_pass.get_config())
    print(x_rotations([0.1, 0.2]))