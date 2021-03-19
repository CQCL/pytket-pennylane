# pytket-pennylane
[Pytket](https://cqcl.github.io/pytket) extension and [PennyLane](https://github.com/PennyLaneAI/pennylane) plugin which allows pytket backends and compilation to be used as a PennyLane device.


Pytket is a quantum SDK python package which provides state of the art compilation for quantum
circuits and a unified interface for execution on a number of "backends" (devices and simulators).
PennyLane is a package for differentiable programming of quantum computer, which also provides a way
to execute circuits on a variety of "devices". This package allows users to easily leverage the 
differentiablecircuits of PennyLane combined with the compilation available in Pytket.

The package is available for python 3.7 and above and can be installed by cloning and installing from source, or via pip:
```bash
pip install pytket-pennylane
```

See the PennyLane [documentation](https://pennylane.readthedocs.io) and Pytket [documentation](https://cqcl.github.io/pytket) to get an intro to the packages.

To use the integration once installed, initialise your pytket backend (in this example, an `AerBackend` which uses Qiskit Aer), and construct a PennyLane `PytketDevice` using this backend:

```python
import pennylane as qml
from pytket.extensions.qiskit import AerBackend

# initialise pytket backend
pytket_backend = AerBackend()

# construct PennyLane device
dev = qml.device(
    "pytket.pytketdevice",
    wires=2,
    pytket_backend=pytket_backend,
    shots=1000
)

# define a PennyLane Qnode with this device
@qml.qnode(dev)
def my_quantum_function(x, y):
    qml.RZ(x, wires=0)
    qml.RX(y, wires=1)
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

# call the node
print(my_quantum_function(0.1, 0.2))

```

The example above uses the Pytket default compilation pass for the backend, you can change the optimisation
level of the default backend pass (0, 1 or 2) by setting the `optimisation_level` parameter:

```python
dev = qml.device(
    "pytket.pytketdevice",
    wires=2,
    pytket_backend=pytket_backend,
    optimisation_level=2,
    shots=1000
)
```

You can also use any Pytket [compilation pass](https://cqcl.github.io/pytket/build/html/manual_compiler.html) using the `compilation_pass` parameter, which is used instead of the default pass:

```python
from pytket.passes import PauliSimp, SequencePass

# use a Chemistry optimised pass before the backend's default pass

custom_pass = SequencePass([PauliSimp(), pytket_backend.default_compilation_pass()])

dev = qml.device(
    "pytket.pytketdevice",
    wires=2,
    pytket_backend=pytket_backend,
    compilation_pass=custom_pass,
    shots=1000
)

```