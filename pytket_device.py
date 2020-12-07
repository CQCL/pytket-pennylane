#creating device class to build Pennylane plugin

import pennylane as qml

from ._version import __version__

PYTKET_OPERATION_MAP = {
    qml.Hadamard : OpType.H,
    qml.PauliX : OpType.X,
    qml.PauliY : OpType.Y,
    qml.PauliZ : OpType.Z,
    qml.S : OpType.S,
    qml.T : OpType.T,
    qml.RX : OpType.Rx, 
    qml.RY : OpType.Ry,
    qml.RZ : OpType.Rz,
    qml.CNOT : OpType.CX,
    qml.CY : OpType.CY,
    qml.CZ : OpType.CZ,
    qml.SWAP : OpType.SWAP,
    qml.U1 : OpType.U1,
    qml.U2 : OpType.U2,
    qml.U3 : OpType.U3,
    qml.CRZ : OpType.CRz,
    qml.Toffoli : OpType.CCX,
    qml.CSWAP : OpType.CSWAP
}

class pytketDevice(QubitDevice):
    """MyDevice docstring"""
    name = 'pytket-pennylane plugin'
    short_name = 'pytket.mydevice'
    pennylane_requires = '>=0.13.0'
    version = '0.13.0'
    plugin_version = __version__
    author = 'KN'

    _operation_map = {**PYTKET_OPERATION_MAP}
    operations = set(_operation_map.keys())
    observables = {"PauliX", "PauliY", "PauliZ"}
