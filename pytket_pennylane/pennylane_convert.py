import numpy as np
from math import pi
from typing import Dict, List, Optional, Tuple, Union
import warnings
import sympy

import pennylane as qml

from pytket.circuit import Circuit, Op, OpType, Qubit, Bit, Node, CircBox, Unitary2qBox, UnitType
from pytket.device import Device, QubitErrorContainer
from pytket.routing import Architecture, FullyConnected
#
from qiskit import (
    ClassicalRegister,
    QuantumCircuit,  # type: ignore
    QuantumRegister,
)
from qiskit.circuit import (
    Barrier,
    Instruction,
    Measure,
    Parameter,
    ParameterExpression,
    Reset,
)
import qiskit.circuit.library.standard_gates as qiskit_gates
from qiskit.extensions.unitary import UnitaryGate

_known_qiskit_gate = {
    # Exact equivalents (same signature except for factor of pi in each parameter):
    qiskit_gates.C3XGate: OpType.CnX,
    qiskit_gates.C4XGate: OpType.CnX,
    qiskit_gates.CCXGate: OpType.CCX,
    qiskit_gates.CHGate: OpType.CH,
    qiskit_gates.CPhaseGate: OpType.CU1,
    qiskit_gates.CRYGate: OpType.CnRy,
    qiskit_gates.CRZGate: OpType.CRz,
    qiskit_gates.CSwapGate: OpType.CSWAP,
    qiskit_gates.CU1Gate: OpType.CU1,
    qiskit_gates.CU3Gate: OpType.CU3,
    qiskit_gates.CXGate: OpType.CX,
    qiskit_gates.CYGate: OpType.CY,
    qiskit_gates.CZGate: OpType.CZ,
    qiskit_gates.HGate: OpType.H,
    qiskit_gates.IGate: OpType.noop,
    qiskit_gates.iSwapGate: OpType.ISWAPMax,
    qiskit_gates.PhaseGate: OpType.U1,
    qiskit_gates.RGate: OpType.PhasedX,
    qiskit_gates.RXGate: OpType.Rx,
    qiskit_gates.RXXGate: OpType.XXPhase,
    qiskit_gates.RYGate: OpType.Ry,
    qiskit_gates.RYYGate: OpType.YYPhase,
    qiskit_gates.RZGate: OpType.Rz,
    qiskit_gates.RZZGate: OpType.ZZPhase,
    qiskit_gates.SdgGate: OpType.Sdg,
    qiskit_gates.SGate: OpType.S,
    qiskit_gates.SwapGate: OpType.SWAP,
    qiskit_gates.SXdgGate: OpType.Vdg,
    qiskit_gates.SXGate: OpType.V,
    qiskit_gates.TdgGate: OpType.Tdg,
    qiskit_gates.TGate: OpType.T,
    qiskit_gates.U1Gate: OpType.U1,
    qiskit_gates.U2Gate: OpType.U2,
    qiskit_gates.U3Gate: OpType.U3,
    qiskit_gates.UGate: OpType.U3,
    qiskit_gates.XGate: OpType.X,
    qiskit_gates.YGate: OpType.Y,
    qiskit_gates.ZGate: OpType.Z,
    # Multi-controlled gates (qiskit expects a list of controls followed by the target):
    qiskit_gates.MCXGate: OpType.CnX,
    qiskit_gates.MCXGrayCode: OpType.CnX,
    qiskit_gates.MCXRecursive: OpType.CnX,
    qiskit_gates.MCXVChain: OpType.CnX,
    # Special types:
    Barrier: OpType.Barrier,
    Instruction: OpType.CircBox,
    Measure: OpType.Measure,
    Reset: OpType.Reset,
    UnitaryGate: OpType.Unitary2qBox,
}
# Not included in the above list:
# qiskit_gates.CUGate != OpType.CU3 : CUGate has an extra phase parameter

# Some qiskit gates are aliases (e.g. UGate and U3Gate).
# In such cases this reversal will select one or the other.
_known_qiskit_gate_rev = {v: k for k, v in _known_qiskit_gate.items()}

# Ensure U3 maps to U3Gate. (UGate not yet fully supported in Qiskit.)
_known_qiskit_gate_rev[OpType.U3] = qiskit_gates.U3Gate
#
_OPERATION_MAP = {
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
_INV_OPERATION_MAP = dict((reversed(item) for item in _OPERATION_MAP.items()))

#tk->qiskit->qml
def param_to_qiskit(
    p: sympy.Expr, symb_map: Dict[Parameter, sympy.Symbol]
) -> Union[float, ParameterExpression]:
    ppi = p * sympy.pi
    if len(ppi.free_symbols) == 0:
        return float(ppi.evalf())
    else:
        return ParameterExpression(symb_map, ppi)

def append_tk_command_to_qiskit(
    op, args, qcirc, qregmap, cregmap, symb_map
) -> Instruction:
    optype = op.type
    if optype == OpType.Measure:
        qubit = args[0]
        bit = args[1]
        qb = qregmap[qubit.reg_name][qubit.index[0]]
        b = cregmap[bit.reg_name][bit.index[0]]
        return qcirc.measure(qb, b)

    if optype == OpType.Reset:
        qb = qregmap[args[0].reg_name][args[0].index[0]]
        return qcirc.reset(qb)

    if optype in [OpType.CircBox, OpType.ExpBox, OpType.PauliExpBox]:
        subcircuit = op.get_circuit()
        subqc = tk_to_qiskit(subcircuit)
        n_qb = subcircuit.n_qubits
        qargs = []
        cargs = []
        for a in args:
            if a.type == UnitType.qubit:
                qargs.append(qregmap[a.reg_name][a.index[0]])
            else:
                cargs.append(cregmap[a.reg_name][a.index[0]])
        return qcirc.append(subqc.to_instruction(), qargs, cargs)
    if optype == OpType.Unitary2qBox:
        qargs = [qregmap[q.reg_name][q.index[0]] for q in args]
        u = op.get_matrix()
        g = UnitaryGate(u)
        return qcirc.append(g, qargs=qargs)
    if optype == OpType.Barrier:
        qargs = [qregmap[q.reg_name][q.index[0]] for q in args]
        g = Barrier(len(args))
        return qcirc.append(g, qargs=qargs)
    if optype == OpType.ConditionalGate:
        width = op.width
        regname = args[0].reg_name
        if len(cregmap[regname]) != width:
            raise NotImplementedError("OpenQASM conditions must be an entire register")
        for i, a in enumerate(args[:width]):
            if a.reg_name != regname:
                raise NotImplementedError(
                    "OpenQASM conditions can only use a single register"
                )
            if a.index != [i]:
                raise NotImplementedError(
                    "OpenQASM conditions must be an entire register in order"
                )
        instruction = append_tk_command_to_qiskit(
            op.op, args[width:], qcirc, qregmap, cregmap, symb_map
        )

        instruction.c_if(cregmap[regname], op.value)
        return instruction
    # normal gates
    qargs = [qregmap[q.reg_name][q.index[0]] for q in args]
    if optype == OpType.CnX:
        return qcirc.mcx(qargs[:-1], qargs[-1])
    # others are direct translations
    try:
        gatetype = _known_qiskit_gate_rev[optype]
    except KeyError as error:
        raise NotImplementedError(
            "Cannot convert tket Op to Qiskit gate: " + op.get_name()
        ) from error
    params = [param_to_qiskit(p, symb_map) for p in op.params]
    g = gatetype(*params)
    return qcirc.append(g, qargs=qargs)

def tk_to_qiskit(
    tkcirc: Circuit,
) -> Union[QuantumCircuit, Tuple[QuantumCircuit, sympy.Expr]]:
    """Convert back

    :param tkcirc: A circuit to be converted
    :type tkcirc: Circuit
    :return: The converted circuit
    :rtype: QuantumCircuit
    """
    tkc = tkcirc
    qcirc = QuantumCircuit(name=tkc.name)
    qreg_sizes: Dict[str, int] = {}
    for qb in tkc.qubits:
        if len(qb.index) != 1:
            raise NotImplementedError("Qiskit registers must use a single index")
        if (qb.reg_name not in qreg_sizes) or (qb.index[0] >= qreg_sizes[qb.reg_name]):
            qreg_sizes.update({qb.reg_name: qb.index[0] + 1})
    creg_sizes: Dict[str, int] = {}
    for b in tkc.bits:
        if len(b.index) != 1:
            raise NotImplementedError("Qiskit registers must use a single index")
        if (b.reg_name not in creg_sizes) or (b.index[0] >= creg_sizes[b.reg_name]):
            creg_sizes.update({b.reg_name: b.index[0] + 1})
    qregmap = {}
    for reg_name, size in qreg_sizes.items():
        qis_reg = QuantumRegister(size, reg_name)
        qregmap.update({reg_name: qis_reg})
        qcirc.add_register(qis_reg)
    cregmap = {}
    for reg_name, size in creg_sizes.items():
        qis_reg = ClassicalRegister(size, reg_name)
        cregmap.update({reg_name: qis_reg})
        qcirc.add_register(qis_reg)
    symb_map = {Parameter(str(s)): s for s in tkc.free_symbols()}
    for command in tkc:
        append_tk_command_to_qiskit(
            command.op, command.args, qcirc, qregmap, cregmap, symb_map
        )
    try:
        a = float(tkc.phase)
        qcirc.global_phase += a * pi
    except TypeError:
        warnings.warn("Qiskit circuits cannot have symbolic global phase: ignoring.")
    return qcirc

def tket_qiskit_qml(tkcircuit):
    qmlcircuit=qml.from_qiskit(tk_to_qiskit(tkcircuit))
    return qmlcircuit

