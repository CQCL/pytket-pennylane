# run battery of pennylane tests with default backend

import platform
from pennylane.devices.tests import test_device  # type: ignore

pytest_args = ["-x", "-s"]

# if platform.system() == "Darwin":
#     # TODO Remove this exclusion.
#     # https://github.com/CQCL/pytket-pennylane/issues/2
#     pytest_args.extend(["-k", "not test_supported_gate_two_wires_with_parameters"])

test_device("pytket.pytketdevice", shots=None, pytest_args=pytest_args)
