# run battery of pennylane tests with default backend

from pennylane.devices.tests import test_device  # type: ignore

test_device("pytket.pytketdevice", shots=None, pytest_args=["-x", "-s"])
