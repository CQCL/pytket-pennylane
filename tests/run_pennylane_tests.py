# run battery of pennylane tests with default backend


from pennylane.devices.tests import test_device

test_device("pytket.pytketdevice", shots=None, pytest_args=["-x", "-s"])
