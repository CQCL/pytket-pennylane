import shutil
import os
from setuptools import setup, find_namespace_packages  # type: ignore

metadata: dict = {}
with open("_metadata.py") as fp:
    exec(fp.read(), metadata)
shutil.copy(
    "_metadata.py",
    os.path.join("pytket", "extensions", "pennylane", "_metadata.py"),
)

devices_list = [
    "pytket.pytketdevice=pytket.extensions.pennylane:PytketDevice",
]

setup(
    name=metadata["__extension_name__"],
    version=metadata["__extension_version__"],
    author="KN",
    author_email="seyon.sivarajah@cambridgequantum.com",
    python_requires=">=3.8",
    url="https://github.com/CQCL/pytket-pennylane",
    description="Pytket extension and Pennylane plugin.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="Apache 2",
    packages=find_namespace_packages(include=["pytket.*"]),
    include_package_data=True,
    install_requires=[
        "pytket ~= 1.5",
        "pennylane ~= 0.25.0",
        "pytket-qiskit ~= 0.28.0",
    ],
    classifiers=[
        "Environment :: Console",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: Other/Proprietary License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
    ],
    zip_safe=False,
    entry_points={"pennylane.plugins": devices_list},
)
