# Copyright 2018 Carsten Blank

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python3
from setuptools import setup

with open("pytket-pennylane/_version.py") as f:
    version = f.readlines()[-1].split()[-1].strip("\"'")

requirements = [
    "numpy"
]

info = {
    'name': 'pytket-pennylane',
    'version': version,
    'maintainer': 'Xanadu',
    'maintainer_email': 'software@xanadu.ai',
    'url': 'https://github.com/XanaduAI/pennylane-qiskit',
    'license': 'Apache License 2.0',
    'packages': [
        'pennylane_qiskit'
    ]

}

classifiers = [
    "Natural Language :: English",
    "Programming Language :: Python",
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3 :: Only',
]

devices_list = [
        'pytket.mydevice = pytket-pennylane.pytket_device:pytketDevice'
    ]

setup(classifiers=classifiers, **(info), entry_points={'pennylane.plugins': devices_list})
