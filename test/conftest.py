import sys
from unittest.mock import MagicMock

to_mock = [
    "openfold",
    "openfold.model",
    "openfold.utils",
    "openfold.config",
    "openfold.data",
    "openfold.model.model",
    "openfold.np",
    "openfold.utils.import_weights",
    "openfold.utils.script_utils",
    "reciprocalspaceship",
    "SFC_Torch",
]

for mod in to_mock:
    sys.modules[mod] = MagicMock()
