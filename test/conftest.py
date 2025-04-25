import sys
from unittest.mock import MagicMock

# Mock the package and all submodules you import from
sys.modules["openfold"] = MagicMock()
sys.modules["openfold.model"] = MagicMock()
sys.modules["openfold.utils"] = MagicMock()
sys.modules["openfold.config"] = MagicMock()
sys.modules["openfold.data"] = MagicMock()
sys.modules["openfold.model.model"] = MagicMock()
sys.modules["reciprocalspaceship"] = MagicMock()
sys.modules["SFC_Torch"] = MagicMock()
