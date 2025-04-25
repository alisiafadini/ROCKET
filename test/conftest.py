import sys
from unittest.mock import MagicMock

# Mock *before* pytest imports any rocket modules
sys.modules["openfold"] = MagicMock()
sys.modules["reciprocalspaceship"] = MagicMock()
sys.modules["SFC_Torch"] = MagicMock()
