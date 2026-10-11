"""PyTorch compatibility binding for Tensor."""

from tensor_torch.compile.backend import Backend, backend
from tensor_torch.compile.reporting import reports
from tensor_torch.kernel import Kernel
from tensor_torch.loading import load
from tensor_torch.runtime.session import close

__version__ = "0.1.0"
__all__ = ["Backend", "Kernel", "backend", "reports", "load", "close"]
