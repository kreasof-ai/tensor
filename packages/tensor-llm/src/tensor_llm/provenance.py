"""Bind an inference plan to the runtime and kernel template implementations."""
import hashlib
from pathlib import Path


def implementation_hashes():
    import tensor.runtime.abi
    import tensor.runtime.signature
    import tensor.providers.cuda
    import tensor.providers.cuda_graph
    import tensor.artifacts.format
    modules = (tensor.runtime.abi, tensor.runtime.signature, tensor.providers.cuda,
               tensor.providers.cuda_graph, tensor.artifacts.format)
    result = {module.__name__: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
              for module in modules}
    for name in ('model', 'config', 'kernels', 'gguf', 'tokenizer', 'provenance'):
        result['tensor_llm.' + name] = hashlib.sha256(
            Path(__file__).with_name(name + '.py').read_bytes()).hexdigest()
    return result
