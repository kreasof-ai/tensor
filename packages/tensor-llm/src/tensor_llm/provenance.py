"""Bind an inference plan to the runtime and kernel template implementations."""
import hashlib
from pathlib import Path


def implementation_hashes(provider='cuda'):
    import tensor.runtime.abi
    import tensor.runtime.signature
    import tensor.providers.cuda
    import tensor.providers.cuda_graph
    import tensor.artifacts.format
    modules = (tensor.runtime.abi, tensor.runtime.signature, tensor.providers.cuda,
               tensor.providers.cuda_graph, tensor.artifacts.format)
    names=('model', 'config', 'kernels', 'gguf', 'tokenizer', 'provenance')
    if provider=='webgpu':
        import tensor.providers.webgpu
        import tensor.providers.webgpu_contract
        modules=(tensor.runtime.abi,tensor.runtime.signature,tensor.providers.webgpu,
                 tensor.providers.webgpu_contract,tensor.artifacts.format)
        names+=('webgpu_kernels',)
    result = {module.__name__: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
              for module in modules}
    for name in names:
        result['tensor_llm.' + name] = hashlib.sha256(
            Path(__file__).with_name(name + '.py').read_bytes()).hexdigest()
    if provider=='webgpu':
        native=Path(tensor.providers.webgpu.__file__).parents[1]/'native'/'webgpu_plan.c'
        result['tensor.native.webgpu_plan']=hashlib.sha256(native.read_bytes()).hexdigest()
    return result
