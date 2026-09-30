"""Opt-in checks for the independent Triton benchmark's numerical contracts."""
import os

import pytest

pytestmark = pytest.mark.skipif(os.environ.get('TENSOR_DIRECT_CUDA') != '1',
                                reason='set TENSOR_DIRECT_CUDA=1 with Torch/Triton/CUDA')


def exercise(operation, args, expected, *, atol=.002, rtol=.02):
    import torch
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.inference_mode(), torch.cuda.stream(stream):
        output = torch.empty_like(expected)
        operation.into(args, output)
        torch.testing.assert_close(output, expected, atol=atol, rtol=rtol, equal_nan=True)
        operation.compiled_into(args, output)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            operation.compiled_into(args, output)
        output.fill_(float('nan'))
        graph.replay()
        torch.testing.assert_close(output, expected, atol=atol, rtol=rtol, equal_nan=True)
        torch.testing.assert_close(operation.allocating_compiled(*args), expected,
                                   atol=atol, rtol=rtol, equal_nan=True)
        stream.synchronize()


def test_pointwise_tail_and_nan():
    import torch
    from benchmarks.inference.direct_triton_kernels import Operation
    a, b = (torch.randn(257, device='cuda') for _ in range(2))
    a[0] = float('nan')
    b[-1] = float('nan')
    exercise(Operation('pointwise', (a,b)), (a,b), (a*2+b).relu(), atol=0, rtol=0)


@pytest.mark.parametrize('relu', [False, True])
def test_linear_all_three_tails(relu):
    import torch
    from benchmarks.inference.direct_triton_kernels import Operation
    args = tuple(torch.randn(shape, device='cuda', dtype=torch.float16)
                 for shape in ((33,65), (67,65), (67,)))
    # Explicit FP32 accumulation/bias then FP16 rounding is Tensor's contract.
    expected = torch.nn.functional.linear(args[0].float(), args[1].float(), args[2].float()).half()
    if relu:
        expected = expected.relu()
    exercise(Operation('linear', args, relu=relu), args, expected)


@pytest.mark.parametrize('length,dim,causal',
                         [(17,64,True), (65,64,True), (129,128,False), (257,128,True)])
def test_attention_uniform_scores_short_causal_blocks_and_tails(length, dim, causal):
    import torch
    from benchmarks.inference.direct_triton_kernels import Operation
    shape = (2,2,length,dim)
    q, k = (torch.zeros(shape, device='cuda', dtype=torch.float16) for _ in range(2))
    v = ((torch.arange(q.numel(), device='cuda')%17-8)/8).half().reshape(shape)
    if causal:
        expected = (v.float().cumsum(2)/torch.arange(1,length+1,device='cuda').reshape(1,1,length,1)).half()
    else:
        expected = v.float().mean(2,keepdim=True).expand(shape).half()
    args = (q,k,v)
    exercise(Operation('attention',args,causal=causal), args, expected, atol=.001, rtol=.01)
