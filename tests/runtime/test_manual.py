"""Manual backward ownership, gradient metadata and failure semantics."""
import pytest
from tensor.providers import Device
from tensor.manual import ManualFunction


def test_manual_backward_context_lifetime_and_optional_integer_gradient():
    with Device(provider='cpu') as device:
        x=device.ones((3,));tokens=device.zeros((3,),'int32');dy=device.ones((3,))
        def forward(ctx,x,tokens):ctx.save_for_backward(x);return x
        def backward(ctx,dy):assert ctx.saved_tensors==(x,);return dy,None
        operation=ManualFunction(forward,backward)
        output,context=operation.forward(x,tokens)
        assert output is x
        with pytest.raises(RuntimeError,match='previous'):operation.forward(x,tokens)
        assert operation.backward(context,dy)==(dy,None)
        assert context.consumed and context.saved_tensors==()
        with pytest.raises(RuntimeError,match='consumed'):operation.backward(context,dy)
        _,context=operation.forward(x,tokens);context.discard()
        _,context=operation.forward(x,tokens);operation.backward(context,dy)


def test_manual_backward_checks_devices_shapes_and_released_saved_buffers():
    with Device(provider='cpu') as device, Device(provider='cpu') as other:
        x=device.ones((3,));dy=device.ones((3,));saved=device.ones((3,));wrong=other.ones((3,))
        operation=ManualFunction(lambda ctx,x:(ctx.save_for_backward(saved),x)[1],lambda ctx,dy:dy)
        _,context=operation.forward(x)
        with pytest.raises(ValueError,match='shape, dtype and device'):operation.backward(context,wrong)
        with pytest.raises(ValueError,match='shape, dtype and device'):operation.backward(context,device.ones((2,)))
        saved.release()
        with pytest.raises(RuntimeError,match='released'):operation.backward(context,dy)
        context.discard()
        other_op=ManualFunction(lambda ctx,x:x,lambda ctx,dy:dy)
        _,context=other_op.forward(x)
        with pytest.raises(ValueError,match='another'):operation.backward(context,dy)
        context.discard()


def test_failed_backward_cannot_repeat_partial_accumulation():
    with Device(provider='cpu') as device:
        x=device.ones((3,))
        def fail(ctx,dy):raise ValueError('partial update')
        operation=ManualFunction(lambda ctx,x:x,fail)
        _,context=operation.forward(x)
        with pytest.raises(ValueError,match='partial'):operation.backward(context,x)
        assert context.consumed
        with pytest.raises(RuntimeError,match='consumed'):operation.backward(context,x)


def test_manual_backward_rejects_integer_gradient_and_wrong_callback_results():
    with Device(provider='cpu') as device:
        tokens=device.zeros((3,),'int32');y=device.ones((3,))
        operation=ManualFunction(lambda ctx,x:y,lambda ctx,dy:tokens)
        _,context=operation.forward(tokens)
        with pytest.raises(ValueError,match='integer'):operation.backward(context,y)
        operation=ManualFunction(lambda ctx,x:y,lambda ctx,dy:(y,y))
        _,context=operation.forward(y)
        with pytest.raises(ValueError,match='one gradient'):operation.backward(context,y)
