"""Exact rollback against every frozen recurrent checkpoint and inactive slot."""
import os
import numpy as np
import pytest


@pytest.mark.skipif(os.environ.get('TENSOR_QWEN_CUDA')!='1',reason='physical verifier rollback qualification')
@pytest.mark.parametrize('chunk',[64,128])
def test_recompute_matches_every_frozen_prefix(tmp_path,chunk):
    import tensor
    from tensor.compiler.entry import export_source
    from tensor.compiler.build import build_artifact
    s,c=3,chunk;r=s*c;rng=np.random.default_rng(71063)
    artifacts={}
    for name,module,factory,args in (
        ('control_scan','speculative','make_kernel',('gdn_scan',dict(slots=s,chunk=c))),
        ('scan','prefill','make_kernel',('gdn_scan',dict(slots=s,chunk=c))),
        ('control_conv','speculative','make_kernel',('gdn_conv',dict(slots=s,chunk=c))),
        ('conv','prefill','make_kernel',('gdn_conv',dict(slots=s,chunk=c))),
        *((name,'recompute',name,(dict(slots=s,chunk=c),))
          for name in ('save_scan','save_conv','restore_scan','restore_conv'))):
        entry=tmp_path/(name+'.py');artifact=entry.with_suffix('.tbin')
        entry.write_text(export_source('tensor_llm.qwen35.kernels.'+module,factory,*args,
            dependencies=('tensor.compiler.entry',)))
        build_artifact(entry,artifact,target='sm_90',compiler='nvrtc',
            nvrtc_home=os.environ.get('TENSOR_NVRTC_HOME','build/nvrtc-12.9'))
        artifacts[name]=artifact
    with tensor.Device() as dev:
        kernels={n:dev.load(p) for n,p in artifacts.items()}
        lengths=np.array([c,c//2+5,0],'int32');lens=dev.from_numpy(lengths)
        inputs=[dev.from_numpy(rng.normal(0,.025,(r,32,128)).astype('float32')) for _ in range(3)]
        g=dev.from_numpy(rng.uniform(-.2,-.01,(r,32)).astype('float32'))
        beta=dev.from_numpy(rng.uniform(.01,.8,(r,32)).astype('float32'))
        initial_values=rng.normal(0,.1,(s,32,128,128)).astype('float32')
        control_state=dev.from_numpy(initial_values);state=dev.from_numpy(initial_values)
        checkpoints=dev.empty((s,c-1,32,128,128));out=dev.empty((r,32,128));control_out=dev.empty(out.shape)
        initial=dev.empty(state.shape)
        saved=[dev.empty(x.shape) for x in (*inputs,g,beta)]
        replay_lengths=dev.empty((s,),'int32')
        kernels['save_scan'].launch(state,*inputs,g,beta,initial,*saved)
        kernels['control_scan'].launch(*inputs,g,beta,lens,control_state,control_out,checkpoints)
        kernels['scan'].launch(*inputs,g,beta,lens,state,out)
        np.testing.assert_array_equal(out.to_numpy(),control_out.to_numpy())
        final=control_state.to_numpy();np.testing.assert_array_equal(state.to_numpy(),final)
        frozen=checkpoints.to_numpy()
        x=dev.from_numpy(rng.normal(0,.2,(r,8192)).astype('float32'))
        weights=dev.from_numpy(rng.normal(0,.1,(8192,1,4)).astype('float32'),dtype='bfloat16')
        history_values=rng.normal(0,.1,(s,8192,3)).astype('float32')
        control_history=dev.from_numpy(history_values);history=dev.from_numpy(history_values)
        conv_out=dev.empty(x.shape);reference_out=dev.empty(x.shape)
        saved_x=dev.empty(x.shape,'bfloat16');initial_history=dev.empty(history.shape)
        history_checkpoints=dev.empty((s,c-1,8192,3))
        kernels['save_conv'].launch(x,history,saved_x,initial_history)
        kernels['control_conv'].launch(x,weights,lens,control_history,reference_out,history_checkpoints)
        kernels['conv'].launch(x,weights,lens,history,conv_out)
        np.testing.assert_array_equal(conv_out.to_numpy(),reference_out.to_numpy())
        final_history=control_history.to_numpy();np.testing.assert_array_equal(history.to_numpy(),final_history)
        frozen_history=history_checkpoints.to_numpy()
        for depth in range(1,c+1):
            counts=np.minimum(lengths,depth).astype('int32');accepted=dev.from_numpy(counts)
            # Begin from fully verified state; full acceptance must keep it.
            dev.driver.call('cuMemcpyDtoD_v2',state.pointer,control_state.pointer,state.nbytes)
            dev.driver.call('cuMemcpyDtoD_v2',history.pointer,control_history.pointer,history.nbytes)
            kernels['restore_scan'].launch(initial,accepted,lens,state,replay_lengths)
            kernels['scan'].launch(*saved,replay_lengths,state,out)
            kernels['restore_conv'].launch(initial_history,saved_x,accepted,lens,history)
            expected=final.copy();expected_history=final_history.copy()
            for slot,count in enumerate(counts):
                if 0<count<lengths[slot]:
                    expected[slot]=frozen[slot,count-1]
                    expected_history[slot]=frozen_history[slot,count-1]
            np.testing.assert_array_equal(state.to_numpy(),expected)
            np.testing.assert_array_equal(history.to_numpy(),expected_history)
            accepted.release()


@pytest.mark.skipif(os.environ.get('TENSOR_QWEN_CUDA')!='1',reason='physical large-copy qualification')
def test_c64_input_copy_at_int32_bitcount_boundary(tmp_path):
    import tensor
    from tensor.compiler.entry import export_source
    from tensor.compiler.build import build_artifact
    from tensor.runtime.dtypes import encode_bfloat16,decode_bfloat16
    source=tmp_path/'save_conv.py';artifact=source.with_suffix('.tbin')
    source.write_text(export_source('tensor_llm.qwen35.kernels.recompute','save_conv',
        dict(slots=64,chunk=128),dependencies=('tensor.compiler.entry',)))
    build_artifact(source,artifact,target='sm_90',compiler='nvrtc',
        nvrtc_home=os.environ.get('TENSOR_NVRTC_HOME','build/nvrtc-12.9'))
    rng=np.random.default_rng(50293)
    x=rng.standard_normal((8192,8192),dtype=np.float32)
    history=rng.standard_normal((64,8192,3),dtype=np.float32)
    with tensor.Device() as device:
        xx=device.from_numpy(x);state=device.from_numpy(history)
        saved=device.empty(x.shape,'bfloat16');initial=device.empty(history.shape)
        device.load(artifact).launch(xx,state,saved,initial)
        np.testing.assert_array_equal(saved.to_numpy(),decode_bfloat16(encode_bfloat16(x)))
        np.testing.assert_array_equal(initial.to_numpy(),history)


@pytest.mark.skipif(os.environ.get('TENSOR_QWEN_CUDA')!='1',reason='physical large-snapshot qualification')
def test_c64_short_snapshot_restore_without_flat_bitcount_view(tmp_path):
    import tensor
    from tensor.compiler.entry import export_source
    from tensor.compiler.build import build_artifact
    source=tmp_path/'restore.py';artifact=source.with_suffix('.tbin')
    source.write_text(export_source('tensor_llm.qwen35.kernels.hopper_restore','restore_kernel',
        dict(slots=64,chunk=4,shape=[32,128,128]),dependencies=('tensor.compiler.entry',)))
    build_artifact(source,artifact,target='sm_90',compiler='nvrtc',
        nvrtc_home=os.environ.get('TENSOR_NVRTC_HOME','build/nvrtc-12.9'))
    rng=np.random.default_rng(50294)
    checkpoints=rng.standard_normal((64,3,32,128,128),dtype=np.float32)
    final=rng.standard_normal((64,32,128,128),dtype=np.float32)
    lengths=np.resize(np.array([4,3,1,0,4,2,3,1],'int32'),64)
    counts=np.resize(np.array([1,2,1,0,4,2,2,1],'int32'),64)
    expected=final.copy()
    for slot,count in enumerate(counts):
        if 0<count<lengths[slot]:expected[slot]=checkpoints[slot,count-1]
    with tensor.Device() as device:
        cp=device.from_numpy(checkpoints);state=device.from_numpy(final)
        accepted=device.from_numpy(counts);ll=device.from_numpy(lengths)
        device.load(artifact).launch(cp,state,accepted,ll)
        np.testing.assert_array_equal(state.to_numpy(),expected)
