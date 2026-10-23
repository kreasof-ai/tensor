"""Column copies preserve unrelated data, ordering and borrowed buffer lifetimes."""

import ctypes
import os
import numpy as np
import pytest
import tensor


def enabled(provider):
    if provider != 'cpu' and os.environ.get('TENSOR_'+provider.upper())!='1':
        pytest.skip('enable provider for device column copies')


@pytest.mark.parametrize('provider',['cpu','cuda','webgpu'])
@pytest.mark.parametrize('dtype,width,destination_width,offset',[
    ('float32',5,9,2),('float16',6,12,2),('int32',3,7,4),('uint32',4,8,0)])
def test_column_copy_preserves_untouched_columns_without_host_readback(provider,dtype,width,destination_width,offset,monkeypatch):
    enabled(provider)
    values=np.arange(3*width,dtype=dtype).reshape(3,width)
    initial=np.full((3,destination_width),7,dtype=dtype)
    expected=initial.copy();expected[:,offset:offset+width]=values
    with tensor.Device(provider=provider) as device:
        source=device.from_numpy(values);destination=device.from_numpy(initial)
        source_view=source.view((3,width));destination_view=destination.view((3,destination_width))
        with monkeypatch.context() as patch:
            patch.setattr(device,'_download',lambda *_:pytest.fail('column copy downloaded its source'))
            assert destination_view.copy_columns_from(source_view,offset) is destination_view
        np.testing.assert_array_equal(destination.to_numpy(),expected)
        np.testing.assert_array_equal(source.to_numpy(),values)
        destination_view.release();source_view.release()
        assert not source._released and not destination._released


@pytest.mark.parametrize('provider',['cpu','cuda','webgpu'])
def test_feature_packing_copies_five_disjoint_column_groups(provider):
    enabled(provider)
    rng=np.random.default_rng(41)
    arrays=[rng.normal(size=(33,2048)).astype('float32') for _ in range(5)]
    with tensor.Device(provider=provider) as device:
        destination=device.empty((33,5*2048),'float32')
        sources=[device.from_numpy(a) for a in arrays]
        for index,source in enumerate(sources):destination.copy_columns_from(source,index*2048)
        np.testing.assert_array_equal(destination.to_numpy(),np.concatenate(arrays,axis=1))
        # A copy onto the identical storage is a no-op even through another view.
        destination.copy_columns_from(destination.view(destination.shape))
        np.testing.assert_array_equal(destination.to_numpy(),np.concatenate(arrays,axis=1))


@pytest.mark.parametrize('provider',['cpu','cuda','webgpu'])
def test_odd_half_column_copy_obeys_backend_alignment(provider):
    enabled(provider)
    values=np.arange(9,dtype='float16').reshape(3,3)
    with tensor.Device(provider=provider) as device:
        source=device.from_numpy(values);destination=device.full((3,8),-1,'float16')
        if provider=='webgpu':
            with pytest.raises(ValueError,match='four-byte'):destination.copy_columns_from(source,1)
            np.testing.assert_array_equal(destination.to_numpy(),-np.ones((3,8),dtype='float16'))
        else:
            destination.copy_columns_from(source,1)
            expected=-np.ones((3,8),dtype='float16');expected[:,1:4]=values
            np.testing.assert_array_equal(destination.to_numpy(),expected)


@pytest.mark.parametrize('offset',[-1,True,1.5,7])
def test_column_copy_checks_bounds_before_changing_destination(offset):
    with tensor.Device(provider='cpu') as device:
        source=device.ones((2,3));destination=device.full((2,5),7.)
        with pytest.raises(ValueError,match='rows or columns'):destination.copy_columns_from(source,offset)
        np.testing.assert_array_equal(destination.to_numpy(),7)


def test_column_copy_checks_dtype_rank_rows_session_and_release():
    with tensor.Device(provider='cpu') as device,tensor.Device(provider='cpu') as other:
        destination=device.zeros((2,5),'float32')
        for source in (device.zeros((2,3),'int32'),device.zeros((6,),'float32'),device.zeros((1,3),'float32')):
            with pytest.raises(ValueError):destination.copy_columns_from(source)
        with pytest.raises(ValueError,match='same device'):destination.copy_columns_from(other.zeros((2,3),'float32'))
        source=device.zeros((2,3),'float32');source.release()
        with pytest.raises(RuntimeError,match='released'):destination.copy_columns_from(source)


def test_cpu_whole_copy_and_reshape_download_preserve_base_storage():
    with tensor.Device(provider='cpu') as device:
        source=device.from_numpy(np.arange(12,dtype='float32').reshape(3,4))
        destination=device.empty((12,),'float32')
        view=destination.view((3,4));view.copy_from(source)
        np.testing.assert_array_equal(view.to_numpy(),source.to_numpy())
        view.release();np.testing.assert_array_equal(destination.to_numpy(),np.arange(12))


def test_cuda_pitched_copy_descriptor_matches_cuda_driver_header_layout():
    from tensor.providers.cuda.driver import Memcpy3D
    assert ctypes.sizeof(Memcpy3D)==200
    assert Memcpy3D.srcPitch.offset==72
    assert Memcpy3D.dstPitch.offset==160
    assert Memcpy3D.WidthInBytes.offset==176
    assert Memcpy3D.Depth.offset==192
