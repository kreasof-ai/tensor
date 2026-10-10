"""Single isolated launch for Nsight Compute; never a throughput benchmark."""
import argparse
import ctypes as ct
import json
import numpy as np
import tensor


def main():
    p=argparse.ArgumentParser();p.add_argument('--artifact',required=True)
    p.add_argument('--copy',action='store_true');a=p.parse_args()
    with tensor.Device() as dev:
        kernel=dev.load(a.artifact)
        if a.copy:
            args=[dev.empty((268435456,),'uint32') for _ in range(2)]
            fill=dev.driver.lib.cuMemsetD32_v2
            fill.argtypes=[ct.c_uint64,ct.c_uint,ct.c_size_t];fill.restype=ct.c_int
            dev.driver.call('cuMemsetD32_v2',args[0].pointer,123456789,268435456)
        else:
            rng=np.random.default_rng(90210);s,c,cap,d=8,512,48000,256;rows=s*c
            args=[dev.from_numpy(rng.standard_normal((rows,16,d),dtype='float32'),dtype='bfloat16'),
                  *[dev.from_numpy(rng.integers(0,112,(s,2,cap,d),dtype='uint8')) for _ in range(2)],
                  *[dev.from_numpy(rng.uniform(.001,.02,(s,2,cap,2)).astype('float32')) for _ in range(2)],
                  dev.from_numpy(rng.standard_normal((rows,8192),dtype='float32')),
                  dev.from_numpy(np.full(s,32000,'int32')),dev.from_numpy(np.full(s,c,'int32')),
                  dev.empty((rows,4096),'bfloat16')]
        kernel.launch(*args);dev.synchronize()
        start=dev.driver.lib.cuProfilerStart;stop=dev.driver.lib.cuProfilerStop
        start.argtypes=[];start.restype=ct.c_int;stop.argtypes=[];stop.restype=ct.c_int
        dev.driver.call('cuProfilerStart')
        kernel.launch(*args);dev.synchronize()
        dev.driver.call('cuProfilerStop')
        print(json.dumps(dict(device=dev.info,artifact=a.artifact,scope='isolated diagnostic')))


if __name__=='__main__':main()
