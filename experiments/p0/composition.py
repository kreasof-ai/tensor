"""Bounded pointwise inlining of two serialized frontend TIRx artifacts.

Supports one producer store and identity-indexed consumer loads only. This is
an architecture probe, not a general fusion engine or a stable artifact ABI.
"""


def compose(producer, consumer):
    import tilelang
    tir = tilelang.tvm.tirx
    stores = []
    tir.stmt_functor.post_order_visit(producer.body,
        lambda node: stores.append(node) if isinstance(node,tir.BufferStore) else None)
    if len(stores) != 1:
        raise ValueError("P0 composition requires exactly one producer store")
    store = stores[0]
    if len(producer.params)!=3 or len(consumer.params)!=2 or len(store.indices)!=1:
        raise ValueError("P0 composition requires affine producer and unary 1D consumer")
    source = consumer.buffer_map[consumer.params[0]]
    produced = producer.buffer_map[producer.params[2]]
    output = consumer.buffer_map[consumer.params[1]]
    if not store.buffer.same_as(produced):
        raise ValueError("P0 producer must write only its declared output")
    if any(not tilelang.tvm.ir.structural_equal(produced.shape, buffer.shape)
           for buffer in (source, output)) or len(produced.shape)!=1:
        raise ValueError("P0 composition requires equal one-dimensional shapes")
    if len({str(buffer.dtype) for buffer in (produced, source, output)})!=1:
        raise ValueError("P0 composition requires equal dtypes")
    consumer_stores=[]
    tir.stmt_functor.post_order_visit(consumer.body,
        lambda node: consumer_stores.append(node) if isinstance(node,tir.BufferStore) else None)
    if len(consumer_stores)!=1 or not consumer_stores[0].buffer.same_as(output):
        raise ValueError("P0 consumer must write only its declared output")
    inlined = [0]
    def replace_load(node):
        if not isinstance(node,tir.BufferLoad) or not node.buffer.same_as(source):
            return None
        def at_consumer_index(load):
            if isinstance(load,tir.BufferLoad):
                if not tilelang.tvm.ir.structural_equal(load.indices[0],store.indices[0]):
                    raise ValueError("P0 producer must use identity-indexed loads")
                return tir.BufferLoad(load.buffer,node.indices)
            return None
        inlined[0]+=1
        rewritten = tir.stmt_functor.ir_transform(tir.Evaluate(store.value),None,at_consumer_index,["tirx.BufferLoad"])
        return rewritten.value
    body=tir.stmt_functor.ir_transform(consumer.body,None,replace_load,["tirx.BufferLoad"])
    if inlined[0]!=1:
        raise ValueError("P0 consumer must load the producer output once")
    params=[producer.params[0],producer.params[1],consumer.params[1]]
    buffers={params[0]:producer.buffer_map[params[0]],params[1]:producer.buffer_map[params[1]],
             params[2]:consumer.buffer_map[params[2]]}
    return tir.PrimFunc(params,body,buffer_map=buffers,attrs=consumer.attrs).with_attr("global_symbol","composed")


def probe(root):
    import tilelang
    import torch
    from experiments.p0.validation_worker import runtime, check_elementwise
    from experiments.p0.validation_kernels import composition_stage
    from experiments.p0.upstream_probe import materialize
    _,_,device=runtime()
    try:
        compose(composition_stage(129),composition_stage(128,activation=True))
    except ValueError as error:
        shape_rejection=str(error)
    else:
        raise AssertionError("mismatched producer and consumer shapes were accepted")
    rows=[]
    for size in (1,127,128,129,1025):
        producer=composition_stage(size)
        producer_json=tilelang.tvm.ir.save_json(producer)
        (root/f"producer-{size}.json").write_text(producer_json)
        for tile in (64,128):
            consumer_json=tilelang.tvm.ir.save_json(composition_stage(size,tile,activation=True))
            (root/f"consumer-{size}-{tile}.json").write_text(consumer_json)
            combined=compose(tilelang.tvm.ir.load_json(producer_json),tilelang.tvm.ir.load_json(consumer_json))
            saved=tilelang.tvm.ir.save_json(combined)
            (root/f"composed-{size}-{tile}.json").write_text(saved)
            kernel=tilelang.compile(tilelang.tvm.ir.load_json(saved),out_idx=None,
                target={"kind":"cuda","arch":device["arch"]},execution_backend="tvm_ffi")
            materialize(kernel)
            source=kernel.get_kernel_source()
            import re
            assert len(re.findall(r'extern "C" __global__ void .*?\{',source))==1,source
            error=check_elementwise(kernel,torch,size)
            rows.append({"size":size,"consumer_tile":tile,"kernel_definitions":1,
                         "max_abs_error":error,"ir_bytes":len(saved)})
    return {"rows":rows,"representation":"serialized frontend TIRx; no new IR",
            "negative_shape_case":shape_rejection,
            "scope":"one identity-indexed pointwise producer/consumer; not general fusion",
            "schedule_variants":"consumer tile/thread count 64 and 128; producer artifact reused",
            "post_lower_tile_op":"earlier re-lowering rejection remains; frontend capture required"}
