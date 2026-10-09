"""Pack active expert row tiles into a bounded, graph-compatible launch list."""


def tile_map_kernel(p):
    import tilelang.language as T
    from tensor.compiler.entry import primitive
    rows, m = p['rows'], p['block_m']
    experts, top = 256, 8
    # sum ceil(count/m) <= ceil(sum count/m) + experts - 1.
    max_tiles = T.ceildiv(rows*top, m)+experts-1
    @T.macro
    def algorithm(counts, tile_experts, tile_offsets):
        with T.Kernel(1, threads=256):
            prefix = T.alloc_shared((experts+1,), 'int32')
            tx = T.get_thread_binding()
            if tx == 0:
                prefix[0] = 0
                for expert in T.serial(experts):
                    prefix[expert+1] = prefix[expert]+T.ceildiv(counts[expert], m)
            for i in T.Parallel(max_tiles):
                tile_experts[i] = -1
                tile_offsets[i] = 0
            T.sync_threads()
            for expert in T.Parallel(experts):
                for tile in T.serial(T.ceildiv(rows, m)):
                    if tile < T.ceildiv(counts[expert], m):
                        index = prefix[expert]+tile
                        tile_experts[index] = expert
                        tile_offsets[index] = tile
    return primitive([('counts', (experts,), 'int32'),
                      ('tile_experts', (max_tiles,), 'int32'),
                      ('tile_offsets', (max_tiles,), 'int32')], algorithm)
