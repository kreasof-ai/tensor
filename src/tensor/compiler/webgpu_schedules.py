"""WebGPU schedule spaces and moves supplied to generic discovery."""
import itertools

SPACES = {
    'partitioned': {'tile_m': (4,8,16,32), 'tile_n': (4,8,16,32,64,128),
                    'threads': (64,128,256), 'partitions': (2,4,8,16,32),
                    'unroll': (1,2,4,8,16,32), 'dot_width': (1,4), 'k_layout': ('blocked','striped')},
    'staged': {'tile_m': (8,16,32), 'tile_n': (16,32,64,128),
               'tile_k': (16,32,64,128), 'threads': (64,128,256),
               'dot_width': (1,4), 'unroll': (False,True), 'lhs_pad': (0,1),
               'lhs_transpose': (False,True)},
}
SPACES['partitioned_rows'] = {**SPACES['partitioned'], 'tile_n': (4,5,8,10,16,20,32,40,64), 'dot_width': (1,2,4)}


def coupled_moves(config, space):
    if config['family'].startswith('partitioned'):
        for threads, partitions in itertools.product(space['threads'], space['partitions']):
            if threads // partitions == config['threads'] // config['partitions']:
                yield {**config, 'threads': threads, 'partitions': partitions}
    if all(axis in space for axis in ('tile_m','tile_n','micro_m','micro_n','threads')):
        for axis in ('tile_m','tile_n','micro_m','micro_n'):
            values = space[axis]; index = values.index(config[axis])
            for offset in (-1, 1):
                if not 0 <= index + offset < len(values):
                    continue
                candidate = {**config, axis: values[index + offset]}
                tm, tn, mm, mn = (candidate[name] for name in ('tile_m','tile_n','micro_m','micro_n'))
                if tm % mm or tn % mn:
                    continue
                threads = (tm // mm) * (tn // mn)
                if threads in space['threads']:
                    yield {**candidate, 'threads': threads}
