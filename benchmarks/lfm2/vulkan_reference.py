"""ctypes public-API helper for the pinned Windows llama.cpp b11310 release.

Struct layouts are from f872b591121761ac7b2af18283bd99bdc092a63a/include/llama.h.
This helper is benchmark-only; tensor-llm does not depend on llama.cpp.
"""
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import numpy as np

COMMIT='f872b591121761ac7b2af18283bd99bdc092a63a'
RELEASE_SHA256='b93a7e765c09ce5151457b71028fae2a61d4887af88d5280faa324f87162b4c1'
P=ct.c_void_p;I=ct.c_int32;U=ct.c_uint32;F=ct.c_float;B=ct.c_bool


class ModelParams(ct.Structure):
    _fields_=[(n,P) for n in ('devices','tensor_buft_overrides')]+[(n,I) for n in ('n_gpu_layers','split_mode','load_mode','lazy_mode','main_gpu')]+[(n,P) for n in ('tensor_split','progress_callback','progress_callback_user_data','kv_overrides')]+[(n,B) for n in ('vocab_only','check_tensors','use_extra_bufts','no_host','no_alloc','load_mtp')]


class ContextParams(ct.Structure):
    _fields_=([(n,U) for n in ('n_ctx','n_batch','n_ubatch','n_seq_max','n_rs_seq','n_outputs_max','n_outputs_max_per_seq')]+
              [(n,I) for n in ('n_threads','n_threads_batch','ctx_type','rope_scaling_type','pooling_type','attention_type','flash_attn_type')]+
              [(n,F) for n in ('rope_freq_base','rope_freq_scale','yarn_ext_factor','yarn_attn_factor','yarn_beta_fast','yarn_beta_slow')]+
              [('yarn_orig_ctx',U),('defrag_thold',F),('cb_eval',P),('cb_eval_user_data',P),('type_k',I),('type_v',I),('abort_callback',P),('abort_callback_data',P)]+
              [(n,B) for n in ('embeddings','offload_kqv','no_perf','op_offload','swa_full','kv_unified')]+
              [('samplers',P),('n_samplers',ct.c_size_t),('ctx_other',P)])


class Batch(ct.Structure):
    _fields_=[('n_tokens',I),('token',ct.POINTER(I)),('embd',ct.POINTER(F)),('pos',ct.POINTER(I)),('n_seq_id',ct.POINTER(I)),('seq_id',ct.POINTER(ct.POINTER(I))),('logits',ct.POINTER(ct.c_int8))]


class Reference:
    def __init__(self,model,directory,*,context=512):
        if os.name!='nt':raise RuntimeError('this pinned helper requires Windows')
        directory=Path(directory).resolve()
        release=json.loads((directory/'release.json').read_text())
        if release['commit']!=COMMIT or release['archive_sha256']!=RELEASE_SHA256:
            raise ValueError('requires the pinned b11310 release; run fetch_vulkan.py')
        for name,digest in release['dlls'].items():
            path=(directory/name).resolve()
            if not path.is_relative_to(directory) or hashlib.file_digest(path.open('rb'),'sha256').hexdigest()!=digest:
                raise ValueError('native reference DLL checksum mismatch')
        self.dll_directory=os.add_dll_directory(str(directory));self.logs=[]
        self.ggml=ct.CDLL(str(directory/'ggml.dll'));self.lib=ct.CDLL(str(directory/'llama.dll'))
        def bind(name,restype,args):
            f=getattr(self.lib,name);f.restype=restype;f.argtypes=args;return f
        self.log_callback=ct.CFUNCTYPE(None,I,ct.c_char_p,P)(lambda level,text,data:self.logs.append(text.decode('utf-8',errors='replace')) if level!=1 else None)
        bind('llama_log_set',None,[P,P])(ct.cast(self.log_callback,P),None)
        self.ggml.ggml_backend_load_all_from_path.argtypes=[ct.c_char_p]
        self.ggml.ggml_backend_load_all_from_path(str(directory).encode())
        bind('llama_backend_init',None,[])()
        mp=bind('llama_model_default_params',ModelParams,[])();mp.n_gpu_layers=-1;mp.split_mode=0
        self.model=bind('llama_model_load_from_file',P,[ct.c_char_p,ModelParams])(str(Path(model).resolve()).encode(),mp)
        if not self.model:raise RuntimeError('llama.cpp model load failed: '+''.join(self.logs[-20:]))
        vocab=bind('llama_model_get_vocab',P,[P])(self.model)
        self.vocab=bind('llama_vocab_n_tokens',I,[P])(vocab)
        cp=bind('llama_context_default_params',ContextParams,[])()
        cp.n_ctx=context;cp.n_batch=32;cp.n_ubatch=32;cp.n_seq_max=1;cp.n_threads=6;cp.n_threads_batch=6
        cp.type_k=1;cp.type_v=1;cp.flash_attn_type=1;cp.offload_kqv=True;cp.op_offload=True
        self.ctx=bind('llama_init_from_model',P,[P,ContextParams])(self.model,cp)
        if not self.ctx:raise RuntimeError('llama.cpp context init failed: '+''.join(self.logs[-20:]))
        if not any('using device Vulkan0' in line for line in self.logs):
            raise RuntimeError('native reference did not select Vulkan0')
        self.batch=bind('llama_batch_init',Batch,[I,I,I])(32,0,1)
        bind('llama_decode',I,[P,Batch]);bind('llama_synchronize',None,[P])
        bind('llama_get_logits_ith',ct.POINTER(F),[P,I])
        bind('llama_get_memory',P,[P]);bind('llama_memory_clear',None,[P,B])
        bind('llama_batch_free',None,[Batch]);bind('llama_free',None,[P]);bind('llama_model_free',None,[P])
        self.position=0

    def reset(self):
        self.lib.llama_memory_clear(self.lib.llama_get_memory(self.ctx),True);self.position=0

    def forward(self,tokens):
        for start in range(0,len(tokens),32):
            ids=tokens[start:start+32];b=self.batch;b.n_tokens=len(ids)
            for i,token in enumerate(ids):
                b.token[i]=int(token);b.pos[i]=self.position+i;b.n_seq_id[i]=1;b.seq_id[i][0]=0;b.logits[i]=int(i==len(ids)-1)
            if self.lib.llama_decode(self.ctx,b):raise RuntimeError('llama.cpp decode failed')
            self.position+=len(ids)
        self.lib.llama_synchronize(self.ctx)
        ptr=self.lib.llama_get_logits_ith(self.ctx,-1)
        if not ptr:raise RuntimeError('llama.cpp returned no logits')
        return np.ctypeslib.as_array(ptr,shape=(self.vocab,)).copy()

    def close(self):
        self.lib.llama_batch_free(self.batch);self.lib.llama_free(self.ctx);self.lib.llama_model_free(self.model);self.dll_directory.close()
