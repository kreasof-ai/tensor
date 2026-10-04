// Independent model execution and timing through a pinned llama.cpp CUDA build.
#include "llama.h"
#include "ggml-backend.h"
#include "nlohmann/json.hpp"
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#include <stdexcept>
using json=nlohmann::ordered_json;
using Clock=std::chrono::steady_clock;

static void log_info(ggml_log_level level,const char *text,void *) {
    if(level!=GGML_LOG_LEVEL_DEBUG)std::fputs(text,stderr);
}

struct Trace {
    std::filesystem::path directory;
    bool enabled=false;
    int index=0;
};
static bool trace(ggml_tensor *tensor,bool ask,void *opaque) {
    auto *state=static_cast<Trace *>(opaque);
    if (!state->enabled)return false;
    std::string name=tensor->name;
    bool selected=name.starts_with("l_out") || name.starts_with("result_norm") ||
                  name.find(".conv.conv")!=std::string::npos || name.find(".self_attn.q_layernorm")!=std::string::npos;
    if(ask)return selected;
    if(selected && tensor->type==GGML_TYPE_F32) {
        std::vector<float> data(ggml_nelements(tensor));
        ggml_backend_tensor_get(tensor,data.data(),0,data.size()*sizeof(float));
        std::ofstream out(state->directory/(std::to_string(state->index)+"-"+name+".bin"),std::ios::binary);
        out.write(reinterpret_cast<char *>(data.data()),data.size()*sizeof(float));
    }
    return true;
}

int main(int argc,char **argv) {
    try {
        if(argc!=2)throw std::runtime_error("usage: lfm2-reference specification.json");
        std::ifstream input(argv[1]);json spec;input>>spec;
        std::filesystem::path outdir=spec.at("out").get<std::string>();std::filesystem::create_directories(outdir);
        llama_log_set(log_info,nullptr);ggml_log_set(log_info,nullptr);
        llama_backend_init();
        auto mp=llama_model_default_params();mp.n_gpu_layers=-1;
        auto *model=llama_model_load_from_file(spec.at("model").get<std::string>().c_str(),mp);
        if(!model)throw std::runtime_error("failed to load reference model");
        const auto *vocab=llama_model_get_vocab(model);int nv=llama_vocab_n_tokens(vocab);
        Trace traced{outdir};
        auto cp=llama_context_default_params();cp.n_ctx=spec.value("context",8448);cp.n_batch=128;cp.n_ubatch=128;cp.n_seq_max=1;
        cp.n_threads=4;cp.n_threads_batch=4;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.type_k=GGML_TYPE_F16;cp.type_v=GGML_TYPE_F16;cp.offload_kqv=true;
        if(!spec.value("validation",json::array()).empty()) {
            cp.cb_eval=trace;cp.cb_eval_user_data=&traced;
        }
        auto *ctx=llama_init_from_model(model,cp);if(!ctx)throw std::runtime_error("failed to create reference context");
        auto batch=llama_batch_init(128,0,1);int position=0;
        auto reset=[&](){llama_memory_clear(llama_get_memory(ctx),true);position=0;};
        auto execute=[&](const std::vector<int> &tokens){
            for(size_t offset=0;offset<tokens.size();offset+=128) {
                int n=std::min<size_t>(128,tokens.size()-offset);batch.n_tokens=n;
                for(int i=0;i<n;i++) {
                    batch.token[i]=tokens[offset+i];batch.pos[i]=position+i;batch.n_seq_id[i]=1;batch.seq_id[i][0]=0;batch.logits[i]=(i==n-1);
                }
                if(llama_decode(ctx,batch)!=0)throw std::runtime_error("reference decode failed");
                position+=n;
            }
            llama_synchronize(ctx);
        };
        json report={{"backend","llama.cpp CUDA"},{"system",llama_print_system_info()},{"vocab",nv},{"context",llama_n_ctx(ctx)},
                     {"flash_attention",true},{"gpu_layers",-1},{"prefill_batch",128},{"cache_type","f16"}};
        report["tokenization"]=json::array();
        for(const auto &item:spec.value("tokenization",json::array())) {
            std::string text=item.at("text");bool bos=item.value("bos",true),special=item.value("special",false);
            int count=llama_tokenize(vocab,text.data(),text.size(),nullptr,0,bos,special);
            std::vector<llama_token> ids(-count);
            count=llama_tokenize(vocab,text.data(),text.size(),ids.data(),ids.size(),bos,special);
            if(count<0)throw std::runtime_error("tokenization failed");ids.resize(count);
            report["tokenization"].push_back({{"text",text},{"bos",bos},{"special",special},{"tokens",ids}});
        }
        report["validation"]=json::array();int index=0;
        for(const auto &item:spec.value("validation",json::array())) {
            if(item.value("reset",false))reset();traced.enabled=item.value("trace",false);traced.index=index;
            execute(item.at("tokens").get<std::vector<int>>());traced.enabled=false;
            const float *logits=llama_get_logits_ith(ctx,-1);if(!logits)throw std::runtime_error("missing logits");
            auto filename=std::to_string(index)+"-logits.bin";
            std::ofstream file(outdir/filename,std::ios::binary);file.write(reinterpret_cast<const char *>(logits),nv*sizeof(float));
            report["validation"].push_back({{"file",filename},{"position",position},{"tokens",item.at("tokens")}});index++;
        }
        report["benchmarks"]=json::array();
        for(const auto &item:spec.value("benchmarks",json::array())) {
            auto prompt=item.at("prompt").get<std::vector<int>>();auto decode=item.at("decode").get<std::vector<int>>();
            json samples=json::array();int repeats=item.value("repeats",5),warmups=item.value("warmups",1);
            for(int repeat=-warmups;repeat<repeats;repeat++) {
                reset();auto start=Clock::now();execute(prompt);double prefill=std::chrono::duration<double>(Clock::now()-start).count();
                json latencies=json::array();start=Clock::now();
                for(int token:decode) {
                    auto step=Clock::now();execute({token});latencies.push_back(std::chrono::duration<double>(Clock::now()-step).count());
                }
                double seconds=std::chrono::duration<double>(Clock::now()-start).count();
                if(repeat>=0)samples.push_back({{"prefill_seconds",prefill},{"decode_seconds",seconds},{"decode_latencies",latencies}});
            }
            report["benchmarks"].push_back({{"name",item.at("name")},{"prompt_tokens",prompt.size()},{"decode_tokens",decode.size()},{"warmups",warmups},{"samples",samples}});
        }
        std::ofstream output(outdir/"reference.json");output<<report.dump(2)<<"\n";
        llama_batch_free(batch);llama_free(ctx);llama_model_free(model);llama_backend_free();
        return 0;
    } catch(const std::exception &error) {std::cerr<<error.what()<<"\n";return 1;}
}
