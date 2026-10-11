#include <torch/types.h>
#include <torch/nn/module.h>
#include <torch/optim/adam.h>
#include <torch/serialize.h>
#include <torch/nn/functional/loss.h>
#include <ATen/Parallel.h>
#include <ATen/CPUGeneratorImpl.h>
#include "llama.h"
#include <algorithm>
#include <chrono>
#include <csignal>
#include <ctime>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <sys/resource.h>

namespace fs = std::filesystem;
using torch::Tensor;
constexpr int WIDTH=128, HEADS=4, DIM=32, FF=256, VOCAB=512, CONTEXT=128;
constexpr double EPS=1e-6;
volatile std::sig_atomic_t stop_requested=0;
void request_stop(int) { stop_requested=1; }
void require(bool ok, const std::string& message) { if (!ok) throw std::runtime_error(message); }
Tensor matrix(int a,int b) { return torch::randn({a,b})*0.02; }
Tensor rms(Tensor x,Tensor w) { return x*torch::rsqrt(x.pow(2).mean(-1,true)+EPS)*w; }
std::vector<int64_t> encode(const std::string& s) {
    std::vector<int64_t> ids; for(unsigned char c:s) ids.push_back(3+c); return ids;
}
Tensor ids_tensor(const std::string& s) { return torch::tensor(encode(s),torch::kInt64).unsqueeze(0); }

// Native interleaved RoPE layout, matching llama.cpp's LLAMA graph directly.
// HF's split-half Q/K weight permutation must NOT be applied to these weights.
Tensor rope(Tensor x) {
    auto freq=torch::exp(torch::arange(0,DIM,2)*(-std::log(10000.0)/DIM));
    auto angles=torch::arange(x.size(1)).unsqueeze(1)*freq.unsqueeze(0);
    auto co=angles.cos().view({1,x.size(1),1,DIM/2});
    auto si=angles.sin().view({1,x.size(1),1,DIM/2});
    auto pairs=x.reshape({x.size(0),x.size(1),HEADS,DIM/2,2});
    auto a=pairs.select(-1,0), b=pairs.select(-1,1);
    return torch::stack({a*co-b*si,a*si+b*co},-1).flatten(-2,-1);
}
struct Block:torch::nn::Module {
    Tensor q,k,v,o,gate,up,down,n1,n2;
    Block(bool identity=false) {
        q=register_parameter("q",matrix(WIDTH,WIDTH)); k=register_parameter("k",matrix(WIDTH,WIDTH));
        v=register_parameter("v",matrix(WIDTH,WIDTH)); o=register_parameter("o",matrix(WIDTH,WIDTH));
        gate=register_parameter("gate",matrix(FF,WIDTH)); up=register_parameter("up",matrix(FF,WIDTH));
        down=register_parameter("down",matrix(WIDTH,FF));
        n1=register_parameter("n1",torch::ones({WIDTH})); n2=register_parameter("n2",torch::ones({WIDTH}));
        if(identity) { torch::NoGradGuard ng; o.zero_(); down.zero_(); }
    }
    Tensor forward(Tensor x) {
        auto y=rms(x,n1); auto shape=std::vector<int64_t>{x.size(0),x.size(1),HEADS,DIM};
        auto queries=rope(torch::matmul(y,q.t()).view(shape)).transpose(1,2);
        auto keys=rope(torch::matmul(y,k.t()).view(shape)).transpose(1,2);
        auto values=torch::matmul(y,v.t()).view(shape).transpose(1,2);
        auto scores=torch::matmul(queries,keys.transpose(-2,-1))/std::sqrt(double(DIM));
        auto mask=torch::ones({x.size(1),x.size(1)},torch::kBool).triu(1);
        scores=scores.masked_fill(mask,-INFINITY);
        auto context=torch::matmul(scores.softmax(-1),values).transpose(1,2).contiguous().view({x.size(0),x.size(1),WIDTH});
        x=x+torch::matmul(context,o.t()); y=rms(x,n2);
        auto g=torch::matmul(y,gate.t());
        return x+torch::matmul((g*g.sigmoid())*torch::matmul(y,up.t()),down.t());
    }
};
struct Model:torch::nn::Module {
    Tensor embedding,output,norm;
    std::vector<std::shared_ptr<Block>> blocks;
    explicit Model(int layers=4) {
        embedding=register_parameter("embedding",matrix(VOCAB,WIDTH));
        output=register_parameter("output",matrix(VOCAB,WIDTH));
        norm=register_parameter("norm",torch::ones({WIDTH}));
        for(int i=0;i<layers;++i) append(false);
    }
    void append(bool identity) {
        require(blocks.size()<5,"Only four/five-layer experiment supported");
        auto b=std::make_shared<Block>(identity); register_module("block_"+std::to_string(blocks.size()),b); blocks.push_back(b);
    }
    Tensor forward(Tensor ids) {
        require(ids.dim()==2 && ids.size(1)>0 && ids.size(1)<=CONTEXT,"Invalid sequence length");
        auto x=embedding.index_select(0,ids.flatten()).view({ids.size(0),ids.size(1),WIDTH});
        for(auto& b:blocks) x=b->forward(x);
        return torch::matmul(rms(x,norm),output.t());
    }
    int64_t count() const { int64_t n=0;for(auto&p:parameters())n+=p.numel();return n; }
    std::vector<Tensor> base_parameters() {
        std::vector<Tensor> p={embedding,output,norm};
        for(int i=0;i<4;++i)for(auto& t:blocks.at(i)->parameters())p.push_back(t);
        return p;
    }
};
struct Example { std::string prompt,target; int split; };
std::vector<Example> dataset() {
    std::vector<Example> result;
    // Every observable combination occurs once: no identical transitions across splits.
    // b=1 means the selected destination is obstructed. Boundaries also block motion.
    for(int x=0;x<4;++x)for(int y=0;y<4;++y)for(int a=0;a<5;++a)for(int b=0;b<2;++b) {
        int nx=x,ny=y; if(!b) { nx=std::clamp(x+(a==0)-(a==1),0,3);ny=std::clamp(y+(a==2)-(a==3),0,3); }
        std::string p="x="+std::to_string(x)+",y="+std::to_string(y)+",a="+std::string(1,"RLUDW"[a])+",b="+std::to_string(b)+">";
        result.push_back({p,std::to_string(nx)+","+std::to_string(ny)+";",0});
    }
    std::mt19937 split_rng(20261011);std::shuffle(result.begin(),result.end(),split_rng);
    for(size_t i=0;i<result.size();++i) result[i].split=i<112?0:(i<136?1:2);
    return result;
}
std::vector<Example> subset(const std::vector<Example>& all,int split) {
    std::vector<Example> r;for(auto&e:all)if(e.split==split)r.push_back(e);return r;
}
std::pair<Tensor,Tensor> batch(const std::vector<Example>& examples) {
    std::vector<Tensor> inputs,labels;
    for(auto&e:examples) {
        auto tokens=encode(e.prompt+e.target);std::vector<int64_t> x(tokens.begin(),tokens.end()-1),y(tokens.begin()+1,tokens.end());
        std::fill(y.begin(),y.begin()+e.prompt.size()-1,-100);
        inputs.push_back(torch::tensor(x,torch::kInt64));labels.push_back(torch::tensor(y,torch::kInt64));
    }
    return {torch::stack(inputs),torch::stack(labels)};
}
Tensor loss(Model& model,const std::vector<Example>& examples) {
    auto [x,y]=batch(examples);
    return torch::nn::functional::cross_entropy(model.forward(x).reshape({-1,VOCAB}),y.flatten(),torch::nn::functional::CrossEntropyFuncOptions().ignore_index(-100));
}
std::string generate(Model& m,const std::string& prompt) {
    torch::NoGradGuard ng;auto ids=encode(prompt);std::string out;
    for(int i=0;i<4;++i) {
        auto t=torch::tensor(ids,torch::kInt64).unsqueeze(0);
        int token=m.forward(t).select(1,t.size(1)-1).argmax(-1).item<int>();
        ids.push_back(token);out+= token>=3 && token<259 ? char(token-3) : '?';
    }return out;
}
struct Metrics { double loss=0,accuracy=0; };
Metrics evaluate(Model& m,const std::vector<Example>& examples) {
    torch::NoGradGuard ng;Metrics r;r.loss=loss(m,examples).item<double>();
    for(auto&e:examples)r.accuracy+=generate(m,e.prompt)==e.target;
    r.accuracy/=examples.size();return r;
}
std::string json(Metrics m) { std::ostringstream o;o<<std::setprecision(9)<<"{\"loss\":"<<m.loss<<",\"exact_accuracy\":"<<m.accuracy<<"}";return o.str(); }
struct Learner {
    std::shared_ptr<Model> model;
    std::unique_ptr<torch::optim::Adam> optimizer;
    std::mt19937 rng;
    int step=0,seed;
    explicit Learner(int s,int layers=4):rng(s),seed(s) {
        model=std::make_shared<Model>(layers);
        optimizer=std::make_unique<torch::optim::Adam>(model->base_parameters(),torch::optim::AdamOptions(0.001));
        if(layers==5)optimizer->add_param_group(torch::optim::OptimizerParamGroup(model->blocks.back()->parameters()));
    }
    void grow() { model->append(true);optimizer->add_param_group(torch::optim::OptimizerParamGroup(model->blocks.back()->parameters())); }
    void train(const std::vector<Example>& data,int steps,std::ostream* trace=nullptr,const std::string& phase="",const std::vector<Example>* replay=nullptr) {
        auto start=std::chrono::steady_clock::now();
        std::uniform_int_distribution<size_t> pick(0,data.size()-1);
        for(int i=0;i<steps && !stop_requested;++i) {
            std::vector<Example> selected;
            for(int j=0;j<16;++j) {
                if(replay && j<4) {
                    std::uniform_int_distribution<size_t> old_pick(0,replay->size()-1);
                    selected.push_back((*replay)[old_pick(rng)]);
                } else selected.push_back(data[pick(rng)]);
            }
            optimizer->zero_grad();auto l=loss(*model,selected);require(torch::isfinite(l).item<bool>(),"Non-finite loss");
            l.backward();optimizer->step();++step;
            if(i%50==0 || i+1==steps) {
                struct rusage usage{};getrusage(RUSAGE_SELF,&usage);
                require(usage.ru_maxrss<2L*1024*1024,"Training exceeded 2 GiB RSS ceiling");
                require(std::chrono::steady_clock::now()-start<std::chrono::minutes(12),"Training phase exceeded 12 minute limit");
                double value=l.item<double>();std::cout<<phase<<" step="<<step<<" loss="<<value<<std::endl;
                if(trace)*trace<<"{\"phase\":\""<<phase<<"\",\"step\":"<<step<<",\"layers\":"<<model->blocks.size()<<",\"loss\":"<<value<<"}\n";
            }
        }
    }
    void write(torch::serialize::OutputArchive& archive) {
        torch::serialize::OutputArchive weights,opt;
        model->save(weights);optimizer->save(opt);archive.write("model",weights);archive.write("optimizer",opt);
        archive.write("metadata",torch::tensor({int64_t(1),int64_t(model->blocks.size()),int64_t(step),int64_t(seed)},torch::kInt64));
        std::ostringstream text;text<<rng;auto s=text.str();std::vector<uint8_t> bytes(s.begin(),s.end());archive.write("rng",torch::tensor(bytes,torch::kUInt8));
        archive.write("torch_rng",at::detail::getDefaultCPUGenerator().get_state());
    }
    void save(const fs::path& path) {
        torch::serialize::OutputArchive archive;write(archive);
        auto tmp=path.string()+".tmp";archive.save_to(tmp);fs::rename(tmp,path);
    }
    static std::unique_ptr<Learner> read(torch::serialize::InputArchive& archive) {
        torch::serialize::InputArchive weights,opt;Tensor metadata,rngbytes,torch_rng;
        archive.read("metadata",metadata);require(metadata[0].item<int>()==1,"Unsupported checkpoint schema");
        int layers=metadata[1].item<int>();require(layers==4 || layers==5,"Invalid layer count");
        auto l=std::make_unique<Learner>(metadata[3].item<int>(),layers);l->step=metadata[2].item<int>();
        archive.read("model",weights);l->model->load(weights);archive.read("optimizer",opt);l->optimizer->load(opt);
        archive.read("rng",rngbytes);std::string s(reinterpret_cast<char*>(rngbytes.data_ptr<uint8_t>()),rngbytes.numel());std::istringstream input(s);input>>l->rng;
        require(!input.fail(),"Invalid RNG state");archive.read("torch_rng",torch_rng);auto generator=at::detail::getDefaultCPUGenerator();generator.set_state(torch_rng);
        return l;
    }
    static std::unique_ptr<Learner> load(const fs::path& path) {
        torch::serialize::InputArchive archive;archive.load_from(path.string());return read(archive);
    }
};

void export_gguf(Model& m,const fs::path& path,bool half) {
    torch::NoGradGuard ng;
    auto g=std::unique_ptr<gguf_context,decltype(&gguf_free)>(gguf_init_empty(),gguf_free);
    auto c=std::unique_ptr<ggml_context,decltype(&ggml_free)>(ggml_init({1024*1024,nullptr,true}),ggml_free);
    require(bool(c)&&bool(g),"GGUF allocation failed");
    gguf_set_val_str(g.get(),"general.architecture","llama");gguf_set_val_str(g.get(),"general.name","AI Pilot adaptive Llama toy world");
    gguf_set_val_u32(g.get(),"general.file_type",half?1:0);
    gguf_set_val_u32(g.get(),"llama.context_length",CONTEXT);gguf_set_val_u32(g.get(),"llama.embedding_length",WIDTH);
    gguf_set_val_u32(g.get(),"llama.block_count",m.blocks.size());gguf_set_val_u32(g.get(),"llama.feed_forward_length",FF);
    gguf_set_val_u32(g.get(),"llama.attention.head_count",HEADS);gguf_set_val_u32(g.get(),"llama.attention.head_count_kv",HEADS);
    gguf_set_val_u32(g.get(),"llama.rope.dimension_count",DIM);gguf_set_val_f32(g.get(),"llama.rope.freq_base",10000);
    gguf_set_val_f32(g.get(),"llama.attention.layer_norm_rms_epsilon",EPS);
    std::vector<std::string> words={"<unk>","<s>","</s>"};
    // GPT-2 byte alphabet, no merges: a reversible, fixed UTF-8 byte tokenizer.
    // No learned tokenizer or SentencePiece dependency is needed for this toy task.
    int extra=256;
    for(int i=0;i<256;++i) {
        int cp=((i>=33&&i<=126)||(i>=161&&i<=172)||(i>=174))?i:extra++;
        std::string piece;
        if(cp<128)piece+=char(cp);else {piece+=char(0xc0|(cp>>6));piece+=char(0x80|(cp&63));}
        words.push_back(piece);
    }
    while(words.size()<VOCAB)words.push_back("<unused_"+std::to_string(words.size())+">");
    std::vector<const char*> tokens;std::vector<float> scores(VOCAB,-1);std::vector<int32_t> types(VOCAB,5);
    types[0]=2;types[1]=types[2]=3;for(int i=3;i<259;++i)types[i]=1;
    for(auto&s:words)tokens.push_back(s.c_str());
    gguf_set_val_str(g.get(),"tokenizer.ggml.model","gpt2");
    gguf_set_val_str(g.get(),"tokenizer.ggml.pre","default");
    gguf_set_arr_str(g.get(),"tokenizer.ggml.merges",nullptr,0);
    gguf_set_arr_str(g.get(),"tokenizer.ggml.tokens",tokens.data(),tokens.size());
    gguf_set_arr_data(g.get(),"tokenizer.ggml.scores",GGUF_TYPE_FLOAT32,scores.data(),scores.size());
    gguf_set_arr_data(g.get(),"tokenizer.ggml.token_type",GGUF_TYPE_INT32,types.data(),types.size());
    gguf_set_val_u32(g.get(),"tokenizer.ggml.unknown_token_id",0);gguf_set_val_u32(g.get(),"tokenizer.ggml.bos_token_id",1);gguf_set_val_u32(g.get(),"tokenizer.ggml.eos_token_id",2);
    gguf_set_val_bool(g.get(),"tokenizer.ggml.add_bos_token",false);gguf_set_val_bool(g.get(),"tokenizer.ggml.add_eos_token",false);
    gguf_set_val_bool(g.get(),"tokenizer.ggml.add_space_prefix",false);gguf_set_val_bool(g.get(),"tokenizer.ggml.remove_extra_whitespaces",false);
    gguf_set_val_bool(g.get(),"tokenizer.ggml.escape_whitespaces",false);
    std::vector<Tensor> keep;
    auto add=[&](const std::string& name,Tensor t) {
        t=t.detach().to(half && t.dim()==2?torch::kFloat16:torch::kFloat32).contiguous();keep.push_back(t);
        ggml_tensor* out=t.dim()==2?ggml_new_tensor_2d(c.get(),half?GGML_TYPE_F16:GGML_TYPE_F32,t.size(1),t.size(0)):ggml_new_tensor_1d(c.get(),GGML_TYPE_F32,t.size(0));
        ggml_set_name(out,name.c_str());out->data=t.data_ptr();gguf_add_tensor(g.get(),out);
    };
    add("token_embd.weight",m.embedding);add("output.weight",m.output);add("output_norm.weight",m.norm);
    for(size_t i=0;i<m.blocks.size();++i) {
        auto& b=*m.blocks[i];auto p="blk."+std::to_string(i)+".";
        add(p+"attn_q.weight",b.q);add(p+"attn_k.weight",b.k);add(p+"attn_v.weight",b.v);add(p+"attn_output.weight",b.o);
        add(p+"attn_norm.weight",b.n1);add(p+"ffn_norm.weight",b.n2);add(p+"ffn_gate.weight",b.gate);add(p+"ffn_up.weight",b.up);add(p+"ffn_down.weight",b.down);
    }
    auto tmp=path.string()+".tmp";require(gguf_write_to_file(g.get(),tmp.c_str(),false),"GGUF write failed");fs::rename(tmp,path);
}
void quiet_log(enum ggml_log_level level,const char* text,void*) { if(level==GGML_LOG_LEVEL_ERROR)std::cerr<<text; }
struct Runtime {
    std::unique_ptr<llama_model,decltype(&llama_model_free)> model{nullptr,llama_model_free};
    std::unique_ptr<llama_context,decltype(&llama_free)> ctx{nullptr,llama_free};
    explicit Runtime(const fs::path& path) {
        llama_log_set(quiet_log,nullptr);
        static const bool loaded=[](){ggml_backend_load_all_from_path(LLAMA_BACKEND_DIR);return true;}();
        (void)loaded;llama_backend_init();auto mp=llama_model_default_params();mp.n_gpu_layers=0;
        model.reset(llama_model_load_from_file(path.c_str(),mp));require(bool(model),"llama.cpp failed to load GGUF");
        auto cp=llama_context_default_params();cp.n_ctx=CONTEXT;cp.n_batch=CONTEXT;cp.n_ubatch=CONTEXT;cp.n_threads=2;cp.n_threads_batch=2;
        cp.type_k=GGML_TYPE_F32;cp.type_v=GGML_TYPE_F32;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;cp.no_perf=true;
        ctx.reset(llama_init_from_model(model.get(),cp));require(bool(ctx),"llama.cpp context failed");
    }
    Tensor logits(const std::vector<int64_t>& ids) {
        std::vector<llama_token> tokens(ids.begin(),ids.end());llama_memory_clear(llama_get_memory(ctx.get()),true);
        auto b=llama_batch_get_one(tokens.data(),tokens.size());require(llama_decode(ctx.get(),b)==0,"llama_decode failed");
        auto ptr=llama_get_logits_ith(ctx.get(),-1);require(ptr!=nullptr,"Missing llama logits");return torch::from_blob(ptr,{VOCAB},torch::kFloat32).clone();
    }
    void tokenizer_check(const std::string& prompt) {
        std::vector<llama_token> actual(prompt.size()*4+32);auto vocab=llama_model_get_vocab(model.get());
        auto n=llama_tokenize(vocab,prompt.data(),prompt.size(),actual.data(),actual.size(),false,false);
        auto expected=encode(prompt);require(n==int(expected.size()),"Tokenizer length mismatch");
        std::string decoded;
        for(int i=0;i<n;++i) {
            require(actual[i]==expected[i],"Tokenizer token mismatch at "+std::to_string(i));
            char piece[16];int size=llama_token_to_piece(vocab,actual[i],piece,16,0,false);require(size>=0,"Decode buffer too small");decoded.append(piece,size);
        }require(decoded==prompt,"Tokenizer round-trip mismatch");
    }
    std::string generate(const std::string& prompt) {
        auto ids=encode(prompt);std::string out;
        for(int i=0;i<4;++i){int id=logits(ids).argmax().item<int>();ids.push_back(id);out+=id>=3&&id<259?char(id-3):'?';}return out;
    }
};
double parity(Model& m,const fs::path& file,const std::vector<Example>& checks,double tolerance) {
    torch::NoGradGuard ng;Runtime runtime(file);double worst=0;
    for(auto&s:std::vector<std::string>{"x=1,y=2,a=R,b=0>"," spaces  \n",u8"árvíz"})runtime.tokenizer_check(s);
    for(size_t i=0;i<std::min(size_t(8),checks.size());++i) {
        auto ids=ids_tensor(checks[i].prompt);auto expected=m.forward(ids).select(1,ids.size(1)-1).squeeze(0);
        auto actual=runtime.logits(encode(checks[i].prompt));worst=std::max(worst,(actual-expected).abs().max().item<double>());
    }require(worst<tolerance,"GGUF logit error "+std::to_string(worst)+" exceeds "+std::to_string(tolerance));return worst;
}
void export_checked(Model& m,const fs::path& dir,const std::string& name,const std::vector<Example>& examples,std::ostream& report) {
    auto f32=dir/(name+"-f32.gguf"), f16=dir/(name+"-f16.gguf");export_gguf(m,f32,false);export_gguf(m,f16,true);
    double a=parity(m,f32,examples,0.0003),b=parity(m,f16,examples,0.02);
    report<<"{\"name\":\""<<name<<"\",\"f32_max_logit_error\":"<<a<<",\"f16_max_logit_error\":"<<b<<",\"f16_bytes\":"<<fs::file_size(f16)<<"}";
}
bool accept(Metrics candidate,Metrics control,Metrics retention,Metrics reference) {
    return candidate.loss<control.loss*0.99 && candidate.accuracy>=control.accuracy && retention.accuracy>=reference.accuracy-0.02;
}
void self_test(const fs::path& dir,int seed) {
    auto data=dataset();auto train=subset(data,0),validation=subset(data,1);auto prompt=ids_tensor(train[0].prompt);
    Learner learner(seed);require(learner.model->count()==787584,"Four-layer parameter count");
    {
        torch::NoGradGuard ng;
        auto prefix=ids_tensor("x=1,y=2");auto extended=ids_tensor("x=1,y=2,a=R,b=0>");
        require(torch::allclose(learner.model->forward(prefix),learner.model->forward(extended).slice(1,0,prefix.size(1)),1e-5,1e-6),"Future tokens leaked through causal mask");
    }
    learner.train(train,3);learner.save(dir/"resume-4.pt");auto restored=Learner::load(dir/"resume-4.pt");
    learner.train(train,2);restored->train(train,2);
    require(torch::equal(learner.model->forward(prompt),restored->model->forward(prompt)),"Four-layer optimizer/RNG resume diverged");
    Tensor before;{torch::NoGradGuard ng;before=learner.model->forward(prompt).clone();}
    learner.grow();require(learner.model->count()==951680,"Five-layer parameter count");
    require(torch::equal(before,learner.model->forward(prompt)),"Growth changed initial function");
    learner.train(train,3);require(learner.model->blocks.back()->o.abs().sum().item<double>()>0,"New block did not learn");
    learner.save(dir/"resume-5.pt");auto five=Learner::load(dir/"resume-5.pt");learner.train(train,2);five->train(train,2);
    require(torch::equal(learner.model->forward(prompt),five->model->forward(prompt)),"Five-layer optimizer/RNG resume diverged");
    require(accept({0.5,0.8},{1,0.7},{0.9,0.8},{1,0.8}),"Acceptance positive case");
    require(!accept({1.1,0.8},{1,0.7},{0.9,0.8},{1,0.8}),"Acceptance must reject worse loss");
    require(!accept({0.5,0.8},{1,0.7},{0.9,0.5},{1,0.8}),"Acceptance must reject forgetting");
    std::ofstream checks(dir/"checks.json");checks<<"{\"passed\":true,\"exports\":[";
    export_checked(*restored->model,dir,"check-4",validation,checks);checks<<",";export_checked(*five->model,dir,"check-5",validation,checks);checks<<"]}\n";
    std::cout<<"PASS: exact optimizer/RNG resume, residual-preserving growth, learned new weights, acceptance guards, tokenizer and GGUF parity (four/five layers)."<<std::endl;
}
std::string safe_string(const std::string& s) {
    std::ostringstream out;out<<'"';for(unsigned char c:s){if(c=='"'||c=='\\')out<<'\\'<<c;else if(c<32||c>126||c=='<')out<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<int(c)<<std::dec;else out<<c;}out<<'"';return out.str();
}
#include "experiment.h"
#include "continual.h"

int main(int argc,char**argv) {
    try {
        at::set_num_threads(2);at::set_num_interop_threads(1);
        std::signal(SIGINT,request_stop);std::signal(SIGTERM,request_stop);
        require(argc>=3,"Usage: run OUTPUT [SEED STEPS EXTRA UI PAUSE_AFTER], resume OUTPUT [UI PAUSE_AFTER], test OUTPUT, predict CHECKPOINT PROMPT, or compare STATE_A STATE_B");
        std::string mode=argv[1];
        if(mode=="campaign" || mode=="campaign-resume") {
            campaign(argv[2],argc>3?argv[3]:"",mode=="campaign-resume",argc>4?std::stoi(argv[4]):0);return 0;
        }
        if(mode=="predict") {
            require(argc==4,"Usage: predict CHECKPOINT PROMPT");
            auto learner=Learner::load(argv[2]);std::cout<<generate(*learner->model,argv[3])<<std::endl;return 0;
        }
        if(mode=="compare") {
            require(argc==4,"Usage: compare STATE_A STATE_B");compare_sessions(argv[2],argv[3]);return 0;
        }
        if(mode=="compare-checkpoints") {
            require(argc==4,"Usage: compare-checkpoints CHECKPOINT_A CHECKPOINT_B");
            auto a=Learner::load(argv[2]),b=Learner::load(argv[3]);compare_learners(*a,*b);
            std::cout<<"PASS: checkpoint weights, sampling state and next optimizer update match."<<std::endl;return 0;
        }
        if(mode=="resume") {
            auto state=Session::load(fs::path(argv[2])/"state.pt");
            if(state.phase==Phase::Finished && fs::exists(fs::path(argv[2])/"COMPLETE")) {
                std::cout<<"Already complete; no training or files changed."<<std::endl;return 0;
            }
            ++state.resumes;
            int pause=argc>4?std::stoi(argv[4]):0;require(pause>=0,"Invalid pause interval");
            execute(state,argv[2],argc>3?argv[3]:"",pause);return 0;
        }
        require(mode=="test"||mode=="run","Unknown command");fs::path dir=argv[2];
        int seed=argc>3?std::stoi(argv[3]):42,steps=argc>4?std::stoi(argv[4]):600,extra=argc>5?std::stoi(argv[5]):200;
        int pause=argc>7?std::stoi(argv[7]):0;
        require(seed>=0&&steps>0&&steps<=10000&&extra>0&&extra<=2000&&pause>=0,"Invalid seed/step bounds");
        require(!fs::exists(dir),"Output already exists; choose a new run directory");fs::create_directories(dir);
        torch::manual_seed(seed);
        if(mode=="test") {self_test(dir,seed);policy_tests();continual_tests(dir);}
        else {
            Session state(seed,steps,extra);auto validation=subset(dataset(),1);
            state.initial=evaluate(*state.active->model,validation);state.history.push_back({0,state.initial});
            state.save(dir/"state.pt");execute(state,dir,argc>6?argv[6]:"",pause);
        }
        return 0;
    } catch(const std::exception& e) { std::cerr<<"ERROR: "<<e.what()<<std::endl;return 1; }
}
