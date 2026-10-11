// Bounded continual state and active inquiry extend the existing Graph runtime.
// No task labels or simulator functions are accepted by the learner APIs below.
struct Skill { Graph graph; std::vector<Example> witnesses; Descriptor descriptor{}; };
struct Abstraction {
    Graph constants;
    Holes holes{};
    std::vector<int> supports;
    std::string json() const {
        std::ostringstream s; s << "{\"shape\":" << constants.json() << ",\"holes\":[";
        for(int k=0;k<2;++k){if(k)s<<',';s<<'['<<holes[k][0]<<','<<holes[k][1]<<','<<holes[k][2]<<']';}
        s << "],\"supports\":[";for(size_t i=0;i<supports.size();++i){if(i)s<<',';s<<supports[i];}s<<"]}";return s.str();
    }
};
bool topology(const Graph& a,const Graph& b){return a.delay==b.delay&&a.split==b.split;}
struct WindowResult {
    int route=-1; bool reused=false, inserted=false, full=false, macro_accepted=false;
    int64_t search=0,routing=0,retention=0,extraction=0,matching=0;
    Graph before,after; std::vector<std::string> events;
};
struct Continual {
    Learner learner;
    std::vector<Skill> bank;
    std::vector<Abstraction> library;
    std::vector<Example> replay;
    int windows=0;
    void extract(int64_t& work) {
        // Anti-unification is learned from two or more different accepted graphs.
        library.clear();
        for(size_t i=0;i<bank.size();++i){
            if(std::any_of(library.begin(),library.end(),[&](const auto& a){return topology(a.constants,bank[i].graph);}))continue;
            Abstraction a;a.constants=bank[i].graph;
            for(size_t j=i;j<bank.size();++j){++work;if(!topology(a.constants,bank[j].graph))continue;a.supports.push_back(int(j));
                for(int k=0;k<(a.constants.split<0?1:2);++k){auto c=a.constants.leaves[k],d=bank[j].graph.leaves[k];a.holes[k][0]=a.holes[k][0]||c.c0!=d.c0;a.holes[k][1]=a.holes[k][1]||c.c1!=d.c1;a.holes[k][2]=a.holes[k][2]||c.c2!=d.c2;}}
            if(a.supports.size()>=2)library.push_back(a);
        }
    }
    int route(const std::vector<Example>& evidence,int64_t& work) const {
        int chosen=-1;double best=std::numeric_limits<double>::infinity();
        for(size_t i=0;i<bank.size();++i){double loss=mse(bank[i].graph,evidence);work+=evidence.size();double score=loss+.001*bank[i].graph.complexity();if(loss==0&&score<best){best=score;chosen=int(i);}}
        return chosen;
    }
    WindowResult observe(const Data& input,const std::string& method,int64_t cap=20000) {
        need(input.train.size()==64&&input.val.size()==32,"Window needs 64 train/32 validation examples");
        need(method=="single"||method=="replay"||method=="bank"||method=="bank-macros","Unknown continual method");
        WindowResult out;out.before=learner.current;Data data=input;data.test.clear(); // evaluation never enters state
        const bool banking=method=="bank"||method=="bank-macros";
        if(banking){out.route=route(data.val,out.routing);if(out.route>=0){learner.current=bank[out.route].graph;out.reused=true;}}
        if(!out.reused){
            if(method=="replay"&&!replay.empty()){
                data.train.resize(32);data.val.resize(16);
                for(int i=0;i<48;++i)(i<32?data.train:data.val).push_back(replay[size_t(i)%replay.size()]);
            }
            if(banking)learner.current=Graph{};
            // adapt_v2 reports test accuracy but uses no test labels; a nonempty
            // training-only copy keeps its report field defined without leakage.
            data.test=data.train;
            if(method=="bank-macros"&&!library.empty()){
                auto descriptor=describe(Graph{},data.train);int chosen=-1;double nearest=std::numeric_limits<double>::infinity();
                for(size_t i=0;i<library.size();++i)for(int support:library[i].supports){double distance=0;for(int j=0;j<5;++j){distance+=std::pow(descriptor[j]-bank[support].descriptor[j],2);++out.matching;}if(distance<nearest){nearest=distance;chosen=int(i);}}
                const auto& a=library[size_t(chosen)];Budget budget{0,cap};Graph before=learner.current,candidate=before;bool exhausted=false;double old=mse(before,data.val),fresh=old;
                try{candidate=fit(a.constants,data.train,budget,&a.constants,&a.holes);fresh=mse(candidate,data.val,&budget);out.macro_accepted=fresh+.001*candidate.complexity()<old+.001*before.complexity()-1e-9;if(out.macro_accepted)learner.current=candidate;}catch(const Exhausted&){exhausted=true;}
                std::ostringstream event;event<<"{\"trial\":1,\"operator\":\"learned-template\",\"template\":"<<chosen<<",\"accepted\":"<<(out.macro_accepted?"true":"false")<<",\"exhausted\":"<<(exhausted?"true":"false")<<",\"before_mse\":"<<old<<",\"candidate_mse\":"<<(exhausted?"null":std::to_string(fresh))<<",\"work\":"<<budget.used<<",\"total_work\":"<<budget.used<<",\"before\":"<<before.json()<<",\"candidate\":"<<(exhausted?"null":candidate.json())<<",\"selected\":"<<learner.current.json()<<"}";out.events.push_back(event.str());out.search=budget.used;
            }
            if(mse(learner.current,data.val)>0 && out.search<cap){auto result=adapt_v2(learner,data,"fixed-012",cap-out.search);out.search+=result.work;out.events.insert(out.events.end(),result.events.begin(),result.events.end());}
            // Small effects can require two edits before either alone improves
            // validation. Search the composed topology without committing an
            // unhelpful intermediate graph; the same remaining cap still applies.
            if(mse(learner.current,data.val)>0 && out.search<cap && !(learner.current.delay&&learner.current.split>=0)){
                Graph before=learner.current,shape=before,candidate=before;shape.delay=true;
                Budget budget{0,cap-out.search};bool accept=false,exhausted=false;double old=mse(before,data.val),fresh=old;
                try{candidate=propose(shape,1,data,budget);fresh=mse(candidate,data.val,&budget);accept=fresh+.001*candidate.complexity()<old+.001*before.complexity()-1e-9;if(accept)learner.current=candidate;}catch(const Exhausted&){exhausted=true;}
                std::ostringstream e;e<<"{\"operator\":\"compose-delay-split\",\"accepted\":"<<(accept?"true":"false")<<",\"exhausted\":"<<(exhausted?"true":"false")<<",\"before_mse\":"<<old<<",\"candidate_mse\":"<<(exhausted?"null":std::to_string(fresh))<<",\"work\":"<<budget.used<<",\"before\":"<<before.json()<<",\"candidate\":"<<(exhausted?"null":candidate.json())<<",\"selected\":"<<learner.current.json()<<"}";out.events.push_back(e.str());out.search+=budget.used;
            }
            // An insertion adds a new immutable version; it never overwrites an old skill.
            if(banking && mse(learner.current,input.val)==0){
                if(bank.size()<16){bank.push_back({learner.current,input.val,describe(Graph{},input.train)});out.route=int(bank.size())-1;out.inserted=true;if(method=="bank-macros")extract(out.extraction);}
                else out.full=true;
            }
        }
        for(const auto& skill:bank){out.retention+=skill.witnesses.size();need(mse(skill.graph,skill.witnesses)==0,"Protected skill lost its evidence");}
        if(method=="replay"){replay.insert(replay.end(),input.train.begin(),input.train.end());if(replay.size()>256)replay.erase(replay.begin(),replay.end()-256);}
        learner.episodes=input.train;++windows;out.after=learner.current;return out;
    }
};
void putexamples(std::ostream& s,const std::vector<Example>& examples){s<<examples.size()<<'\n';for(auto e:examples)s<<e.x<<' '<<e.a<<' '<<e.prev<<' '<<e.b<<' '<<e.n<<' '<<e.y<<'\n';}
std::vector<Example> getexamples(std::istream& s,size_t maximum){size_t n=0;s>>n;need(n<=maximum,"Example count limit");std::vector<Example> examples;for(size_t i=0;i<n;++i){Example e{};s>>e.x>>e.a>>e.prev>>e.b>>e.n>>e.y;need(bounded(e.x,-100,100)&&bounded(e.y,-110,110)&&bounded(e.a,-1,1)&&bounded(e.prev,-1,1)&&bounded(e.b,0,1)&&bounded(e.n,0,1),"Invalid evidence");examples.push_back(e);}return examples;}
std::string continual_checkpoint(const Continual& c){
    std::ostringstream s;s<<std::setprecision(17)<<"REPAIR_CONTINUAL 1\n"<<c.windows<<'\n'<<std::quoted(checkpoint(c.learner))<<'\n'<<c.bank.size()<<'\n';
    for(const auto& b:c.bank){putgraph(s,b.graph);putexamples(s,b.witnesses);for(double x:b.descriptor)s<<x<<' ';s<<'\n';}
    // Library is deterministic anti-unification of the persisted bank. Store it
    // explicitly as evidence and verify reconstructed equality when restoring.
    std::vector<std::string> definitions;for(const auto& a:c.library)definitions.push_back(a.json());
    s<<std::quoted(array(definitions))<<'\n';putexamples(s,c.replay);return s.str();
}
Continual restore_continual(const std::string& bytes){
    need(bytes.size()<1000000,"Checkpoint too large");std::istringstream s(bytes);std::string magic,native,definitions;int version=0;size_t count=0;Continual c;
    s>>magic>>version>>c.windows>>std::quoted(native)>>count;need(magic=="REPAIR_CONTINUAL"&&version==1&&bounded(c.windows,0,100000)&&count<=16,"Invalid continual header");c.learner=restore(native);
    for(size_t i=0;i<count;++i){Skill b;b.graph=getgraph(s);b.witnesses=getexamples(s,32);need(!b.witnesses.empty(),"Missing witnesses");for(auto& x:b.descriptor){s>>x;need(std::isfinite(x),"Invalid descriptor");}need(mse(b.graph,b.witnesses)==0,"Invalid retained skill");c.bank.push_back(b);}
    s>>std::quoted(definitions);if(definitions!="[]"){int64_t work=0;c.extract(work);}std::vector<std::string> actual;for(const auto& a:c.library)actual.push_back(a.json());need(array(actual)==definitions,"Invalid abstraction support");
    c.replay=getexamples(s,256);need(bool(s),"Truncated continual state");s>>std::ws;need(s.eof(),"Trailing continual state");return c;
}

// Active learner: both anonymous sources start with the identical graph grammar.
std::vector<Graph> hypothesis_grammar(){
    std::vector<Graph> graphs;
    for(bool delay:{false,true})for(int split:{-1,0,1}){
        std::vector<Coeff> coefficients;for(int c0=-1;c0<=1;++c0)for(int c1=-2;c1<=2;++c1)for(int c2=delay?-2:0;c2<=(delay?2:0);++c2)coefficients.push_back({c0,c1,c2});
        for(auto a:coefficients)for(size_t j=0;j<(split<0?1:coefficients.size());++j){Graph g;g.delay=delay;g.split=split;g.leaves[0]=a;g.leaves[1]=split<0?Coeff{}:coefficients[j];graphs.push_back(g);}
    }
    std::stable_sort(graphs.begin(),graphs.end(),[](const auto& a,const auto& b){return a.complexity()<b.complexity();});return graphs;
}
const std::vector<Graph>& hypotheses(){static const auto graphs=hypothesis_grammar();return graphs;}
std::vector<Example> probes(){std::vector<Example> out;for(int a=-1;a<=1;++a)for(int prev=-1;prev<=1;++prev)for(int b=0;b<2;++b)for(int n=0;n<2;++n)out.push_back({0,a,prev,b,n,0});return out;}
struct Inquiry {int source=-1,probe=-1;double score=0;};
struct Explorer {
    std::array<std::vector<int>,2> candidates;
    std::array<Graph,2> prediction;
    std::array<std::vector<int>,2> errors;
    std::mt19937 rng{42};int steps=0;int64_t evaluations=0;
    Explorer(){for(auto& c:candidates){c.resize(hypotheses().size());std::iota(c.begin(),c.end(),0);}}
    Inquiry choose(const std::string& method){
        need(method=="random"||method=="error"||method=="active","Invalid inquiry method");const auto inputs=probes();
        if(method=="random")return {int(rng()%2),int(rng()%inputs.size()),0};
        if(method=="error"){
            if(steps<2)return {steps,int(rng()%inputs.size()),0};
            double best=-1;int source=0;for(int i=0;i<2;++i){double score=errors[i].empty()?0:double(std::accumulate(errors[i].begin(),errors[i].end(),0))/errors[i].size();if(score>best){best=score;source=i;}}return {source,int(rng()%inputs.size()),best};
        }
        Inquiry best;
        for(int source=0;source<2;++source){const auto& set=candidates[source];if(set.empty())continue;
            for(size_t p=0;p<inputs.size();++p){std::map<int,int> counts;for(int id:set){++counts[hypotheses()[id].predict(inputs[p])];++evaluations;}double same=0;for(auto item:counts)same+=double(item.second)*item.second;double score=1-same/(double(set.size())*set.size());if(score>best.score+1e-12)best={source,int(p),score};}
        }return best;
    }
    void observe(const Inquiry& query,const Example& evidence){
        need(bounded(query.source,0,1)&&bounded(query.probe,0,35),"Invalid inquiry");auto source=size_t(query.source);auto& history=errors[source];history.push_back(std::abs(prediction[source].predict(evidence)-evidence.y));if(history.size()>8)history.erase(history.begin());
        auto& set=candidates[source];std::vector<int> consistent;for(int id:set){++evaluations;if(hypotheses()[id].predict(evidence)==evidence.y)consistent.push_back(id);}set=std::move(consistent);if(!set.empty())prediction[source]=hypotheses()[set.front()];++steps;
    }
};
std::string explorer_checkpoint(const Explorer& e){std::ostringstream s;s<<"REPAIR_EXPLORER 1\n"<<e.steps<<' '<<e.evaluations<<'\n'<<e.rng<<'\n';for(int source=0;source<2;++source){putgraph(s,e.prediction[source]);s<<e.candidates[source].size()<<'\n';for(int id:e.candidates[source])s<<id<<' ';s<<'\n'<<e.errors[source].size()<<'\n';for(int error:e.errors[source])s<<error<<' ';s<<'\n';}return s.str();}
Explorer restore_explorer(const std::string& bytes){need(bytes.size()<1000000,"Explorer checkpoint limit");std::istringstream s(bytes);std::string magic;int version=0;Explorer e;s>>magic>>version>>e.steps>>e.evaluations>>e.rng;need(magic=="REPAIR_EXPLORER"&&version==1&&bounded(e.steps,0,100000)&&e.evaluations>=0,"Invalid explorer header");for(int source=0;source<2;++source){e.prediction[source]=getgraph(s);size_t count=0;s>>count;need(count<=hypotheses().size(),"Hypothesis count limit");auto& set=e.candidates[source];set.clear();int previous=-1;for(size_t i=0;i<count;++i){int id=-1;s>>id;need(id>previous&&id<int(hypotheses().size()),"Invalid hypothesis ID");set.push_back(id);previous=id;}s>>count;need(count<=8,"Error window limit");e.errors[source].clear();for(size_t i=0;i<count;++i){int error=0;s>>error;need(bounded(error,0,220),"Invalid error");e.errors[source].push_back(error);}}need(bool(s),"Truncated explorer state");s>>std::ws;need(s.eof(),"Trailing explorer state");return e;}

void test_v3(const fs::path& directory){
    fs::create_directories(directory);
    Continual c;auto first=world(3,-1,9900001),second=world(3,1,9900002),transfer=world(3,2,9900003);
    auto initial=c.observe(first,"bank-macros");need(initial.inserted&&accuracy(c.learner.current,first.test)==1,"First retained graph not learned");
    auto witness=c.bank.front().graph.json();c.observe(second,"bank-macros");need(c.library.size()==1&&c.library.front().supports.size()==2,"Missing learned abstraction");
    need(c.library.front().holes[0][2]&&!c.library.front().holes[1][2],"Incorrect anti-unification holes");
    auto saved=continual_checkpoint(c);atomic(directory/"continual.model",saved);auto restored=restore_continual(saved);need(continual_checkpoint(restored)==saved,"Continual checkpoint round trip");
    auto a=c.observe(transfer,"bank-macros"),b=restored.observe(transfer,"bank-macros");need(a.macro_accepted&&accuracy(c.learner.current,transfer.test)==1,"Learned macro failed transfer");need(a.events==b.events&&continual_checkpoint(c)==continual_checkpoint(restored),"Next window differs after restore");
    need(c.bank.front().graph.json()==witness,"Old graph overwritten");auto returned=c.observe(world(3,-1,9900004),"bank-macros");need(returned.reused&&returned.search==0&&accuracy(c.learner.current,first.test)==1,"Automatic return routing failed");
    auto mutated=transfer;for(auto& e:mutated.test)e.y+=90;auto x=restore_continual(saved),y=x;auto rx=x.observe(transfer,"bank-macros"),ry=y.observe(mutated,"bank-macros");need(rx.events==ry.events&&continual_checkpoint(x)==continual_checkpoint(y),"Held-out labels changed continual decisions");
    Continual tiny;auto failed=tiny.observe(first,"bank-macros",1);need(failed.search==1&&tiny.bank.empty()&&tiny.learner.current.json()==Graph{}.json(),"Partial graph committed");
    Continual full;for(int k=0;k<16;++k){Graph g;g.delay=true;g.leaves[0]={k%3-1,(k/3)%5-2,1};auto evidence=first.val;for(auto& e:evidence)e.y=g.predict(e);full.bank.push_back({g,evidence,{}});}auto limit=full.observe(world(0,2,9900005),"bank");need(limit.full&&full.bank.size()==16,"Bank limit not enforced");
    // Repeated identical observations with opposite outputs cannot be explained
    // by any deterministic graph; the guard also applies to missing grammar.
    Explorer explorer;explorer.rng.seed(123);Inquiry query{1,0,0};auto probe=probes()[0];probe.y=1;explorer.observe(query,probe);probe.y=-1;explorer.observe(query,probe);need(explorer.candidates[1].empty(),"Contradictory source stayed supported");
    auto bytes=explorer_checkpoint(explorer);atomic(directory/"explorer.model",bytes);auto again=restore_explorer(bytes);need(explorer_checkpoint(again)==bytes,"Explorer checkpoint round trip");auto q1=explorer.choose("active"),q2=again.choose("active");need(q1.source==0&&q1.probe==q2.probe&&q1.score==q2.score&&explorer_checkpoint(explorer)==explorer_checkpoint(again),"Query/overhead changed on restore");
    auto random1=restore_explorer(bytes),random2=random1;auto r1=random1.choose("random"),r2=random2.choose("random");need(r1.source==r2.source&&r1.probe==r2.probe,"Query RNG restore");
    auto swapped=restore_explorer(bytes);std::swap(swapped.candidates[0],swapped.candidates[1]);need(swapped.choose("active").source==1,"Source ID determined informativeness");
    auto clean=world(3,2,9900006);Explorer learned;for(int i=0;i<32;++i){auto q=learned.choose("active");if(q.source<0)break;auto e=probes()[q.probe];e.y=q.source==0?world_outcome(3,2,e):(i%2?1:-1);learned.observe(q,e);}need(accuracy(learned.prediction[0],clean.test)==1,"Active probes did not identify composed fixture");
    for(auto malformed:{saved.substr(0,30),saved+"trailing"}){bool rejected=false;try{restore_continual(malformed);}catch(const std::exception&){rejected=true;}need(rejected,"Malformed continual state accepted");}
    for(auto malformed:{bytes.substr(0,30),bytes+"trailing"}){bool rejected=false;try{restore_explorer(malformed);}catch(const std::exception&){rejected=true;}need(rejected,"Malformed explorer state accepted");}
    std::cout<<"PASS: automatic return routing, immutable witnesses, memory bound, held-out-label isolation, budget rollback, anti-unification/transfer, continual and inquiry checkpoint/next-decision restore, contradiction guard, source permutation and active composed prediction.\n";
}
std::string library_json(const Continual& c){std::vector<std::string> definitions;for(const auto& a:c.library)definitions.push_back(a.json());return array(definitions);}
void campaign_v3(const fs::path& directory,const fs::path& ui){
    need(!fs::exists(directory),"Output exists; choose a fresh directory");fs::create_directories(directory);const auto started=std::chrono::steady_clock::now();double start_cpu=cpu();
    std::vector<std::string> streams,inquiries;
    for(int seed:{201,202,203}){
        std::vector<std::pair<int,int>> order;for(int family=0;family<4;++family)for(int sign:{-1,1})order.push_back({family,sign});std::mt19937 order_rng(5000000+seed);std::shuffle(order.begin(),order.end(),order_rng);
        for(const std::string method:{"single","replay","bank","bank-macros"}){
            Continual learner;std::vector<std::string> windows;std::vector<Data> old_tasks;int64_t all_search=0,all_routing=0,all_retention=0,all_extraction=0,all_matching=0;int reused=0,macros=0;double accuracy_sum=0,pre_sum=0,return_sum=0,transfer_sum=0;
            const auto t0=std::chrono::steady_clock::now();double c0=cpu();
            for(int segment=0;segment<24;++segment)for(int window=0;window<2;++window){
                auto [family,sign]=order[size_t(segment%8)];int gain=sign*(segment>=8&&segment<16?2:1);auto data=world(family,gain,6000000+seed*10000+segment*100+window);
                double pre=accuracy(learner.learner.current,data.test);auto result=learner.observe(data,method);double after=accuracy(learner.learner.current,data.test);
                double active_old=0;for(const auto& old:old_tasks)active_old+=accuracy(learner.learner.current,old.test);if(!old_tasks.empty())active_old/=old_tasks.size();
                if(window==0&&segment<16)old_tasks.push_back(data);
                all_search+=result.search;all_routing+=result.routing;all_retention+=result.retention;all_extraction+=result.extraction;all_matching+=result.matching;reused+=result.reused;macros+=result.macro_accepted;accuracy_sum+=after;pre_sum+=pre;if(segment>=16)return_sum+=after;if(segment>=8&&segment<16)transfer_sum+=after;
                std::ostringstream event;event<<std::setprecision(10)<<"{\"window\":"<<learner.windows<<",\"phase\":"<<quote(segment<8?"acquisition":segment<16?"transfer":"return")<<",\"family\":"<<family<<",\"gain\":"<<gain<<",\"accuracy\":"<<after<<",\"pre_accuracy\":"<<pre<<",\"active_old_accuracy\":"<<active_old<<",\"route\":"<<result.route<<",\"reused\":"<<(result.reused?"true":"false")<<",\"inserted\":"<<(result.inserted?"true":"false")<<",\"macro_accepted\":"<<(result.macro_accepted?"true":"false")<<",\"bank_size\":"<<learner.bank.size()<<",\"search_work\":"<<result.search<<",\"routing_work\":"<<result.routing<<",\"retention_work\":"<<result.retention<<",\"extraction_work\":"<<result.extraction<<",\"matching_work\":"<<result.matching<<",\"before\":"<<result.before.json()<<",\"selected\":"<<result.after.json()<<",\"library\":"<<library_json(learner)<<",\"repairs\":"<<array(result.events)<<"}";windows.push_back(event.str());
            }
            std::string id=std::to_string(seed)+"-"+method;auto state=continual_checkpoint(learner);need(continual_checkpoint(restore_continual(state))==state,"Final continual restore");atomic(directory/(id+".model"),state);
            std::ostringstream row;row<<std::setprecision(10)<<"{\"id\":"<<quote(id)<<",\"seed\":"<<seed<<",\"method\":"<<quote(method)<<",\"accuracy\":"<<accuracy_sum/48<<",\"pre_accuracy\":"<<pre_sum/48<<",\"return_accuracy\":"<<return_sum/16<<",\"transfer_accuracy\":"<<transfer_sum/16<<",\"search_work\":"<<all_search<<",\"routing_work\":"<<all_routing<<",\"retention_work\":"<<all_retention<<",\"extraction_work\":"<<all_extraction<<",\"matching_work\":"<<all_matching<<",\"reused\":"<<reused<<",\"macro_uses\":"<<macros<<",\"bank_size\":"<<learner.bank.size()<<",\"library\":"<<library_json(learner)<<",\"checkpoint_bytes\":"<<state.size()<<",\"wall_seconds\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count()<<",\"cpu_seconds\":"<<cpu()-c0<<",\"windows\":"<<array(windows)<<"}";streams.push_back(row.str());
        }
        for(int family=0;family<4;++family)for(int gain:{-2,2})for(int cap:{4,8,16,32})for(const std::string method:{"random","error","active"}){
            Explorer explorer;explorer.rng.seed(7000000+seed*1000+family*10+gain+2);std::mt19937 noisy(8000000+seed*1000+family*10+gain+2);const int clean_source=seed%2;std::vector<std::string> events;int noisy_queries=0;const auto t0=std::chrono::steady_clock::now();double c0=cpu();
            for(int step=0;step<cap;++step){auto before_work=explorer.evaluations;auto query=explorer.choose(method);if(query.source<0)break;auto e=probes()[query.probe];int predicted=explorer.prediction[query.source].predict(e);size_t prior=explorer.candidates[query.source].size();bool unpredictable=query.source!=clean_source;e.y=unpredictable?int(noisy()%5)-2:world_outcome(family,gain,e);noisy_queries+=unpredictable;explorer.observe(query,e);
                std::ostringstream trace;trace<<std::setprecision(10)<<"{\"step\":"<<step+1<<",\"source\":"<<query.source<<",\"unpredictable\":"<<(unpredictable?"true":"false")<<",\"probe\":"<<query.probe<<",\"action\":"<<e.a<<",\"previous_action\":"<<e.prev<<",\"gate\":"<<e.b<<",\"noise_sensor\":"<<e.n<<",\"predicted\":"<<predicted<<",\"observed\":"<<e.y<<",\"score\":"<<query.score<<",\"before_hypotheses\":"<<prior<<",\"after_hypotheses\":"<<explorer.candidates[query.source].size()<<",\"evaluations\":"<<explorer.evaluations-before_work<<",\"selected\":"<<explorer.prediction[clean_source].json()<<"}";events.push_back(trace.str());
            }
            auto evaluation=world(family,gain,9000000+seed*1000+family*10+gain+2);std::string id=std::to_string(seed)+"-"+std::to_string(family)+"-"+std::to_string(gain)+"-"+method+"-"+std::to_string(cap);auto state=explorer_checkpoint(explorer);need(explorer_checkpoint(restore_explorer(state))==state,"Final inquiry restore");atomic(directory/(id+".inquiry"),state);
            std::ostringstream row;row<<std::setprecision(10)<<"{\"id\":"<<quote(id)<<",\"seed\":"<<seed<<",\"family\":"<<family<<",\"gain\":"<<gain<<",\"cap\":"<<cap<<",\"method\":"<<quote(method)<<",\"clean_source\":"<<clean_source<<",\"accuracy\":"<<accuracy(explorer.prediction[clean_source],evaluation.test)<<",\"queries\":"<<explorer.steps<<",\"noisy_queries\":"<<noisy_queries<<",\"unsupported_sources\":"<<int(explorer.candidates[0].empty())+int(explorer.candidates[1].empty())<<",\"evaluations\":"<<explorer.evaluations<<",\"checkpoint_bytes\":"<<state.size()<<",\"wall_seconds\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count()<<",\"cpu_seconds\":"<<cpu()-c0<<",\"events\":"<<array(events)<<"}";inquiries.push_back(row.str());
        }
    }
    rusage usage{};getrusage(RUSAGE_SELF,&usage);std::ostringstream out;out<<std::setprecision(10)<<"{\"schema\":3,\"status\":\"complete\",\"experiment\":\"repair-graph-v3\",\"hypothesis_count\":"<<hypotheses().size()<<",\"wall_seconds\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()<<",\"cpu_seconds\":"<<cpu()-start_cpu<<",\"peak_rss_kib\":"<<usage.ru_maxrss<<",\"streams\":"<<array(streams)<<",\"inquiries\":"<<array(inquiries)<<"}";atomic(directory/"status.json",out.str());auto html=read(ui);auto at=html.find("__REPORT_JSON__");need(at!=std::string::npos,"Missing report placeholder");html.replace(at,15,out.str());atomic(directory/"index.html",html);atomic(directory/"COMPLETE","repair-graph-v3\n");std::cout<<"Completed "<<streams.size()<<" continual streams (576 windows) and "<<inquiries.size()<<" inquiry conditions.\n";
}
