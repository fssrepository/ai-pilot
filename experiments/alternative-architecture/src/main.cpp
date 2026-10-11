// Repair Graph v1. Standard-library C++ learner; no transformer or Python training.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#include <sys/resource.h>
namespace fs=std::filesystem;
void need(bool b,const std::string& s){if(!b)throw std::runtime_error(s);}
bool bounded(int v,int low,int high){return v>=low&&v<=high;}
int integer(const std::string& s){size_t used=0;int v=std::stoi(s,&used);need(used==s.size(),"Invalid integer");return v;}
std::string quote(const std::string& s){std::ostringstream o;o<<'"';for(unsigned char c:s){if(c=='"'||c=='\\')o<<'\\'<<c;else if(c=='\n')o<<"\\n";else if(c<32)o<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<int(c)<<std::dec;else o<<c;}o<<'"';return o.str();}
std::string array(const std::vector<std::string>& xs){std::string s="[";for(auto& x:xs){if(s.size()>1)s+=",";s+=x;}return s+"]";}
void atomic(const fs::path& p,const std::string& s){auto tmp=p.string()+".tmp";{std::ofstream f(tmp);need(bool(f),"Cannot write "+tmp);f<<s;f.flush();need(bool(f),"Write failed");}fs::rename(tmp,p);}
std::string read(const fs::path& p){std::ifstream f(p);need(bool(f),"Cannot read "+p.string());return {std::istreambuf_iterator<char>(f),{}};}
double cpu(){rusage u{};getrusage(RUSAGE_SELF,&u);return u.ru_utime.tv_sec+u.ru_utime.tv_usec/1e6+u.ru_stime.tv_sec+u.ru_stime.tv_usec/1e6;}
struct Example{int x,a,prev,b,n,y;};
struct Data{std::vector<Example> train,val,test;};
// World parameters are confined to the simulator. Learner APIs accept Example only.
int world_outcome(int family,int gain,const Example& e){int delta=gain*(family>=2?e.prev:e.a);if((family==1||family==3)&&e.b)delta=0;return e.x+delta;}
Data world(int family,int gain,unsigned seed){std::mt19937 rng(seed);Data d;for(int split=0;split<3;++split){auto& out=split==0?d.train:split==1?d.val:d.test;int count=split==0?64:split==1?32:128;for(int i=0;i<count;++i){int a=int(rng()%3)-1,p=int(rng()%3)-1,b=rng()%2,n=rng()%2;int x=split==0?int(rng()%7)-3:split==1?-7+int(rng()%4):4+int(rng()%4);Example e{x,a,p,b,n,0};e.y=world_outcome(family,gain,e);out.push_back(e);}}return d;}
struct Coeff{int c0=0,c1=0,c2=0;int complexity()const{return (c0!=0)+(c1!=0)+(c2!=0);} };
struct Graph{
 bool delay=false;int split=-1;std::array<Coeff,2> leaves{};
 int leaf(const Example&e)const{return split==0?e.b:split==1?e.n:0;}
 int predict(const Example&e)const{auto c=leaves[leaf(e)];return e.x+c.c0+c.c1*e.a+(delay?c.c2*e.prev:0);}
 int complexity()const{return 3+(delay?2:0)+(split>=0?3:0)+leaves[0].complexity()+(split>=0?leaves[1].complexity():0);}
 std::string json()const{std::ostringstream o;o<<"{\"delay\":"<<(delay?"true":"false")<<",\"split\":"<<split<<",\"complexity\":"<<complexity()<<",\"coefficients\":[";for(int k=0;k<2;++k){if(k)o<<',';auto c=leaves[k];o<<'['<<c.c0<<','<<c.c1<<','<<c.c2<<']';}o<<"]}";return o.str();}
};
struct Budget{int64_t used=0,cap=20000;bool charge(){if(used>=cap)return false;++used;return true;}};
struct Exhausted{};
double mse(const Graph&g,const std::vector<Example>& d,Budget* budget=nullptr){double s=0;need(!d.empty(),"Empty evaluation");for(auto&e:d){if(budget&&!budget->charge())throw Exhausted{};double error=g.predict(e)-e.y;s+=error*error;}return s/d.size();}
double accuracy(const Graph&g,const std::vector<Example>& d){int correct=0;for(auto&e:d)correct+=g.predict(e)==e.y;return double(correct)/d.size();}
using Holes=std::array<std::array<bool,3>,2>;
Graph fit(Graph shape,const std::vector<Example>& data,Budget& budget,const Graph* constants=nullptr,const Holes* holes=nullptr){
 for(int leaf=0;leaf<(shape.split>=0?2:1);++leaf){double best=std::numeric_limits<double>::infinity();Coeff winner{};
  for(int c0=-1;c0<=1;++c0)for(int c1=-2;c1<=2;++c1)for(int c2=shape.delay?-2:0;c2<=(shape.delay?2:0);++c2){Coeff c{c0,c1,c2};if(constants&&holes){auto fixed=constants->leaves[leaf];auto h=(*holes)[leaf];if((!h[0]&&c0!=fixed.c0)||(!h[1]&&c1!=fixed.c1)||(!h[2]&&c2!=fixed.c2))continue;}double s=0;for(auto&e:data)if(shape.leaf(e)==leaf){if(!budget.charge())throw Exhausted{};double diff=e.x+c0+c1*e.a+c2*e.prev-e.y;s+=diff*diff;}
   if(s<best || (s==best&&c.complexity()<winner.complexity())){best=s;winner=c;}
  }shape.leaves[leaf]=winner;
 }return shape;
}
const std::array<std::string,3> ops={"refit","split","delay"};
Graph propose(const Graph& current,int op,const Data& d,Budget& budget){
 if(op==0)return fit(current,d.train,budget);
 if(op==2){auto shape=current;shape.delay=true;return fit(shape,d.train,budget);}
 if(current.split>=0)return fit(current,d.train,budget);
 Graph best;double loss=std::numeric_limits<double>::infinity();
 for(int sensor=0;sensor<2;++sensor){auto shape=current;shape.split=sensor;auto candidate=fit(shape,d.train,budget);double l=mse(candidate,d.train,&budget)+.001*candidate.complexity();if(l<loss){loss=l;best=candidate;}}
 return best;
}
using Descriptor=std::array<double,7>;
Descriptor describe(const Graph&g,const std::vector<Example>& data){Descriptor f{};double error2=0;std::array<double,4> denom{};for(auto&e:data){double r=e.y-g.predict(e);std::array<double,4> v={double(e.a),double(e.prev),double(e.a*e.b),double(e.prev*e.b)};error2+=r*r;for(int i=0;i<4;++i){f[i]+=r*v[i];denom[i]+=v[i]*v[i];}}for(int i=0;i<4;++i)f[i]=std::abs(f[i])/std::sqrt(error2*denom[i]+1e-12);f[4]=error2/(data.size()*16);f[5]=g.delay?2.:0.;f[6]=g.split>=0?2.:0.;return f;}
struct Record{Descriptor descriptor{};std::array<double,3> utility{};};
struct Learner{
 Graph current,archive;int policy_version=1;int previous_action=0;std::vector<Example> episodes;std::vector<Record> memory;std::mt19937 rng{42};
 std::array<int,3> order(bool learned,const Data& d)const{std::array<int,3> result={0,1,2};if(!learned||memory.empty())return result;auto f=describe(current,d.train);std::vector<std::pair<double,size_t>> nearest;for(size_t i=0;i<memory.size();++i){double distance=0;for(int j=0;j<(policy_version==1?5:7);++j)distance+=std::pow(f[j]-memory[i].descriptor[j],2);nearest.push_back({distance,i});}std::sort(nearest.begin(),nearest.end());std::array<double,3> u{};for(size_t k=0;k<std::min(size_t(policy_version==1?3:1),nearest.size());++k)for(int op=0;op<3;++op)u[op]+=memory[nearest[k].second].utility[op]/(.01+nearest[k].first);std::stable_sort(result.begin(),result.end(),[&](int a,int b){return u[a]>u[b];});return result;}
 int observe_and_predict(int x,int a,int b,int n){Example e{x,a,previous_action,b,n,0};int out=current.predict(e);previous_action=a;return out;}
};
void putgraph(std::ostream&f,const Graph&g){f<<g.delay<<' '<<g.split;for(auto c:g.leaves)f<<' '<<c.c0<<' '<<c.c1<<' '<<c.c2;f<<'\n';}
Graph getgraph(std::istream&f){Graph g;int delay=0;f>>delay>>g.split;need(delay==0||delay==1,"Invalid delay");g.delay=delay;need(g.split>=-1&&g.split<=1,"Invalid split");for(auto&c:g.leaves){f>>c.c0>>c.c1>>c.c2;need(bounded(c.c0,-1,1)&&bounded(c.c1,-2,2)&&bounded(c.c2,-2,2),"Invalid coefficient");need(g.delay||c.c2==0,"Invalid undelayed coefficient");}return g;}
std::string checkpoint(const Learner&l){std::ostringstream f;f<<"REPAIR_GRAPH "<<l.policy_version<<"\n"<<std::setprecision(17);putgraph(f,l.current);putgraph(f,l.archive);f<<l.previous_action<<'\n'<<l.episodes.size()<<'\n';for(auto&e:l.episodes)f<<e.x<<' '<<e.a<<' '<<e.prev<<' '<<e.b<<' '<<e.n<<' '<<e.y<<'\n';f<<l.memory.size()<<'\n';for(auto&r:l.memory){for(int i=0;i<(l.policy_version==1?5:7);++i)f<<r.descriptor[i]<<' ';for(auto x:r.utility)f<<x<<' ';f<<'\n';}f<<l.rng<<'\n';return f.str();}
Learner restore(const std::string&bytes){std::istringstream f(bytes);std::string magic;int version=0;f>>magic>>version;need(magic=="REPAIR_GRAPH"&&(version==1||version==2),"Unsupported checkpoint schema");Learner l;l.policy_version=version;l.current=getgraph(f);l.archive=getgraph(f);f>>l.previous_action;need(bounded(l.previous_action,-1,1),"Invalid history");size_t count=0;f>>count;need(count<=256,"Episode limit");for(size_t i=0;i<count;++i){Example e{};f>>e.x>>e.a>>e.prev>>e.b>>e.n>>e.y;need(bounded(e.x,-100,100)&&bounded(e.y,-110,110)&&bounded(e.a,-1,1)&&bounded(e.prev,-1,1)&&(e.b==0||e.b==1)&&(e.n==0||e.n==1),"Invalid episode");l.episodes.push_back(e);}f>>count;need(count<=size_t(version==1?64:512),"Repair memory limit");for(size_t i=0;i<count;++i){Record r;for(int j=0;j<(version==1?5:7);++j){f>>r.descriptor[j];need(std::isfinite(r.descriptor[j]),"Invalid descriptor");}for(auto&x:r.utility){f>>x;need(std::isfinite(x),"Invalid utility");}l.memory.push_back(r);}f>>l.rng;need(bool(f),"Truncated checkpoint");f>>std::ws;need(f.eof(),"Trailing checkpoint data");return l;}
std::vector<Record> train_policy(unsigned seed,int64_t& cost){std::vector<Record> memory;for(int family=0;family<3;++family)for(int gain:{-1,1}){auto d=world(family,gain,seed+100+family*20+(gain+1));Budget init;Graph g=fit(Graph{},world(0,1,seed).train,init);cost+=init.used;Record r;r.descriptor=describe(g,d.train);double old=mse(g,d.val)+.001*g.complexity();for(int op=0;op<3;++op){Budget b;auto candidate=propose(g,op,d,b);double fresh=mse(candidate,d.val,&b)+.001*candidate.complexity();r.utility[op]=std::max(0.,old-fresh)/b.used;cost+=b.used;}memory.push_back(r);}return memory;}
struct Outcome{Graph graph;double first_accuracy=0;int trials=0,recovery=-1,accepted=0,rejected=0;int64_t work=0;std::vector<std::string> events;};
Outcome adapt(Learner&l,const Data& d,const std::string&method,int64_t cap){Outcome out;Budget budget{0,cap};std::array<int,3> order=l.order(method=="learned-selector",d);for(int trial=0;trial<(method=="fixed-affine"?1:6);++trial){int op=method=="fixed-affine"?0:order[trial%3];auto before=l.current;double before_loss=mse(before,d.val),after_loss=before_loss;bool accept=false,exhausted=false;Graph candidate=before;int64_t start=budget.used;
 try{candidate=propose(before,op,d,budget);after_loss=mse(candidate,d.val,&budget);accept=after_loss+.001*candidate.complexity()<before_loss+.001*before.complexity()-1e-9;if(accept){l.current=candidate;++out.accepted;}else ++out.rejected;}catch(const Exhausted&){exhausted=true;++out.rejected;candidate=before;}
 ++out.trials;std::ostringstream event;event<<std::setprecision(10)<<"{\"trial\":"<<out.trials<<",\"operator\":"<<quote(ops[op])<<",\"accepted\":"<<(accept?"true":"false")<<",\"exhausted\":"<<(exhausted?"true":"false")<<",\"before_mse\":"<<before_loss<<",\"candidate_mse\":"<<(exhausted?"null":std::to_string(after_loss))<<",\"work\":"<<budget.used-start<<",\"total_work\":"<<budget.used<<",\"before\":"<<before.json()<<",\"candidate\":"<<(exhausted?"null":candidate.json())<<",\"selected\":"<<l.current.json()<<"}";out.events.push_back(event.str());
 if(trial==0)out.first_accuracy=accuracy(l.current,d.test);
 if(mse(l.current,d.val)==0){out.recovery=out.trials;break;}
 if(exhausted)break;
 // Re-rank only between full passes, using observed residuals, never test labels.
 if(trial==2)order=l.order(method=="learned-selector",d);
 }out.graph=l.current;out.work=budget.used;l.episodes=d.train;return out;}
using Key=std::tuple<int,int,int>;
double tabular(const Data&d){std::map<Key,std::map<int,int>> table;for(auto&e:d.train)table[{e.a,e.b,e.n}][e.y-e.x]++;int correct=0;for(auto&e:d.test){auto it=table.find({e.a,e.b,e.n});int prediction=e.x;if(it!=table.end()){int best=-1;for(auto[y,count]:it->second)if(count>best){prediction=e.x+y;best=count;}}correct+=prediction==e.y;}return double(correct)/d.test.size();}
void test(const fs::path&dir){fs::create_directories(dir);auto d=world(3,-2,91);for(auto&a:d.train)for(auto&b:d.test)need(a.x!=b.x,"Split leak");Budget b;Graph shape;shape.delay=true;shape.split=0;auto g=fit(shape,d.train,b);need(accuracy(g,d.test)==1,"Composed graph failed held-out prediction");Graph no_delay;no_delay.split=0;Budget b2;auto fixed=fit(no_delay,d.train,b2);need(accuracy(fixed,d.test)<1,"Memory control unexpectedly solves delayed world");for(auto e:d.test){int p=g.predict(e);e.n=1-e.n;need(g.predict(e)==p,"Spurious noise dependency");}
 Learner l;l.current=g;l.archive=fixed;l.previous_action=-1;l.episodes=d.train;int64_t cost=0;l.memory=train_policy(42,cost);l.rng.seed(123);atomic(dir/"roundtrip.model",checkpoint(l));auto r=restore(read(dir/"roundtrip.model"));need(checkpoint(l)==checkpoint(r),"Checkpoint round-trip");need(l.rng()==r.rng(),"RNG resume");need(l.observe_and_predict(2,1,0,1)==r.observe_and_predict(2,1,0,1),"Delayed state resume");auto a=adapt(l,world(1,2,777),"learned-selector",20000);auto c=adapt(r,world(1,2,777),"learned-selector",20000);need(a.events==c.events&&checkpoint(l)==checkpoint(r),"Next repair differs after restore");
 for(auto bad:{std::string("REPAIR_GRAPH 99\n"),checkpoint(l).substr(0,30),checkpoint(l)+"garbage"}){bool rejected=false;try{restore(bad);}catch(const std::exception&){rejected=true;}need(rejected,"Malformed checkpoint accepted");}
 Learner tiny;Budget base;auto stable=world(0,1,42);tiny.current=fit(Graph{},stable.train,base);auto copy=tiny.current.json();auto exhausted=adapt(tiny,d,"fixed-selector",1);need(exhausted.work==1&&tiny.current.json()==copy,"Budget mutated partial candidate");
 Learner reject;reject.current=g;auto old=reject.current.json();auto same=adapt(reject,d,"fixed-selector",20000);need(same.accepted==0&&reject.current.json()==old,"Rejection changed model");
 auto frozen=l.memory;auto policy_before=checkpoint(l);auto test_mutated=d;for(auto&e:test_mutated.test)e.y+=99;Learner a1=l,a2=l;auto o1=adapt(a1,d,"learned-selector",20000),o2=adapt(a2,test_mutated,"learned-selector",20000);need(o1.events==o2.events&&a1.current.json()==a2.current.json(),"Test leakage into selection");need(l.memory.size()==frozen.size()&&checkpoint(l)==policy_before,"Unexpected original mutation");
 // Synthetic policy evidence proves ranking responds to memory, rather than a fixed implementation path.
 Learner ranked;Record evidence;evidence.descriptor=describe(ranked.current,d.train);evidence.utility={0.,1.,2.};ranked.memory.push_back(evidence);
 need(ranked.order(true,d)==std::array<int,3>{2,1,0},"Repair evidence did not change ranking");
 need(ranked.order(false,d)==std::array<int,3>{0,1,2},"Memory ablation did not restore fixed order");
 need(checkpoint(a1)==checkpoint(a2),"Test labels altered saved experience or policy");
 bool invalid=false;try{integer("2oops");}catch(const std::exception&){invalid=true;}need(invalid,"Invalid CLI integer accepted");
 invalid=false;try{auto bytes=checkpoint(l);auto at=bytes.find("\n");bytes.replace(at+1,bytes.find("\n",at+1)-at-1,"0 -1 -2147483648 0 0 0 0 0");restore(bytes);}catch(const std::exception&){invalid=true;}need(invalid,"Out-of-range coefficient accepted");
 std::cout<<"PASS: position splits, compositional held-out prediction, delay necessity, noise invariance, exact checkpoint/RNG/history/next-repair restoration, malformed-schema rejection, budget rollback, rejected-edit preservation test-label isolation, evidence-driven ranking and memory ablation.\n";
}
void campaign(const fs::path&dir,const fs::path&ui){need(!fs::exists(dir),"Output exists; choose a fresh directory");fs::create_directories(dir);auto start=std::chrono::steady_clock::now();double cpu0=cpu();std::vector<std::string> rows;int64_t policy_work=0;std::vector<std::string> policies;const std::array<std::string,4> families={"parameter","gate","delay","gated-delay"};
 for(int seed:{42,43,44}){int64_t cost=0;auto memory=train_policy(seed,cost);policy_work+=cost;std::ostringstream pr;pr<<"{\"seed\":"<<seed<<",\"records\":"<<memory.size()<<",\"work\":"<<cost<<"}";policies.push_back(pr.str());
 auto old=world(0,1,seed);Budget base;Graph initial=fit(Graph{},old.train,base);
 for(int partition=0;partition<2;++partition)for(int family=0;family<4;++family)for(int gain:{-2,2}){if(partition==0 && !(family==0&&gain==-2) && family!=3)continue;int actual_gain=partition==0?(family==0?0:gain/2):gain;auto d=world(family,actual_gain,seed+1000+family*30+(actual_gain+2));std::string split=partition==0?"development":"test";
 for(int cap:{2500,20000})for(const auto& method:{"tabular","fixed-affine","fixed-selector","learned-selector"}){Learner learner;learner.current=initial;learner.archive=initial;learner.memory=memory;learner.rng.seed(seed);double c0=cpu();auto t0=std::chrono::steady_clock::now();Outcome out;
 if(std::string(method)=="tabular"){out.graph=initial;out.work=d.train.size();out.first_accuracy=tabular(d);}else out=adapt(learner,d,method,cap);
 double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count(),spent=cpu()-c0;double acc=std::string(method)=="tabular"?tabular(d):accuracy(learner.current,d.test);std::string id=std::to_string(seed)+"-"+split+"-"+families[family]+"-"+std::to_string(actual_gain)+"-"+std::to_string(cap)+"-"+method;
 size_t bytes=0;if(std::string(method)!="tabular"){auto state=checkpoint(learner);bytes=state.size();atomic(dir/(id+".model"),state);need(checkpoint(restore(state))==state,"Campaign checkpoint verification");}
 std::ostringstream row;row<<std::setprecision(10)<<"{\"id\":"<<quote(id)<<",\"seed\":"<<seed<<",\"partition\":"<<quote(split)<<",\"family\":"<<quote(families[family])<<",\"gain\":"<<actual_gain<<",\"method\":"<<quote(method)<<",\"cap\":"<<cap<<",\"accuracy\":"<<acc<<",\"first_accuracy\":"<<out.first_accuracy<<",\"validation_mse\":"<<(std::string(method)=="tabular"?"null":std::to_string(mse(learner.current,d.val)))<<",\"old_before\":"<<accuracy(initial,old.test)<<",\"old_after\":"<<(std::string(method)=="tabular"?"null":std::to_string(accuracy(learner.current,old.test)))<<",\"archived_old\":"<<(std::string(method)=="tabular"?"null":std::to_string(accuracy(learner.archive,old.test)))<<",\"work\":"<<out.work<<",\"trials\":"<<out.trials<<",\"recovery_trials\":"<<out.recovery<<",\"accepted\":"<<out.accepted<<",\"rejected\":"<<out.rejected<<",\"wall_seconds\":"<<elapsed<<",\"cpu_seconds\":"<<spent<<",\"checkpoint_bytes\":"<<bytes<<",\"graph\":"<<(std::string(method)=="tabular"?"null":learner.current.json())<<",\"events\":"<<array(out.events)<<"}";rows.push_back(row.str());
 }
 }
 }
 double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();rusage usage{};getrusage(RUSAGE_SELF,&usage);std::ostringstream out;out<<std::setprecision(10)<<"{\"schema\":1,\"status\":\"complete\",\"experiment\":\"repair-graph-v1\",\"policy_training_work\":"<<policy_work<<",\"policy_runs\":"<<array(policies)<<",\"wall_seconds\":"<<elapsed<<",\"cpu_seconds\":"<<cpu()-cpu0<<",\"peak_rss_kib\":"<<usage.ru_maxrss<<",\"results\":"<<array(rows)<<"}";auto json=out.str();atomic(dir/"status.json",json);auto html=read(ui);auto at=html.find("__REPORT_JSON__");need(at!=std::string::npos,"Missing report placeholder");html.replace(at,15,json);atomic(dir/"index.html",html);atomic(dir/"COMPLETE","repair-graph-v1\n");std::cout<<"Completed "<<rows.size()<<" conditions in "<<elapsed<<" seconds; RSS "<<usage.ru_maxrss<<" KiB.\n";
}
#include "repair_policy.h"
#include "continual.h"
void print_prediction(Learner learner,int argc,char** argv,int start) {
    need(argc==start+4||argc==start+5,"Prediction needs x action gate noise [previous_action]");
    int x=integer(argv[start]),a=integer(argv[start+1]),b=integer(argv[start+2]),n=integer(argv[start+3]);
    if(argc==start+5)learner.previous_action=integer(argv[start+4]);
    need(bounded(x,-100,100)&&bounded(a,-1,1)&&bounded(learner.previous_action,-1,1)&&bounded(b,0,1)&&bounded(n,0,1),"Input outside pilot bounds");
    std::cout<<"{\"next_x\":"<<learner.observe_and_predict(x,a,b,n)<<",\"graph\":"<<learner.current.json()<<"}\n";
}
int main(int argc,char** argv){
    try {
        need(argc>=3,"Usage: repair-graph test[-v2|-v3] DIR | campaign[-v2|-v3] DIR UI | predict CHECKPOINT x action gate noise [previous_action] | predict-inquiry CHECKPOINT source x action gate noise [previous_action]");
        const std::string mode=argv[1];
        if(mode=="test")test(argv[2]);
        else if(mode=="test-v2")test_v2(argv[2]);
        else if(mode=="test-v3")test_v3(argv[2]);
        else if(mode=="campaign"||mode=="campaign-v2"||mode=="campaign-v3"){
            need(argc==4,"Campaign needs DIR UI");
            if(mode=="campaign")campaign(argv[2],argv[3]);
            else if(mode=="campaign-v2")campaign_v2(argv[2],argv[3]);
            else campaign_v3(argv[2],argv[3]);
        }else if(mode=="predict"){
            auto bytes=read(argv[2]);auto learner=bytes.rfind("REPAIR_CONTINUAL ",0)==0?restore_continual(bytes).learner:restore(bytes);
            print_prediction(learner,argc,argv,3);
        }else if(mode=="predict-inquiry"){
            need(argc>=8,"Inquiry prediction needs source and inputs");int source=integer(argv[3]);need(bounded(source,0,1),"Source must be 0 or 1");
            auto explorer=restore_explorer(read(argv[2]));Learner learner;learner.current=explorer.prediction[source];print_prediction(learner,argc,argv,4);
        }else throw std::runtime_error("Unknown mode");
        return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
