// Continual-learning campaign; model, training, checkpoint and export owners stay in main.cpp.
struct Worlds {
    std::vector<Example> a,b,train_a,train_b,val_a,val_b,test_a,test_b,changed_a,changed_b;
    Worlds() {
        auto original=dataset();
        for(auto e:original) {
            auto next=e;
            int x=e.prompt[2]-'0',y=e.prompt[6]-'0';char action=e.prompt[10];bool blocked=e.prompt[14]=='1';
            int nx=x,ny=y;
            if(!blocked){nx=std::clamp(x-(action=='R')+(action=='L'),0,3);ny=std::clamp(y-(action=='U')+(action=='D'),0,3);}
            next.target=std::to_string(nx)+","+std::to_string(ny)+";";
            e.prompt="w=0,"+e.prompt;next.prompt="w=1,"+next.prompt;
            a.push_back(e);b.push_back(next);
            if(e.split==2 && e.target!=next.target){changed_a.push_back(e);changed_b.push_back(next);}
        }
        train_a=subset(a,0);train_b=subset(b,0);val_a=subset(a,1);val_b=subset(b,1);test_a=subset(a,2);test_b=subset(b,2);
    }
};
int64_t update_work(int layers,int sequence) {
    return int64_t(3)*16*sequence*(int64_t(WIDTH)*VOCAB+layers*(4LL*WIDTH*WIDTH+3LL*WIDTH*FF+2LL*sequence*WIDTH));
}
double process_cpu() { struct rusage u{};getrusage(RUSAGE_SELF,&u);return u.ru_utime.tv_sec+u.ru_utime.tv_usec/1e6+u.ru_stime.tv_sec+u.ru_stime.tv_usec/1e6; }
void atomic_text(const fs::path& p,const std::string& value) {
    auto tmp=p.string()+".tmp";{std::ofstream out(tmp);require(bool(out),"Cannot write "+tmp);out<<value;out.flush();require(bool(out),"Write failed "+tmp);}fs::rename(tmp,p);
}
struct CampaignPaused {};
struct CampaignLive {
    fs::path dir,ui;
    std::vector<std::string> events,rows;
    int pause_after,chunks=0;
    explicit CampaignLive(fs::path path,fs::path template_path,int pause):dir(path),ui(template_path),pause_after(pause) {
        std::ifstream old(dir/"events.jsonl");std::string line;while(std::getline(old,line))if(!line.empty())events.push_back(line);
        std::vector<fs::path> completed;
        for(auto& f:fs::recursive_directory_iterator(dir))if(f.path().filename()=="result.json")completed.push_back(f.path());
        std::sort(completed.begin(),completed.end());
        for(auto&p:completed){std::ifstream in(p);rows.emplace_back(std::istreambuf_iterator<char>(in),std::istreambuf_iterator<char>());}
    }
    std::string snapshot(const std::string& status) {
        auto now=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        std::ostringstream out;out<<"{\"schema\":1,\"status\":"<<safe_string(status)<<",\"updated_ms\":"<<now<<",\"total_conditions\":12,\"events\":[";
        for(size_t i=0;i<events.size();++i){if(i)out<<",";out<<events[i];}out<<"],\"results\":[";
        for(size_t i=0;i<rows.size();++i){if(i)out<<",";out<<rows[i];}out<<"]}";return out.str();
    }
    void publish(const std::string& status) {atomic_text(dir/"status.json",snapshot(status));}
    void observe(Learner& l,const Worlds& w,int seed,const std::string& branch,const std::string& phase) {
        auto old=evaluate(*l.model,w.val_a),fresh=evaluate(*l.model,w.val_b);
        std::ostringstream e;e<<"{\"seed\":"<<seed<<",\"branch\":"<<safe_string(branch)<<",\"phase\":"<<safe_string(phase)<<",\"step\":"<<l.step<<",\"layers\":"<<l.model->blocks.size()<<",\"old_validation\":"<<json(old)<<",\"new_validation\":"<<json(fresh)<<"}";
        events.push_back(e.str());{std::ofstream log(dir/"events.jsonl",std::ios::app);log<<e.str()<<"\n";}
        publish("running");
    }
    void finish() {
        auto data=snapshot("complete");atomic_text(dir/"status.json",data);
        std::ifstream in(ui);require(bool(in),"Missing campaign report template");std::string html((std::istreambuf_iterator<char>(in)),{});
        auto at=html.find("__REPORT_JSON__");require(at!=std::string::npos,"Missing report placeholder");html.replace(at,15,data);atomic_text(dir/"index.html",html);
        atomic_text(dir/"COMPLETE","All 12 conditions evaluated and exported.\n");
    }
};
struct SegmentStats { double wall=0,cpu=0;int updates=0; };
// Save statistics and learner in one atomic archive, so interrupted accounting stays paired.
void save_segment(Learner& l,const fs::path& path,int start,int budget,SegmentStats stats) {
    torch::serialize::OutputArchive out;l.write(out);
    out.write("segment",torch::tensor({double(start),double(budget),stats.wall,stats.cpu},torch::kFloat64));
    auto tmp=path.string()+".tmp";out.save_to(tmp);fs::rename(tmp,path);
}
SegmentStats train_segment(std::unique_ptr<Learner>& learner,const Worlds& w,const std::vector<Example>& train,const std::vector<Example>* replay,
                           int steps,const fs::path& file,CampaignLive& live,int seed,const std::string& branch,const std::string& phase) {
    int start=learner->step;SegmentStats stats;
    if(fs::exists(file)) {
        torch::serialize::InputArchive in;in.load_from(file.string());Tensor meta;in.read("segment",meta);
        require(meta.numel()==4&&meta[0].item<int>()==start&&meta[1].item<int>()==steps,"Saved segment budget mismatch");
        learner=Learner::read(in);stats.wall=meta[2].item<double>();stats.cpu=meta[3].item<double>();
    } else {save_segment(*learner,file,start,steps,stats);live.observe(*learner,w,seed,branch,phase);}
    require(learner->step>=start&&learner->step<=start+steps,"Invalid segment progress");
    while(learner->step<start+steps) {
        if(stop_requested || (live.pause_after>0&&live.chunks>=live.pause_after))throw CampaignPaused{};
        auto clock=std::chrono::steady_clock::now();double cpu=process_cpu();
        int count=std::min(start+steps-learner->step,50-(learner->step-start)%50);
        learner->train(train,count,nullptr,branch+"/"+phase,replay);
        stats.wall+=std::chrono::duration<double>(std::chrono::steady_clock::now()-clock).count();stats.cpu+=process_cpu()-cpu;
        save_segment(*learner,file,start,steps,stats);++live.chunks;
        live.observe(*learner,w,seed,branch,phase);
        if(stop_requested)throw CampaignPaused{};
    }
    stats.updates=steps;return stats;
}
bool campaign_growth_needed(Metrics old,Metrics fresh,Metrics reference) {
    return !(fresh.accuracy>=0.98&&fresh.loss<=0.02&&old.accuracy>=reference.accuracy-0.02);
}
void continual_tests(const fs::path& dir) {
    Worlds w;require(w.a.size()==160&&w.train_a.size()==112&&w.val_a.size()==24&&w.test_a.size()==24,"World partition sizes");
    require(!w.changed_b.empty(),"Changed-answer test subset is empty");
    for(size_t i=0;i<w.a.size();++i) {
        require(w.a[i].split==w.b[i].split&&w.a[i].prompt.substr(4)==w.b[i].prompt.substr(4),"Cross-world split leakage");
        require(w.a[i].prompt.size()==w.b[i].prompt.size(),"Unequal example shapes");
        if(w.b[i].prompt=="w=1,x=1,y=2,a=R,b=0>")require(w.b[i].target=="0,2;","Reversed action rule");
    }
    int seq=w.a[0].prompt.size()+3;
    require(update_work(5,seq)>update_work(4,seq),"Compute proxy must charge for extra layer");
    require(!campaign_growth_needed({.01,1},{.01,1},{.01,1}),"Solved switch should skip growth");
    require(campaign_growth_needed({.3,.7},{.01,1},{.01,1}),"Lost old knowledge should trigger comparison");
    for(int layers:{4,5}) {
        Learner direct(42,layers);direct.train(w.train_b,2,nullptr,"resume-check",&w.train_a);
        auto file=dir/("continual-resume-"+std::to_string(layers)+".pt");
        save_segment(direct,file,0,4,{1.25,2.5,2});
        auto restored=Learner::load(file);
        direct.train(w.train_b,2,nullptr,"resume-check",&w.train_a);restored->train(w.train_b,2,nullptr,"resume-check",&w.train_a);
        compare_learners(direct,*restored);
        torch::serialize::InputArchive in;in.load_from(file.string());Tensor meta;in.read("segment",meta);
        require(meta[0].item<int>()==0&&meta[1].item<int>()==4&&meta[2].item<double>()==1.25&&meta[3].item<double>()==2.5,"Segment accounting did not survive save/load");
    }
    std::cout<<"PASS: two-world split/rules, changed-answer subset, compute accounting and adaptation trigger."<<std::endl;
    std::cout<<"PASS: replay training and atomic segment accounting resume exactly for four/five layers."<<std::endl;
}
void campaign(const fs::path& dir,const fs::path& ui,bool resume,int pause_after=0) {
    require(pause_after>=0,"Invalid campaign pause interval");
    const std::string config="continual-v1 seeds=42,43,44 pretrain=600 adapt=400 warm=100 replay=4/16\n";
    if(!resume){require(!fs::exists(dir),"Campaign output exists; use campaign-resume");fs::create_directories(dir);atomic_text(dir/"CONFIG",config);}
    else {std::ifstream f(dir/"CONFIG");std::string saved((std::istreambuf_iterator<char>(f)),{});require(saved==config,"Campaign configuration mismatch");if(fs::exists(dir/"COMPLETE")){std::cout<<"Already complete; no files changed."<<std::endl;return;}}
    Worlds w;int seq=w.a[0].prompt.size()+3;int64_t c4=update_work(4,seq),c5=update_work(5,seq),adapt_budget=400*c4;
    if(!resume) {
        std::ofstream data(dir/"episodes.jsonl");for(const auto* world:{&w.a,&w.b})for(auto&e:*world)data<<"{\"prompt\":"<<safe_string(e.prompt)<<",\"target\":"<<safe_string(e.target)<<",\"split\":"<<e.split<<"}\n";
    }
    CampaignLive live(dir,ui,pause_after);live.publish("running");auto wall_start=std::chrono::steady_clock::now();double cpu_start=process_cpu();
    try {
      for(int seed:{42,43,44}) {
        auto seed_dir=dir/("seed-"+std::to_string(seed));fs::create_directories(seed_dir);
        for(const std::string condition:{"fixed4-no-replay","fixed4-replay","fixed5-replay","adaptive-replay"}) {
            auto out=seed_dir/condition;fs::create_directories(out);if(fs::exists(out/"result.json"))continue;
            int layers=condition=="fixed5-replay"?5:4;int64_t cost=layers==5?c5:c4;
            torch::manual_seed(seed);auto learner=std::make_unique<Learner>(seed,layers);
            int pre_steps=600*c4/cost;auto basefile=seed_dir/("base-"+std::to_string(layers)+".pt");
            auto pre=train_segment(learner,w,w.train_a,nullptr,pre_steps,basefile,live,seed,"base-"+std::to_string(layers),"world-a");
            auto reference=evaluate(*learner->model,w.val_a);
            // Reporting only: these test metrics never enter proposal or selection.
            auto initial_a=evaluate(*learner->model,w.test_a),initial_b=evaluate(*learner->model,w.test_b);
            auto initial_changed=evaluate(*learner->model,w.changed_b);
            SegmentStats work;int64_t used=0;bool proposed=false,accepted=false;
            Metrics control_new,control_old,candidate_new,candidate_old;
            auto add=[&](SegmentStats s,int64_t c){work.wall+=s.wall;work.cpu+=s.cpu;work.updates+=s.updates;used+=s.updates*c;};
            if(condition=="adaptive-replay") {
                auto warmfile=out/"warm.pt";
                add(train_segment(learner,w,w.train_b,&w.train_a,100,warmfile,live,seed,condition,"warm"),c4);
                auto old=evaluate(*learner->model,w.val_a),fresh=evaluate(*learner->model,w.val_b);
                proposed=campaign_growth_needed(old,fresh,reference);
                if(proposed) {
                    int64_t each=(adapt_budget-used)/2;
                    add(train_segment(learner,w,w.train_b,&w.train_a,int(each/c4),out/"control.pt",live,seed,condition,"control"),c4);
                    control_new=evaluate(*learner->model,w.val_b);control_old=evaluate(*learner->model,w.val_a);
                    auto candidate=Learner::load(warmfile);candidate->grow();
                    add(train_segment(candidate,w,w.train_b,&w.train_a,int(each/c5),out/"candidate.pt",live,seed,condition,"candidate"),c5);
                    candidate_new=evaluate(*candidate->model,w.val_b);candidate_old=evaluate(*candidate->model,w.val_a);
                    accepted=accept(candidate_new,control_new,candidate_old,reference)&&candidate_old.accuracy>=control_old.accuracy;
                    if(accepted)learner=std::move(candidate);
                } else add(train_segment(learner,w,w.train_b,&w.train_a,int((adapt_budget-used)/c4),out/"continued.pt",live,seed,condition,"continue"),c4);
            } else {
                const auto* replay=condition=="fixed4-no-replay"?nullptr:&w.train_a;
                add(train_segment(learner,w,w.train_b,replay,int(adapt_budget/cost),out/"adapted.pt",live,seed,condition,"world-b"),cost);
            }
            require(used<=adapt_budget,"Adaptation exceeded declared compute proxy budget");
            learner->save(out/"selected.pt");
            auto old=evaluate(*learner->model,w.test_a),fresh=evaluate(*learner->model,w.test_b);
            auto changed_old=evaluate(*learner->model,w.changed_a),changed_new=evaluate(*learner->model,w.changed_b);
            std::ostringstream exports;export_checked(*learner->model,out,"selected",w.val_b,exports);
            int ga=0,gb=0,gchanged=0;{Runtime runtime(out/"selected-f16.gguf");for(auto&e:w.test_a)ga+=runtime.generate(e.prompt)==e.target;for(auto&e:w.test_b)gb+=runtime.generate(e.prompt)==e.target;for(auto&e:w.changed_b)gchanged+=runtime.generate(e.prompt)==e.target;}
            std::ostringstream result;result<<std::setprecision(10)<<"{\"seed\":"<<seed<<",\"condition\":"<<safe_string(condition)<<",\"layers\":"<<learner->model->blocks.size()<<",\"parameters\":"<<learner->model->count()
              <<",\"pretrain_updates\":"<<pre_steps<<",\"adaptation_updates_total\":"<<work.updates<<",\"selected_adaptation_updates\":"<<learner->step-pre_steps
              <<",\"pretrain_proxy\":"<<pre_steps*cost<<",\"pretrain_proxy_budget\":"<<600*c4<<",\"adaptation_proxy\":"<<used<<",\"adaptation_proxy_budget\":"<<adapt_budget
              <<",\"pretrain_cpu_seconds\":"<<pre.cpu<<",\"pretrain_wall_seconds\":"<<pre.wall<<",\"adaptation_cpu_seconds\":"<<work.cpu<<",\"adaptation_wall_seconds\":"<<work.wall
              <<",\"growth_proposed\":"<<(proposed?"true":"false")<<",\"growth_accepted\":"<<(accepted?"true":"false")
              <<",\"initial_old_test\":"<<json(initial_a)<<",\"initial_new_test\":"<<json(initial_b)<<",\"initial_changed_test\":"<<json(initial_changed)
              <<",\"old_test\":"<<json(old)<<",\"new_test\":"<<json(fresh)<<",\"changed_old_test\":"<<json(changed_old)<<",\"changed_new_test\":"<<json(changed_new)<<",\"changed_test_count\":"<<w.changed_b.size()
              <<",\"forgetting_pp\":"<<100*(initial_a.accuracy-old.accuracy)<<",\"acquisition_pp\":"<<100*(fresh.accuracy-initial_b.accuracy)
              <<",\"control_old_validation\":"<<(proposed?json(control_old):"null")<<",\"control_new_validation\":"<<(proposed?json(control_new):"null")
              <<",\"candidate_old_validation\":"<<(proposed?json(candidate_old):"null")<<",\"candidate_new_validation\":"<<(proposed?json(candidate_new):"null")
              <<",\"gguf_old_accuracy\":"<<double(ga)/w.test_a.size()<<",\"gguf_new_accuracy\":"<<double(gb)/w.test_b.size()<<",\"gguf_changed_accuracy\":"<<double(gchanged)/w.changed_b.size()<<",\"export\":"<<exports.str()<<"}";
            atomic_text(out/"result.json",result.str());live.rows.push_back(result.str());live.publish("running");
            std::cout<<"RESULT: seed="<<seed<<" "<<condition<<" old="<<old.accuracy<<" new="<<fresh.accuracy<<" forgetting_pp="<<100*(initial_a.accuracy-old.accuracy)<<" growth="<<accepted<<std::endl;
        }
      }
      struct rusage usage{};getrusage(RUSAGE_SELF,&usage);
      std::ostringstream resource;resource<<"{\"invocation_wall_seconds\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-wall_start).count()<<",\"invocation_cpu_seconds\":"<<process_cpu()-cpu_start<<",\"peak_rss_mib\":"<<usage.ru_maxrss/1024.0<<",\"note\":\"Current invocation only; pretraining checkpoints shared by four-layer conditions.\"}";
      atomic_text(dir/"resources.json",resource.str());live.finish();
    } catch(const CampaignPaused&) {live.publish("paused");std::cout<<"PAUSED: campaign checkpoint saved; use campaign-resume "<<dir<<std::endl;}
      catch(...) {live.publish("failed");throw;}
}
