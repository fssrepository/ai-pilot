// Experiment lifecycle. Uses the single Learner implementation in main.cpp.
// This header is included once: the build still has one bounded compiler process.
struct Observation { int step; Metrics metrics; };
struct GrowthDecision { bool propose; const char* reason; };
GrowthDecision growth_decision(const std::vector<Observation>& history) {
    require(!history.empty(),"Missing validation history");
    const auto& last=history.back();
    if(last.metrics.accuracy>=0.98 && last.metrics.loss<=0.02)
        return {false,"target_reached"};
    if(last.step<200 || history.size()<3) return {false,"insufficient_history"};
    // Two consecutive 50-step validation windows, with less than 5% loss gain.
    // Partial windows after interruption do not qualify.
    for(size_t i=history.size()-2;i<history.size();++i) {
        const auto& before=history[i-1];const auto& after=history[i];
        if(after.step-before.step!=50) return {false,"insufficient_history"};
        if((before.metrics.loss-after.metrics.loss)/std::max(before.metrics.loss,1e-12)>=0.05)
            return {false,"still_improving"};
    }
    return {true,"validation_plateau"};
}
void policy_tests() {
    require(!growth_decision({{100,{0.1,0.8}},{150,{0.02,1}},{200,{0.001,1}}}).propose,"A solved task must not trigger growth");
    require(growth_decision({{100,{0.5,0.8}},{150,{0.49,0.8}},{200,{0.48,0.8}}}).propose,"Sustained plateau must trigger growth");
    require(!growth_decision({{100,{0.5,0.8}},{150,{0.3,0.8}},{200,{0.2,0.9}}}).propose,"An improving task must continue without growth");
    require(!growth_decision({{100,{0.5,0.8}},{175,{0.49,0.8}},{200,{0.48,0.8}}}).propose,"Partial windows must not manufacture a plateau");
    std::cout<<"PASS: validation-driven proposal, solved-task and warmup/improvement guards."<<std::endl;
}

enum class Phase:int { Baseline=0, Control=1, Candidate=2, Finished=3 };
struct Session {
    int seed,baseline_steps,extra_steps,resumes=0;
    Phase phase=Phase::Baseline;
    bool proposed=false,accepted=false;
    Metrics initial,baseline,control,growth,reference,retained;
    double control_seconds=0,growth_seconds=0;
    std::vector<Observation> history;
    std::unique_ptr<Learner> active;
    Session(int seed_,int steps,int extra,bool create=true):seed(seed_),baseline_steps(steps),extra_steps(extra) {
        if(create)active=std::make_unique<Learner>(seed);
    }
    void save(const fs::path& file) {
        torch::serialize::OutputArchive archive,learner;
        active->write(learner);archive.write("learner",learner);
        archive.write("experiment",torch::tensor({int64_t(1),int64_t(seed),int64_t(baseline_steps),int64_t(extra_steps),int64_t(phase),int64_t(proposed),int64_t(accepted),int64_t(resumes)},torch::kInt64));
        std::vector<double> values;
        for(auto m:{initial,baseline,control,growth,reference,retained}){values.push_back(m.loss);values.push_back(m.accuracy);}
        values.push_back(control_seconds);values.push_back(growth_seconds);
        archive.write("metrics",torch::tensor(values,torch::kFloat64));
        values.clear();for(auto h:history){values.push_back(h.step);values.push_back(h.metrics.loss);values.push_back(h.metrics.accuracy);}
        archive.write("history",torch::tensor(values,torch::kFloat64));
        auto temporary=file.string()+".tmp";archive.save_to(temporary);fs::rename(temporary,file);
    }
    static Session load(const fs::path& file) {
        torch::serialize::InputArchive archive,learner;archive.load_from(file.string());Tensor metadata,metrics,points;
        archive.read("experiment",metadata);require(metadata.numel()==8&&metadata[0].item<int>()==1,"Unsupported experiment checkpoint");
        Session state(metadata[1].item<int>(),metadata[2].item<int>(),metadata[3].item<int>(),false);
        require(state.baseline_steps>0&&state.baseline_steps<=10000&&state.extra_steps>0&&state.extra_steps<=2000,"Invalid saved budget");
        int phase=metadata[4].item<int>();require(phase>=0&&phase<=3,"Invalid saved phase");state.phase=Phase(phase);
        state.proposed=metadata[5].item<int>();state.accepted=metadata[6].item<int>();state.resumes=metadata[7].item<int>();
        archive.read("metrics",metrics);require(metrics.numel()==14&&torch::isfinite(metrics).all().item<bool>(),"Invalid saved metrics");
        int i=0;for(auto* m:{&state.initial,&state.baseline,&state.control,&state.growth,&state.reference,&state.retained}) {
            m->loss=metrics[i++].item<double>();m->accuracy=metrics[i++].item<double>();
        }
        state.control_seconds=metrics[12].item<double>();state.growth_seconds=metrics[13].item<double>();
        archive.read("history",points);require(points.numel()%3==0&&torch::isfinite(points).all().item<bool>(),"Invalid saved history");
        for(int64_t j=0;j<points.numel();j+=3)state.history.push_back({int(points[j].item<double>()),{points[j+1].item<double>(),points[j+2].item<double>()}});
        archive.read("learner",learner);state.active=Learner::read(learner);
        require(state.active->seed==state.seed,"Saved seed mismatch");
        require(state.active->step<=(state.phase==Phase::Baseline?state.baseline_steps:state.baseline_steps+state.extra_steps),"Saved step exceeds phase budget");
        return state;
    }
};
void write_report(Session& s,const fs::path& dir,const fs::path& ui) {
    auto all=dataset(),validation=subset(all,1),test=subset(all,2);
    s.active->save(dir/"selected.pt");
    // Only final selection reaches this boundary; test outcomes cannot trigger growth.
    auto tm=evaluate(*s.active->model,test);
    std::ostringstream exports;exports<<"[";
    auto control=Learner::load(dir/"fixed-control.pt");export_checked(*control->model,dir,"fixed-control",validation,exports);
    if(s.proposed){auto candidate=Learner::load(dir/"growth-candidate.pt");exports<<",";export_checked(*candidate->model,dir,"growth-candidate",validation,exports);}
    exports<<"]";export_gguf(*s.active->model,dir/"selected-f16.gguf",true);
    double deployed=0;{Runtime runtime(dir/"selected-f16.gguf");for(auto&e:test)deployed+=runtime.generate(e.prompt)==e.target;}deployed/=test.size();
    struct rusage usage{};getrusage(RUSAGE_SELF,&usage);
    auto decision=growth_decision(s.history);
    std::ostringstream report;report<<std::setprecision(9)<<"{\"schema\":2,\"seed\":"<<s.seed<<",\"threads\":2,\"batch_size\":16,\"baseline_steps\":"<<s.baseline_steps<<",\"extra_steps_per_branch\":"<<s.extra_steps
      <<",\"control_train_seconds\":"<<s.control_seconds<<",\"growth_train_seconds\":"<<s.growth_seconds
      <<",\"proposal\":\"validation-driven rule, not a learned policy\",\"growth_proposed\":"<<(s.proposed?"true":"false")<<",\"proposal_reason\":"<<safe_string(decision.reason)
      <<",\"comparison\":\"equal updates/examples when proposed; not equal CPU time\",\"growth_accepted\":"<<(s.accepted?"true":"false")
      <<",\"selected_layers\":"<<s.active->model->blocks.size()<<",\"selected_parameters\":"<<s.active->model->count()<<",\"initial_validation\":"<<json(s.initial)<<",\"baseline_validation\":"<<json(s.baseline)
      <<",\"control_validation\":"<<json(s.control)<<",\"growth_validation\":"<<(s.proposed?json(s.growth):"null")<<",\"replay_retention_accuracy\":"<<s.retained.accuracy
      <<",\"retention_scope\":\"training replay only; no separate old task yet\",\"test\":"<<json(tm)<<",\"gguf_test_accuracy\":"<<deployed
      <<",\"peak_rss_mib\":"<<usage.ru_maxrss/1024.0<<",\"resumes\":"<<s.resumes<<",\"exports\":"<<exports.str()<<",\"validation_history\":[";
    for(size_t i=0;i<s.history.size();++i){if(i)report<<",";report<<"{\"step\":"<<s.history[i].step<<",\"metrics\":"<<json(s.history[i].metrics)<<"}";}report<<"],\"examples\":[";
    for(size_t i=0;i<std::min(test.size(),size_t(8));++i){if(i)report<<",";report<<"{\"prompt\":"<<safe_string(test[i].prompt)<<",\"expected\":"<<safe_string(test[i].target)<<",\"predicted\":"<<safe_string(generate(*s.active->model,test[i].prompt))<<"}";}
    report<<"]}";std::ofstream(dir/"summary.json")<<report.str()<<"\n";
    if(!ui.empty()) {
        std::ifstream file(ui);require(bool(file),"Missing UI template");std::string html((std::istreambuf_iterator<char>(file)),{});
        auto pos=html.find("__REPORT_JSON__");require(pos!=std::string::npos,"UI placeholder missing");html.replace(pos,15,report.str());std::ofstream(dir/"index.html")<<html;
    }
    std::ofstream(dir/"COMPLETE")<<"All evaluation and GGUF checks completed.\n";
    std::cout<<report.str()<<std::endl;
}
void execute(Session& s,const fs::path& dir,const fs::path& ui,int pause_after) {
    auto all=dataset(),training=subset(all,0),validation=subset(all,1);
    std::ofstream records(dir/"episodes.jsonl");for(auto&e:all)records<<"{\"prompt\":"<<safe_string(e.prompt)<<",\"target\":"<<safe_string(e.target)<<",\"split\":"<<e.split<<"}\n";
    int chunks=0;
    while(s.phase!=Phase::Finished) {
        if(stop_requested || (pause_after>0&&chunks>=pause_after)) {
            s.save(dir/"state.pt");std::cout<<"PAUSED: phase="<<int(s.phase)<<" step="<<s.active->step<<"; resume "<<dir<<std::endl;return;
        }
        int target=s.phase==Phase::Baseline?s.baseline_steps:s.baseline_steps+s.extra_steps;
        if(s.active->step<target) {
            // Preserve absolute 50-step observation boundaries after a mid-chunk stop.
            int count=std::min(target-s.active->step,50-s.active->step%50);
            auto start=std::chrono::steady_clock::now();
            s.active->train(training,count,nullptr,s.phase==Phase::Baseline?"baseline":s.phase==Phase::Control?"fixed-control":"growth-candidate");
            double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            if(s.phase==Phase::Control)s.control_seconds+=seconds;
            if(s.phase==Phase::Candidate)s.growth_seconds+=seconds;
            if(s.phase==Phase::Baseline && (s.active->step%50==0 || s.active->step==target) && s.history.back().step!=s.active->step)
                s.history.push_back({s.active->step,evaluate(*s.active->model,validation)});
            s.save(dir/"state.pt");++chunks;continue;
        }
        if(s.phase==Phase::Baseline) {
            s.baseline=s.history.back().metrics;s.reference=evaluate(*s.active->model,training);
            s.active->save(dir/"baseline.pt");s.proposed=growth_decision(s.history).propose;s.phase=Phase::Control;
            std::cout<<"PROPOSAL: "<<growth_decision(s.history).reason<<" -> "<<(s.proposed?"try growth":"skip growth")<<std::endl;
        } else if(s.phase==Phase::Control) {
            s.control=evaluate(*s.active->model,validation);s.active->save(dir/"fixed-control.pt");
            if(s.proposed){s.active=Learner::load(dir/"baseline.pt");s.active->grow();s.phase=Phase::Candidate;}
            else {s.retained=evaluate(*s.active->model,training);s.phase=Phase::Finished;}
        } else {
            s.growth=evaluate(*s.active->model,validation);s.retained=evaluate(*s.active->model,training);
            s.accepted=accept(s.growth,s.control,s.retained,s.reference);s.active->save(dir/"growth-candidate.pt");
            if(!s.accepted)s.active=Learner::load(dir/"fixed-control.pt");s.phase=Phase::Finished;
        }
        s.save(dir/"state.pt");
    }
    write_report(s,dir,ui);
}
void compare_learners(Learner& a,Learner& b) {
    require(a.step==b.step && a.seed==b.seed && a.rng==b.rng,"Resumed sampling state differs");
    auto equal=[&](){auto ap=a.model->parameters(),bp=b.model->parameters();require(ap.size()==bp.size(),"Resumed architecture differs");for(size_t i=0;i<ap.size();++i)require(torch::equal(ap[i],bp[i]),"Resumed weights differ");};
    equal();auto train=subset(dataset(),0);a.train(train,1);b.train(train,1);equal();
}
void compare_sessions(const fs::path& first,const fs::path& second) {
    auto a=Session::load(first),b=Session::load(second);
    require(a.phase==b.phase && a.proposed==b.proposed && a.accepted==b.accepted && a.seed==b.seed && a.baseline_steps==b.baseline_steps && a.extra_steps==b.extra_steps,"Resumed experiment decision/config differs");
    require(a.history.size()==b.history.size(),"Resumed validation history size differs");
    for(size_t i=0;i<a.history.size();++i)require(a.history[i].step==b.history[i].step&&a.history[i].metrics.loss==b.history[i].metrics.loss&&a.history[i].metrics.accuracy==b.history[i].metrics.accuracy,"Resumed validation evidence differs");
    compare_learners(*a.active,*b.active);
    if(a.phase==Phase::Finished) {
        // Check rejected candidates too: equality of the selected control alone
        // would not establish correct resume of a candidate that was discarded.
        for(const auto& name:{"baseline.pt","fixed-control.pt","growth-candidate.pt"}) {
            bool left=fs::exists(first.parent_path()/name),right=fs::exists(second.parent_path()/name);
            require(left==right,"Resumed checkpoint set differs");
            if(left) {
                auto x=Learner::load(first.parent_path()/name),y=Learner::load(second.parent_path()/name);
                compare_learners(*x,*y);
            }
        }
    }
    std::cout<<"PASS: fresh-process experiment resume preserved decisions, validation history, every branch's weights, RNG and next optimizer update."<<std::endl;
}
