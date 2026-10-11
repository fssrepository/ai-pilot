// Follow-up uses the existing simulator, graph, fitting, checkpoint and renderer contracts.
// Pilot-v1 entry points remain unchanged; schema 2 adds structural state to repair memory.
std::vector<Record> train_policy_v2(unsigned seed, int64_t& work) {
    std::vector<Record> memory;
    Budget base;
    const Graph initial = fit(Graph{}, world(0, 1, 1000000 + seed).train, base);
    work += base.used;
    for (int family = 0; family < 3; ++family) {
        for (int sign = 0; sign < 2; ++sign) {
            for (int replicate = 0; replicate < 4; ++replicate) {
                auto data = world(family, sign == 0 ? -1 : 1,
                                  2000000 + seed * 1000 + family * 100 + sign * 10 + replicate);
                std::vector<Graph> states{initial};
                auto add = [&](const Graph& g) {
                    if (std::none_of(states.begin(), states.end(), [&](const Graph& s) {
                            return s.json() == g.json();
                        })) states.push_back(g);
                };
                for (int op = 0; op < 3; ++op) {
                    Budget budget;
                    Graph candidate = propose(initial, op, data, budget);
                    double fresh = mse(candidate, data.val, &budget) + .001 * candidate.complexity();
                    if (op == 0 || fresh < mse(initial, data.val) + .001 * initial.complexity() - 1e-9)
                        add(candidate);
                    work += budget.used;
                }
                for (const auto& graph : states) {
                    Record record;
                    record.descriptor = describe(graph, data.train);
                    double previous = mse(graph, data.val) + .001 * graph.complexity();
                    for (int op = 0; op < 3; ++op) {
                        Budget budget;
                        auto candidate = propose(graph, op, data, budget);
                        double fresh = mse(candidate, data.val, &budget) + .001 * candidate.complexity();
                        record.utility[op] = std::max(0., (previous - fresh) / std::max(previous, 1e-12))
                                           - .01 * budget.used / 20000.;
                        work += budget.used;
                    }
                    memory.push_back(record);
                }
            }
        }
    }
    need(memory.size() <= 512, "Policy memory limit exceeded");
    return memory;
}

Outcome adapt_v2(Learner& learner, const Data& data, const std::string& method, int64_t cap) {
    if (method == "legacy-fixed") return adapt(learner, data, "fixed-selector", cap);
    Outcome result;
    Budget budget{0, cap};
    std::array<bool, 3> tried{false, false, false};
    bool fitted = false;
    for (int trial = 0; trial < 6; ++trial) {
        std::array<int, 3> priority{0, 1, 2};
        if (method == "learned-v1" || method == "learned-v2") {
            priority = learner.order(true, data);
        } else {
            need(method.size() == 9 && method.substr(0, 6) == "fixed-", "Invalid v2 method");
            for (int i = 0; i < 3; ++i) priority[i] = method[6 + i] - '0';
            auto checked = priority;
            std::sort(checked.begin(), checked.end());
            need(checked == std::array<int, 3>{0, 1, 2}, "Invalid operator permutation");
        }
        int op = trial == 0 ? 0 : -1;
        if (trial > 0) {
            for (int candidate : priority) {
                bool unavailable = tried[candidate] || (candidate == 0 && fitted)
                    || (candidate == 1 && learner.current.split >= 0)
                    || (candidate == 2 && learner.current.delay);
                if (!unavailable) { op = candidate; break; }
            }
        }
        if (op < 0) break;
        Graph before = learner.current, candidate = before;
        const double old_loss = mse(before, data.val);
        double new_loss = old_loss;
        bool accepted = false, exhausted = false;
        auto start = budget.used;
        try {
            candidate = propose(before, op, data, budget);
            new_loss = mse(candidate, data.val, &budget);
            accepted = new_loss + .001 * candidate.complexity()
                       < old_loss + .001 * before.complexity() - 1e-9;
            if (accepted) {
                learner.current = candidate;
                tried.fill(false);
                fitted = true;
                ++result.accepted;
            } else ++result.rejected;
        } catch (const Exhausted&) {
            exhausted = true;
            ++result.rejected;
        }
        tried[op] = true;
        ++result.trials;
        std::ostringstream e;
        e << std::setprecision(10) << "{\"trial\":" << result.trials
          << ",\"operator\":" << quote(ops[op]) << ",\"accepted\":" << (accepted ? "true" : "false")
          << ",\"exhausted\":" << (exhausted ? "true" : "false")
          << ",\"before_mse\":" << old_loss << ",\"candidate_mse\":"
          << (exhausted ? "null" : std::to_string(new_loss)) << ",\"work\":" << budget.used - start
          << ",\"total_work\":" << budget.used << ",\"before\":" << before.json()
          << ",\"candidate\":" << (exhausted ? "null" : candidate.json())
          << ",\"selected\":" << learner.current.json() << "}";
        result.events.push_back(e.str());
        if (trial == 0) result.first_accuracy = accuracy(learner.current, data.test);
        if (mse(learner.current, data.val) == 0) { result.recovery = result.trials; break; }
        if (exhausted) break;
    }
    result.graph = learner.current;
    result.work = budget.used;
    learner.episodes = data.train;
    return result;
}

void test_v2(const fs::path& directory) {
    fs::create_directories(directory);
    Learner learner;
    learner.policy_version = 2;
    Budget base;
    learner.current = fit(Graph{}, world(0, 1, 1000101).train, base);
    learner.archive = learner.current;
    int64_t cost = 0;
    learner.memory = train_policy_v2(101, cost);
    need(learner.memory.size() > 6 && cost > 0, "Missing multi-state repair experience");
    auto data = world(3, -2, 9910101); // test fixture, outside final measurement streams
    atomic(directory / "schema2.model", checkpoint(learner));
    auto restored = restore(read(directory / "schema2.model"));
    need(checkpoint(learner) == checkpoint(restored), "Schema 2 round trip");
    auto a = adapt_v2(learner, data, "learned-v2", 20000);
    auto b = adapt_v2(restored, data, "learned-v2", 20000);
    need(a.events == b.events && checkpoint(learner) == checkpoint(restored), "Next v2 repair after restore");
    auto changed_test = data;
    for (auto& e : changed_test.test) e.y += 17;
    Learner l1 = restore(read(directory / "schema2.model")), l2 = l1;
    auto policy = l1.memory;
    auto x = adapt_v2(l1, data, "learned-v2", 20000);
    auto y = adapt_v2(l2, changed_test, "learned-v2", 20000);
    need(x.events == y.events && checkpoint(l1) == checkpoint(l2), "Final labels changed v2 learning");
    for (size_t i = 0; i < policy.size(); ++i)
        need(policy[i].descriptor == l1.memory[i].descriptor && policy[i].utility == l1.memory[i].utility,
             "Frozen policy changed during evaluation");
    auto original = restore(read(directory / "schema2.model"));
    auto limited = adapt_v2(original, data, "learned-v2", 1);
    need(limited.work == 1 && limited.accepted == 0, "Partial candidate escaped budget");
    // With unlimited-enough declared fixture budget, no redundant refit after an accepted fitted graph.
    Learner fixed = restore(read(directory / "schema2.model"));
    auto sequence = adapt_v2(fixed, data, "fixed-012", 20000);
    need(sequence.events.size() <= 4, "Redundant search trials");
    for (size_t i = 1; i < sequence.events.size(); ++i)
        need(sequence.events[i].find("\"operator\":\"refit\"") == std::string::npos, "Repeated fitting of same topology");
    need(fixed.current.delay && fixed.current.split == 0 && accuracy(fixed.current, data.test) == 1,
         "Composed fixture not solved under declared budget");
    std::cout << "PASS: schema-2 exact restore and next repair, policy freeze, final-label isolation, budget rollback, bounded eligible trials and composed held-out fixture.\n";
}

void campaign_v2(const fs::path& directory, const fs::path& ui) {
    need(!fs::exists(directory), "Output exists; choose a fresh directory");
    fs::create_directories(directory);
    const auto started = std::chrono::steady_clock::now();
    const double initial_cpu = cpu();
    std::vector<std::string> rows, policy_rows;
    const std::array<std::string, 9> methods{
        "legacy-fixed", "fixed-012", "fixed-021", "fixed-102", "fixed-120", "fixed-201", "fixed-210", "learned-v1", "learned-v2"};
    const std::array<std::string, 4> families{"parameter", "gate", "delay", "gated-delay"};
    int64_t all_policy_work = 0;
    for (int seed : {101, 102, 103}) {
        int64_t cost1 = 0, cost2 = 0;
        auto memory1 = train_policy(4000000 + seed, cost1);
        auto memory2 = train_policy_v2(seed, cost2);
        all_policy_work += cost1 + cost2;
        std::ostringstream policy;
        policy << "{\"seed\":" << seed << ",\"v1_work\":" << cost1 << ",\"v2_work\":" << cost2
               << ",\"v1_records\":" << memory1.size() << ",\"v2_records\":" << memory2.size() << "}";
        policy_rows.push_back(policy.str());
        auto old = world(0, 1, 1000000 + seed);
        Budget base;
        Graph initial = fit(Graph{}, old.train, base);
        for (int family = 0; family < 4; ++family) for (int sign = 0; sign < 2; ++sign) {
            int gain = sign == 0 ? -2 : 2;
            auto data = world(family, gain, 3000000 + seed * 1000 + family * 100 + sign * 10);
            for (int cap : {2500, 6000, 20000}) for (const auto& method : methods) {
                Learner learner;
                learner.current = learner.archive = initial;
                learner.rng.seed(seed);
                if (method == "learned-v1") learner.memory = memory1;
                if (method == "learned-v2") { learner.memory = memory2; learner.policy_version = 2; }
                const auto start = std::chrono::steady_clock::now();
                const double cpu0 = cpu();
                auto outcome = adapt_v2(learner, data, method, cap);
                double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                double spent = cpu() - cpu0;
                std::string id = std::to_string(seed) + "-test-" + families[family] + "-" + std::to_string(gain)
                    + "-" + std::to_string(cap) + "-" + method;
                auto state = checkpoint(learner);
                atomic(directory / (id + ".model"), state);
                need(checkpoint(restore(state)) == state, "V2 campaign checkpoint round trip");
                std::ostringstream r;
                r << std::setprecision(10) << "{\"id\":" << quote(id) << ",\"seed\":" << seed
                  << ",\"partition\":\"test\",\"family\":" << quote(families[family]) << ",\"gain\":" << gain
                  << ",\"method\":" << quote(method) << ",\"cap\":" << cap
                  << ",\"accuracy\":" << accuracy(learner.current, data.test) << ",\"first_accuracy\":" << outcome.first_accuracy
                  << ",\"validation_mse\":" << mse(learner.current, data.val) << ",\"old_before\":" << accuracy(initial, old.test)
                  << ",\"old_after\":" << accuracy(learner.current, old.test) << ",\"archived_old\":" << accuracy(learner.archive, old.test)
                  << ",\"work\":" << outcome.work << ",\"trials\":" << outcome.trials
                  << ",\"recovery_trials\":" << outcome.recovery << ",\"accepted\":" << outcome.accepted
                  << ",\"rejected\":" << outcome.rejected << ",\"wall_seconds\":" << wall << ",\"cpu_seconds\":" << spent
                  << ",\"checkpoint_bytes\":" << state.size() << ",\"graph\":" << learner.current.json()
                  << ",\"events\":" << array(outcome.events) << "}";
                rows.push_back(r.str());
            }
        }
    }
    rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
    std::vector<std::string> method_json;
    for (const auto& m : methods) method_json.push_back(quote(m));
    std::ostringstream report;
    report << std::setprecision(10) << "{\"schema\":2,\"status\":\"complete\",\"experiment\":\"repair-graph-v2\",\"methods\":"
           << array(method_json) << ",\"caps\":[2500,6000,20000],\"policy_training_work\":" << all_policy_work
           << ",\"policy_runs\":" << array(policy_rows) << ",\"wall_seconds\":"
           << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()
           << ",\"cpu_seconds\":" << cpu() - initial_cpu << ",\"peak_rss_kib\":" << usage.ru_maxrss
           << ",\"results\":" << array(rows) << "}";
    atomic(directory / "status.json", report.str());
    auto html = read(ui);
    auto at = html.find("__REPORT_JSON__");
    need(at != std::string::npos, "Missing report placeholder");
    html.replace(at, 15, report.str());
    atomic(directory / "index.html", html);
    atomic(directory / "COMPLETE", "repair-graph-v2\n");
    std::cout << "Completed " << rows.size() << " v2 conditions.\n";
}
