#include "LNS.h"
#include "ECBS.h"
#include <queue>
#include <LightGBM/c_api.h>


LNS::LNS(const Instance& instance, double time_limit, const string & init_algo_name, 
         const string & replan_algo_name, const string & destory_name, int neighbor_size, 
         int num_of_iterations, bool use_init_lns, const string & init_destory_name, 
         bool use_sipp, const string & path_results, const string & path_train_data, 
         int num_initial_solutions, int num_lns_runs, int screen, PIBTPPS_option pipp_option) :
         BasicLNS(instance, time_limit, neighbor_size, screen),
         init_algo_name(init_algo_name), replan_algo_name(replan_algo_name),
         num_of_iterations(num_of_iterations), use_init_lns(use_init_lns),
         init_destory_name(init_destory_name), path_table(instance.map_size), 
         pipp_option(pipp_option), path_results(path_results), 
         path_train_data(path_train_data), num_initial_solutions(num_initial_solutions), 
         num_lns_runs(num_lns_runs), generate_init_solutions_time(generate_init_solutions_time), 
         prediction_time(prediction_time), num_agents(instance.getDefaultNumberOfAgents()),
         num_agents_f(instance.getDefaultNumberOfAgents())

{
    start_time = Time::now();
    replan_time_limit = time_limit / 100;
    if (destory_name == "Adaptive")
    {
        ALNS = true;
        destroy_weights.assign(DESTORY_COUNT, 1);
        decay_factor = 0.01;
        reaction_factor = 0.01;
    }
    else if (destory_name == "RandomWalk")
        destroy_strategy = RANDOMWALK;
    else if (destory_name == "Intersection")
        destroy_strategy = INTERSECTION;
    else if (destory_name == "Random")
        destroy_strategy = RANDOMAGENTS;
    else
    {
        cerr << "Destroy heuristic " << destory_name << " does not exists. " << endl;
        exit(-1);
    }

    // Initialize matrices and related vectors
    initializeVariablesForML();

    agents.reserve(num_agents);
    for (int i = 0; i < num_agents; i++)
        agents.emplace_back(instance, i, use_sipp);
    preprocessing_time = ((fsec)(Time::now() - start_time)).count();
    if (screen >= 2)
        cout << "Pre-processing time = " << preprocessing_time << " seconds." << endl;
}

// Initialization of variables in constructor
void LNS::initializeVariablesForML() 
{
    all_mins.reserve(4);
    all_maxs.reserve(4);
    all_sums.reserve(4);
    all_avgs.reserve(4);
    graph_dist.reserve(num_agents);
    start_x.reserve(num_agents);
    start_y.reserve(num_agents);
    goal_x.reserve(num_agents);
    goal_y.reserve(num_agents);
    goal_degree.reserve(num_agents);
    delay.reserve(num_agents);
    ratio_delay_dist.reserve(num_agents);
    agents_output_paths.reserve(num_agents);
    heat_values.reserve(num_agents);

    for (int i = 0; i < num_agents; ++i) 
    {
        graph_dist.push_back(0.0f);  
        start_x.push_back(0);     
        start_y.push_back(0);     
        goal_x.push_back(0);      
        goal_y.push_back(0);      
        goal_degree.push_back(0); 
        delay.push_back(0.0f);      
        ratio_delay_dist.push_back(0.0f);
        agents_output_paths.emplace_back(); 
        heat_values.emplace_back();
    }

    time_steps_vertex_degree.reserve(num_agents);
    for (int i = 0; i < num_agents; ++i)
        time_steps_vertex_degree.emplace_back(4, 0.0f);

    feature_matrix.reserve(num_initial_solutions);
    for (int i = 0; i < num_initial_solutions; ++i)
        feature_matrix.emplace_back(64, 0.0f);

    target_matrix_per_seed.reserve(num_lns_runs);
    for (int i = 0; i < num_lns_runs; ++i) 
        target_matrix_per_seed.emplace_back(0, 0.0);
}


// Reset path_table, agents,...
void LNS::resetComplexMembers() 
{
    path_table.reset();
    iteration_stats.clear();
    sum_of_costs = 0;
    agents.clear();
    agents.reserve(num_agents);
    for (int i = 0; i < num_agents; i++)
        agents.emplace_back(instance, i, true);

    initial_sum_of_costs = -1;
    sum_of_costs_lowerbound = -1;
    restart_times = 0;
    average_group_size = -1;
    destroy_weights.assign(DESTORY_COUNT, 1);
    decay_factor = 0.01;
    reaction_factor = 0.01;
    num_of_failures = 0;

    neighbor.old_paths.clear();
}

bool LNS::generateInitialSolution(bool use_feature_matrix, int seed) 
{         
    // reset path_table, agents,...
    resetComplexMembers();

    // only for statistic analysis, and thus is not included in runtime
    sum_of_distances = 0;
    for (const auto & agent : agents)
    {
        sum_of_distances += agent.path_planner->my_heuristic[agent.path_planner->start_location];
    }
    
    initial_solution_runtime = 0;
    start_time = Time::now();
    bool succ = getInitialSolution();
    initial_solution_runtime = ((fsec)(Time::now() - start_time)).count();
    if (!succ && initial_solution_runtime < time_limit)
    {
        if (use_init_lns)
        {
            init_lns = new InitLNS(instance, agents, time_limit - initial_solution_runtime,
                    replan_algo_name,init_destory_name, neighbor_size, screen);
            succ = init_lns->run();
            if (succ) // accept new paths
            {
                path_table.reset();
                for (const auto & agent : agents)
                {
                    path_table.insertPath(agent.id, agent.path);
                }
                init_lns->clear();
                initial_sum_of_costs = init_lns->sum_of_costs;
                sum_of_costs = initial_sum_of_costs;
            }
            initial_solution_runtime = ((fsec)(Time::now() - start_time)).count();
        }
        else // use random restart
        {
            while (!succ && initial_solution_runtime < time_limit)
            {
                succ = getInitialSolution();
                initial_solution_runtime = ((fsec)(Time::now() - start_time)).count();
                restart_times++;
            }
        }
    }
    if (succ && use_feature_matrix)
    {
        start_time_feat_calc = Time::now();
        calculateSolutionFeatureMatrix(seed);
        generate_init_solutions_time += initial_solution_runtime + ((fsec)(Time::now() - start_time_feat_calc)).count();
        initial_solution_runtime = ((fsec)(Time::now() - start_time)).count();
    }
    iteration_stats.emplace_back(neighbor.agents.size(),
                                initial_sum_of_costs, initial_solution_runtime, init_algo_name);
    runtime = initial_solution_runtime;
    if (succ)
    {
        if (screen >= 1)
            cout << "Initial solution cost = " << initial_sum_of_costs << ", "
                << "runtime = " << initial_solution_runtime << endl;
    }
    else
    {
        cout << "Failed to find an initial solution in "
            << runtime << " seconds after  " << restart_times << " restarts" << endl;
        return false; // terminate because no initial solution is found
    }
    return succ;
}

void LNS::runALNS(int run)
{
    bool succ = false;
    double runtime_last_improvement = 0;
    start_time = Time::now();

    while (runtime < time_limit && iteration_stats.size() <= num_of_iterations)
    {
        runtime = initial_solution_runtime + ((fsec)(Time::now() - start_time)).count();
        if(screen >= 1)
            validateSolution();
        if (ALNS)
            chooseDestroyHeuristicbyALNS();

        switch (destroy_strategy)
        {
            case RANDOMWALK:
                succ = generateNeighborByRandomWalk();
                break;
            case INTERSECTION:
                succ = generateNeighborByIntersection();
                break;
            case RANDOMAGENTS:
                neighbor.agents.resize(agents.size());
                for (int i = 0; i < (int)agents.size(); i++)
                    neighbor.agents[i] = i;
                if (neighbor.agents.size() > neighbor_size)
                {
                    std::random_shuffle(neighbor.agents.begin(), neighbor.agents.end());
                    neighbor.agents.resize(neighbor_size);
                }
                succ = true;
                break;
            default:
                cerr << "Wrong neighbor generation strategy" << endl;
                exit(-1);
        }
        if(!succ)
            continue;

        // store the neighbor information
        neighbor.old_paths.resize(neighbor.agents.size());
        neighbor.old_sum_of_costs = 0;
        for (int i = 0; i < (int)neighbor.agents.size(); i++)
        {
            if (replan_algo_name == "PP")
                neighbor.old_paths[i] = agents[neighbor.agents[i]].path;
            path_table.deletePath(neighbor.agents[i], agents[neighbor.agents[i]].path);
            neighbor.old_sum_of_costs += agents[neighbor.agents[i]].path.size() - 1;
        }

        if (replan_algo_name == "EECBS")
            succ = runEECBS();
        else if (replan_algo_name == "CBS")
            succ = runCBS();
        else if (replan_algo_name == "PP")
            succ = runPP();
        else
        {
            cerr << "Wrong replanning strategy" << endl;
            exit(-1);
        }

        if (ALNS) // update destroy heuristics
        {
            if (neighbor.old_sum_of_costs > neighbor.sum_of_costs )
            {
                runtime_last_improvement = initial_solution_runtime + ((fsec)(Time::now() - start_time)).count();
                destroy_weights[selected_neighbor] =
                        reaction_factor * (neighbor.old_sum_of_costs - neighbor.sum_of_costs) / neighbor.agents.size()
                        + (1 - reaction_factor) * destroy_weights[selected_neighbor];
            }
            else
            {
                destroy_weights[selected_neighbor] =
                        (1 - decay_factor) * destroy_weights[selected_neighbor];
            }
        }
        runtime = initial_solution_runtime + ((fsec)(Time::now() - start_time)).count();
        sum_of_costs += neighbor.sum_of_costs - neighbor.old_sum_of_costs;
        if (screen >= 1)
            cout << "Iteration " << iteration_stats.size() << ", "
                << "group size = " << neighbor.agents.size() << ", "
                << "solution cost = " << sum_of_costs << ", "
                << "remaining time = " << time_limit - runtime << endl;
        iteration_stats.emplace_back(neighbor.agents.size(), sum_of_costs, runtime, replan_algo_name);
    }

    target_matrix_per_seed[run-1].first = sum_of_costs;
    target_matrix_per_seed[run-1].second = runtime_last_improvement;

    average_group_size = - iteration_stats.front().num_of_agents;
    for (const auto& data : iteration_stats)
        average_group_size += data.num_of_agents;
    if (average_group_size > 0)
        average_group_size /= (double)(iteration_stats.size() - 1);

    cout << getSolverName() << ": "
        << "runtime = " << runtime << ", "
        << "iterations = " << iteration_stats.size() << ", "
        << "solution cost = " << sum_of_costs << ", "
        << "initial solution cost = " << initial_sum_of_costs << ", "
        << "failed iterations = " << num_of_failures << endl;
}

bool LNS::run()
{
    bool use_feature_matrix = false;
    int best_seed = 1;
    generateInitialSolution(use_feature_matrix, best_seed);
    runALNS(num_lns_runs);

    return true;
}

void LNS::generateTrainData() 
{   
    bool use_feature_matrix = true;
    vector<pair<double, double>> target_matrix_medians(num_initial_solutions);
    vector<int> target_ranks(num_initial_solutions);

    for (int seed = 1; seed < num_initial_solutions + 1; seed++) 
    {
        for (int run = 1; run < num_lns_runs + 1; run++) 
        {
            srand((int) seed);
            generateInitialSolution(use_feature_matrix, seed);
            validateSolution();

            srand(std::random_device()());
            runALNS(run);
            validateSolution();

        }
        calculateMedians(target_matrix_medians[seed - 1]);
    }

    calculateRanks(target_matrix_medians, target_ranks);

    writeSolutionFeatureMatrixToFile(target_ranks); 
}

int LNS::getBestRankedIndex(const std::string& model_file) 
{    
    // Load trained model
    int num_iterations;
    BoosterHandle booster;
    LGBM_BoosterCreateFromModelfile(model_file.c_str(), &num_iterations, &booster);

    // Make predictions
    int64_t out_len = num_initial_solutions;
    double out_result[num_initial_solutions]; 

    static const char* predict_parameters = "predict_disable_shape_check=true num_threads=1";  // static const for slight optimization
    LGBM_BoosterPredictForMat(booster, feature_matrix.data(), C_API_DTYPE_FLOAT32, num_initial_solutions, 64, 1, 
                              C_API_PREDICT_NORMAL, 0, num_iterations, predict_parameters, &out_len, out_result);

    // Find index of the best ranked element
    int best_index = std::distance(out_result, std::max_element(out_result, out_result + num_initial_solutions)) + 1;

    return best_index;
}

bool LNS::runML()
{
    bool use_feature_matrix = true;

    for (int seed = 1; seed < num_initial_solutions + 1; seed++)
    {
        srand((int) seed);
        generateInitialSolution(use_feature_matrix, seed);
        validateSolution();
    }

    start_time = Time::now();
 
    int best_seed = getBestRankedIndex(path_results + "/trained_ml_model.txt");
 
    prediction_time = ((fsec)(Time::now() - start_time)).count();

    use_feature_matrix = false;
    srand((int) best_seed);
    generateInitialSolution(use_feature_matrix, best_seed);

    runtime = generate_init_solutions_time + prediction_time;

    srand(std::random_device()());
    runALNS(num_lns_runs);

    return true;
}

// Calculate medians for each run of each initial solution
void LNS::calculateMedians(pair<double, double>& target_matrix_medians) 
{
    vector<int> first_column;
    vector<double> second_column;

    // Reserve capacity in the vectors based on the size of target_matrix_per_seed
    first_column.reserve(num_initial_solutions);
    second_column.reserve(num_initial_solutions);

    for (const auto& p : target_matrix_per_seed) 
    {
        first_column.push_back(p.first);
        second_column.push_back(p.second);
    }

    sort(first_column.begin(), first_column.end());
    sort(second_column.begin(), second_column.end());

    double median_first, median_second;
    size_t n = num_lns_runs;

    if (n % 2 == 0) // even-sized
    {  
        median_first = (first_column[n / 2 - 1] + first_column[n / 2]) / 2.0;
        median_second = (second_column[n / 2 - 1] + second_column[n / 2]) / 2.0;
    } 
    else // odd-sized
    {  
        median_first = first_column[n / 2];
        median_second = second_column[n / 2];
    }

    target_matrix_medians = {median_first, median_second};
}

// Assign ranks to each initial solution
void LNS::calculateRanks(const vector<pair<double, double>>& target_matrix_medians, vector<int>& target_ranks) 
{
    // Create vector of indices
    vector<int> indices(num_initial_solutions);
    iota(indices.begin(), indices.end(), 0);

    //Sort indices based on the values in target_matrix_medians
    sort(indices.begin(), indices.end(), 
              [&target_matrix_medians](int a, int b) 
              {
                  if (target_matrix_medians[a].first == target_matrix_medians[b].first) 
                  {
                      return target_matrix_medians[a].second > target_matrix_medians[b].second;
                  }
                  return target_matrix_medians[a].first > target_matrix_medians[b].first;
              });

    //Assign ranks based on the order of sorted indices
    for (int i = 0; i < indices.size(); ++i) {
        target_ranks[indices[i]] = i;
    }
}


bool LNS::getInitialSolution()
{
    neighbor.agents.resize(agents.size());
    for (int i = 0; i < (int)agents.size(); i++)
        neighbor.agents[i] = i;
    neighbor.old_sum_of_costs = MAX_COST;
    neighbor.sum_of_costs = 0;
    bool succ = false;
    if (init_algo_name == "EECBS")
        succ = runEECBS();
    else if (init_algo_name == "PP")
        succ = runPP();
    else if (init_algo_name == "PIBT")
        succ = runPIBT();
    else if (init_algo_name == "PPS")
        succ = runPPS();
    else if (init_algo_name == "winPIBT")
        succ = runWinPIBT();
    else if (init_algo_name == "CBS")
        succ = runCBS();
    else
    {
        cerr <<  "Initial MAPF solver " << init_algo_name << " does not exist!" << endl;
        exit(-1);
    }
    if (succ)
    {
        initial_sum_of_costs = neighbor.sum_of_costs;
        sum_of_costs = neighbor.sum_of_costs;
        return true;
    }
    else
    {
        return false;
    }

}

bool LNS::runEECBS()
{
    vector<SingleAgentSolver*> search_engines;
    search_engines.reserve(neighbor.agents.size());
    for (int i : neighbor.agents)
    {
        search_engines.push_back(agents[i].path_planner);
    }

    ECBS ecbs(search_engines, screen - 1, &path_table);
    ecbs.setPrioritizeConflicts(true);
    ecbs.setDisjointSplitting(false);
    ecbs.setBypass(true);
    ecbs.setRectangleReasoning(true);
    ecbs.setCorridorReasoning(true);
    ecbs.setHeuristicType(heuristics_type::WDG, heuristics_type::GLOBAL);
    ecbs.setTargetReasoning(true);
    ecbs.setMutexReasoning(false);
    ecbs.setConflictSelectionRule(conflict_selection::EARLIEST);
    ecbs.setNodeSelectionRule(node_selection::NODE_CONFLICTPAIRS);
    ecbs.setSavingStats(false);
    double w;
    if (iteration_stats.empty())
        w = 5; // initial run
    else
        w = 1.1; // replan
    ecbs.setHighLevelSolver(high_level_solver_type::EES, w);
    runtime = ((fsec)(Time::now() - start_time)).count();
    double T = time_limit - runtime;
    if (!iteration_stats.empty()) // replan
        T = min(T, replan_time_limit);
    bool succ = ecbs.solve(T, 0);
    if (succ && ecbs.solution_cost < neighbor.old_sum_of_costs) // accept new paths
    {
        auto id = neighbor.agents.begin();
        for (size_t i = 0; i < neighbor.agents.size(); i++)
        {
            agents[*id].path = *ecbs.paths[i];
            path_table.insertPath(agents[*id].id, agents[*id].path);
            ++id;
        }
        neighbor.sum_of_costs = ecbs.solution_cost;
        if (sum_of_costs_lowerbound < 0)
            sum_of_costs_lowerbound = ecbs.getLowerBound();
    }
    else // stick to old paths
    {
        if (!neighbor.old_paths.empty())
        {
            for (int id : neighbor.agents)
            {
                path_table.insertPath(agents[id].id, agents[id].path);
            }
            neighbor.sum_of_costs = neighbor.old_sum_of_costs;
        }
        if (!succ)
            num_of_failures++;
    }
    return succ;
}
bool LNS::runCBS()
{
    if (screen >= 2)
        cout << "old sum of costs = " << neighbor.old_sum_of_costs << endl;
    vector<SingleAgentSolver*> search_engines;
    search_engines.reserve(neighbor.agents.size());
    for (int i : neighbor.agents)
    {
        search_engines.push_back(agents[i].path_planner);
    }

    CBS cbs(search_engines, screen - 1, &path_table);
    cbs.setPrioritizeConflicts(true);
    cbs.setDisjointSplitting(false);
    cbs.setBypass(true);
    cbs.setRectangleReasoning(true);
    cbs.setCorridorReasoning(true);
    cbs.setHeuristicType(heuristics_type::WDG, heuristics_type::ZERO);
    cbs.setTargetReasoning(true);
    cbs.setMutexReasoning(false);
    cbs.setConflictSelectionRule(conflict_selection::EARLIEST);
    cbs.setNodeSelectionRule(node_selection::NODE_CONFLICTPAIRS);
    cbs.setSavingStats(false);
    cbs.setHighLevelSolver(high_level_solver_type::ASTAR, 1);
    runtime = ((fsec)(Time::now() - start_time)).count();
    double T = time_limit - runtime; // time limit
    if (!iteration_stats.empty()) // replan
        T = min(T, replan_time_limit);
    bool succ = cbs.solve(T, 0);
    if (succ && cbs.solution_cost <= neighbor.old_sum_of_costs) // accept new paths
    {
        auto id = neighbor.agents.begin();
        for (size_t i = 0; i < neighbor.agents.size(); i++)
        {
            agents[*id].path = *cbs.paths[i];
            path_table.insertPath(agents[*id].id, agents[*id].path);
            ++id;
        }
        neighbor.sum_of_costs = cbs.solution_cost;
        if (sum_of_costs_lowerbound < 0)
            sum_of_costs_lowerbound = cbs.getLowerBound();
    }
    else // stick to old paths
    {
        if (!neighbor.old_paths.empty())
        {
            for (int id : neighbor.agents)
            {
                path_table.insertPath(agents[id].id, agents[id].path);
            }
            neighbor.sum_of_costs = neighbor.old_sum_of_costs;

        }
        if (!succ)
            num_of_failures++;
    }
    return succ;
}
bool LNS::runPP()
{
    auto shuffled_agents = neighbor.agents;
    std::random_shuffle(shuffled_agents.begin(), shuffled_agents.end());
    if (screen >= 2) {
        for (auto id : shuffled_agents)
            cout << id << "(" << agents[id].path_planner->my_heuristic[agents[id].path_planner->start_location] <<
                "->" << agents[id].path.size() - 1 << "), ";
        cout << endl;
    }
    int remaining_agents = (int)shuffled_agents.size();
    auto p = shuffled_agents.begin();
    neighbor.sum_of_costs = 0;
    runtime = ((fsec)(Time::now() - start_time)).count();
    double T = time_limit - runtime; // time limit
    if (!iteration_stats.empty()) // replan
        T = min(T, replan_time_limit);
    auto time = Time::now();
    ConstraintTable constraint_table(instance.num_of_cols, instance.map_size, &path_table);
    while (p != shuffled_agents.end() && ((fsec)(Time::now() - time)).count() < T)
    {
        int id = *p;
        if (screen >= 3)
            cout << "Remaining agents = " << remaining_agents <<
                 ", remaining time = " << T - ((fsec)(Time::now() - time)).count() << " seconds. " << endl
                 << "Agent " << agents[id].id << endl;
        agents[id].path = agents[id].path_planner->findPath(constraint_table);
        if (agents[id].path.empty()) break;
        neighbor.sum_of_costs += (int)agents[id].path.size() - 1;
        if (neighbor.sum_of_costs >= neighbor.old_sum_of_costs)
            break;
        remaining_agents--;
        path_table.insertPath(agents[id].id, agents[id].path);
        ++p;
    }
    if (remaining_agents == 0 && neighbor.sum_of_costs <= neighbor.old_sum_of_costs) // accept new paths
    {
        return true;
    }
    else // stick to old paths
    {
        if (p != shuffled_agents.end())
            num_of_failures++;
        auto p2 = shuffled_agents.begin();
        while (p2 != p)
        {
            int a = *p2;
            path_table.deletePath(agents[a].id, agents[a].path);
            ++p2;
        }
        if (!neighbor.old_paths.empty())
        {
            p2 = neighbor.agents.begin();
            for (int i = 0; i < (int)neighbor.agents.size(); i++)
            {
                int a = *p2;
                agents[a].path = neighbor.old_paths[i];
                path_table.insertPath(agents[a].id, agents[a].path);
                ++p2;
            }
            neighbor.sum_of_costs = neighbor.old_sum_of_costs;
        }
        return false;
    }
}

bool LNS::runPPS()
{
    auto shuffled_agents = neighbor.agents;
    std::random_shuffle(shuffled_agents.begin(), shuffled_agents.end());

    MAPF P = preparePIBTProblem(shuffled_agents);
    P.setTimestepLimit(pipp_option.timestepLimit);

    // seed for solver
    auto* MT_S = new std::mt19937(0);
    PPS solver(&P,MT_S);
    solver.setTimeLimit(time_limit);
//    solver.WarshallFloyd();
    bool result = solver.solve();
    if (result)
        updatePIBTResult(P.getA(),shuffled_agents);
    return result;
}

bool LNS::runPIBT()
{
    auto shuffled_agents = neighbor.agents;
     std::random_shuffle(shuffled_agents.begin(), shuffled_agents.end());

    MAPF P = preparePIBTProblem(shuffled_agents);

    // seed for solver
    auto MT_S = new std::mt19937(0);
    PIBT solver(&P,MT_S);
    solver.setTimeLimit(time_limit);
    bool result = solver.solve();
    if (result)
        updatePIBTResult(P.getA(),shuffled_agents);
    return result;
}

bool LNS::runWinPIBT()
{
    auto shuffled_agents = neighbor.agents;
    std::random_shuffle(shuffled_agents.begin(), shuffled_agents.end());

    MAPF P = preparePIBTProblem(shuffled_agents);
    P.setTimestepLimit(pipp_option.timestepLimit);

    // seed for solver
    auto MT_S = new std::mt19937(0);
    winPIBT solver(&P,pipp_option.windowSize,pipp_option.winPIBTSoft,MT_S);
    solver.setTimeLimit(time_limit);
    bool result = solver.solve();
    if (result)
        updatePIBTResult(P.getA(),shuffled_agents);
    return result;
}

MAPF LNS::preparePIBTProblem(vector<int>& shuffled_agents)
{
    // seed for problem and graph
    auto MT_PG = new std::mt19937(0);

//    Graph* G = new SimpleGrid(instance);
    Graph* G = new SimpleGrid(instance.getMapFile());

    std::vector<Task*> T;
    PIBT_Agents A;

    for (int i : shuffled_agents){
        assert(G->existNode(agents[i].path_planner->start_location));
        assert(G->existNode(agents[i].path_planner->goal_location));
        auto a = new PIBT_Agent(G->getNode( agents[i].path_planner->start_location));

//        PIBT_Agent* a = new PIBT_Agent(G->getNode( agents[i].path_planner.start_location));
        A.push_back(a);
        Task* tau = new Task(G->getNode( agents[i].path_planner->goal_location));


        T.push_back(tau);
        if(screen>=5){
            cout<<"Agent "<<i<<" start: " <<a->getNode()->getPos()<<" goal: "<<tau->getG().front()->getPos()<<endl;
        }
    }

    return MAPF(G, A, T, MT_PG);
}

void LNS::updatePIBTResult(const PIBT_Agents& A, vector<int>& shuffled_agents)
{
    int soc = 0;
    for (int i=0; i<A.size();i++){
        int a_id = shuffled_agents[i];

        agents[a_id].path.resize(A[i]->getHist().size());
        int last_goal_visit = 0;
        if(screen>=2)
            std::cout<<A[i]->logStr()<<std::endl;
        for (int n_index = 0; n_index < A[i]->getHist().size(); n_index++){
            auto n = A[i]->getHist()[n_index];
            agents[a_id].path[n_index] = PathEntry(n->v->getId());

            //record the last time agent reach the goal from a non-goal vertex.
            if(agents[a_id].path_planner->goal_location == n->v->getId()
                && n_index - 1>=0
                && agents[a_id].path_planner->goal_location !=  agents[a_id].path[n_index - 1].location)
                last_goal_visit = n_index;

        }
        //resize to last goal visit time
        agents[a_id].path.resize(last_goal_visit + 1);
        if(screen>=2)
            std::cout<<" Length: "<< agents[a_id].path.size() <<std::endl;
        if(screen>=5){
            cout <<"Agent "<<a_id<<":";
            for (auto loc : agents[a_id].path){
                cout <<loc.location<<",";
            }
            cout<<endl;
        }
        path_table.insertPath(agents[a_id].id, agents[a_id].path);
        soc += (int)agents[a_id].path.size()-1;
    }

    neighbor.sum_of_costs =soc;
}

void LNS::chooseDestroyHeuristicbyALNS()
{
    rouletteWheel();
    switch (selected_neighbor)
    {
        case 0 : destroy_strategy = RANDOMWALK; break;
        case 1 : destroy_strategy = INTERSECTION; break;
        case 2 : destroy_strategy = RANDOMAGENTS; break;
        default : cerr << "ERROR" << endl; exit(-1);
    }
}

bool LNS::generateNeighborByIntersection()
{
    if (intersections.empty())
    {
        for (int i = 0; i < instance.map_size; i++)
        {
            if (!instance.isObstacle(i) && instance.getDegree(i) > 2)
                intersections.push_back(i);
        }
    }

    set<int> neighbors_set;
    auto pt = intersections.begin();
    std::advance(pt, rand() % intersections.size());
    int location = *pt;
    path_table.get_agents(neighbors_set, neighbor_size, location);
    if (neighbors_set.size() < neighbor_size)
    {
        set<int> closed;
        closed.insert(location);
        std::queue<int> open;
        open.push(location);
        while (!open.empty() && (int) neighbors_set.size() < neighbor_size)
        {
            int curr = open.front();
            open.pop();
            for (auto next : instance.getNeighbors(curr))
            {
                if (closed.count(next) > 0)
                    continue;
                open.push(next);
                closed.insert(next);
                if (instance.getDegree(next) >= 3)
                {
                    path_table.get_agents(neighbors_set, neighbor_size, next);
                    if ((int) neighbors_set.size() == neighbor_size)
                        break;
                }
            }
        }
    }
    neighbor.agents.assign(neighbors_set.begin(), neighbors_set.end());
    if (neighbor.agents.size() > neighbor_size)
    {
        std::random_shuffle(neighbor.agents.begin(), neighbor.agents.end());
        neighbor.agents.resize(neighbor_size);
    }
    if (screen >= 2)
        cout << "Generate " << neighbor.agents.size() << " neighbors by intersection " << location << endl;
    return true;
}

bool LNS::generateNeighborByRandomWalk()
{
    if (neighbor_size >= (int)agents.size())
    {
        neighbor.agents.resize(agents.size());
        for (int i = 0; i < (int)agents.size(); i++)
            neighbor.agents[i] = i;
        return true;
    }

    int a = findMostDelayedAgent();
    if (a < 0)
        return false;
    
    set<int> neighbors_set;
    neighbors_set.insert(a);
    randomWalk(a, agents[a].path[0].location, 0, neighbors_set, neighbor_size, (int) agents[a].path.size() - 1);
    int count = 0;
    while (neighbors_set.size() < neighbor_size && count < 10)
    {
        int t = rand() % agents[a].path.size();
        randomWalk(a, agents[a].path[t].location, t, neighbors_set, neighbor_size, (int) agents[a].path.size() - 1);
        count++;
        // select the next agent randomly
        int idx = rand() % neighbors_set.size();
        int i = 0;
        for (auto n : neighbors_set)
        {
            if (i == idx)
            {
                a = i;
                break;
            }
            i++;
        }
    }
    if (neighbors_set.size() < 2)
        return false;
    neighbor.agents.assign(neighbors_set.begin(), neighbors_set.end());
    if (screen >= 2)
        cout << "Generate " << neighbor.agents.size() << " neighbors by random walks of agent " << a
             << "(" << agents[a].path_planner->my_heuristic[agents[a].path_planner->start_location]
             << "->" << agents[a].path.size() - 1 << ")" << endl;

    return true;
}

int LNS::findMostDelayedAgent()
{
    int a = -1;
    int max_delays = -1;
    for (int i = 0; i < agents.size(); i++)
    {
        if (tabu_list.find(i) != tabu_list.end())
            continue;
        int delays = agents[i].getNumOfDelays();
        if (max_delays < delays)
        {
            a = i;
            max_delays = delays;
        }
    }
    if (max_delays == 0)
    {
        tabu_list.clear();
        return -1;
    }
    tabu_list.insert(a);
    if (tabu_list.size() == agents.size())
        tabu_list.clear();
    return a;
}

int LNS::findRandomAgent() const
{
    int a = 0;
    int pt = rand() % (sum_of_costs - sum_of_distances) + 1;
    int sum = 0;
    for (; a < (int) agents.size(); a++)
    {
        sum += agents[a].getNumOfDelays();
        if (sum >= pt)
            break;
    }
    assert(sum >= pt);
    return a;
}


// a random walk with path that is shorter than upperbound and has conflicting with neighbor_size agents
void LNS::randomWalk(int agent_id, int start_location, int start_timestep,
                     set<int>& conflicting_agents, int neighbor_size, int upperbound)
{
    int loc = start_location;
    for (int t = start_timestep; t < upperbound; t++)
    {
        auto next_locs = instance.getNeighbors(loc);
        next_locs.push_back(loc);
        while (!next_locs.empty())
        {
            int step = rand() % next_locs.size();
            auto it = next_locs.begin();
            advance(it, step);
            int next_h_val = agents[agent_id].path_planner->my_heuristic[*it];
            if (t + 1 + next_h_val < upperbound) // move to this location
            {
                path_table.getConflictingAgents(agent_id, conflicting_agents, loc, *it, t + 1);
                loc = *it;
                break;
            }
            next_locs.erase(it);
        }
        if (next_locs.empty() || conflicting_agents.size() >= neighbor_size)
            break;
    }
}

// Calculate minimum, maximum, sum, and average of a vector
template <typename T>
void LNS::calcMinMaxSumAvgVec(const vector<T>& vec, vector<float>& features, size_t start_index) 
{
    const T* ptr = vec.data();
    size_t size = vec.size();

    float min_val = static_cast<float>(*ptr);
    float max_val = min_val;
    float sum_val = min_val;

    // Mild loop unrolling
    for (size_t i = 1; i < size; ++i) {
        float val = static_cast<float>(ptr[i]);
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
        sum_val += val;
    }

    features[start_index] = min_val;
    features[start_index + 1] = max_val;
    features[start_index + 2] = sum_val;
    features[start_index + 3] = sum_val / static_cast<float>(size);
}

/*
Calculate minimum, maximum, sum, and average w.r.t each group of 
minimum, maximum, sum, and average vectors respectively
*/
void LNS::calcMinMaxSumAvgVecVec(const vector<vector<int>>& vec_vec, vector<float>& features, size_t start_index) {
    // Clear vectors and prepare them for new data
    all_mins.clear();
    all_maxs.clear();
    all_sums.clear();
    all_avgs.clear();

    for (const auto& vec : vec_vec) 
    {
        const int* ptr = vec.data();
        const int* end = ptr + vec.size();

        float min_val = static_cast<float>(*ptr);
        float max_val = min_val;
        float sum_val = min_val;

        for (++ptr; ptr != end; ++ptr) 
        {
            float val = static_cast<float>(*ptr);
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
            sum_val += val;
        }

        all_mins.push_back(min_val);
        all_maxs.push_back(max_val);
        all_sums.push_back(sum_val);
        all_avgs.push_back(sum_val / vec.size());
    }

    // Aggregate min, max, sum, and average across all vectors
    for (size_t i = 0; i < 4; ++i) 
    {
        const auto& values = (i == 0) ? all_mins : (i == 1) ? all_maxs : (i == 2) ? all_sums : all_avgs;
        float aggregated_min = *std::min_element(values.begin(), values.end());
        float aggregated_max = *std::max_element(values.begin(), values.end());
        float aggregated_sum = std::accumulate(values.begin(), values.end(), 0.0f);
        float aggregated_avg = (values.empty()) ? 0 : aggregated_sum / values.size();

        features[start_index++] = aggregated_min;
        features[start_index++] = aggregated_max;
        features[start_index++] = aggregated_sum;
        features[start_index++] = aggregated_avg;
    }
}

// Normalize feature vector
void LNS::minMaxNormalize(vector<float>& inputVector) 
{
    const float* begin = inputVector.data();
    const float* end = begin + inputVector.size();

    float minVal = *begin;
    float maxVal = *begin;

    // Using branchless technique for min/max calculation
    for (const float* ptr = begin; ptr != end; ++ptr) 
    {
        minVal = std::min(minVal, *ptr);
        maxVal = std::max(maxVal, *ptr);
    }

    float diff = maxVal - minVal;

    // Normalize the vector
    for (float* ptr = inputVector.data(); ptr != end; ++ptr) 
    {
        *ptr = (*ptr - minVal) / diff;
    }
}

// calculate all features of a solution
void LNS::calculateSolutionFeatureMatrix(int seed) 
{
    // Clearing and reusing vectors
    countMap.clear();
    for (auto& path : agents_output_paths) path.clear();
    for (auto& heat : heat_values) heat.clear();

    for (const auto& agent : agents) 
    {
        auto path_planner = agent.path_planner;
        int start_loc = path_planner->start_location;
        int goal_loc = path_planner->goal_location;
        int agent_id = agent.id;

        // Precompute and assign values for each agent
        graph_dist[agent_id] = static_cast<float>(path_planner->my_heuristic[start_loc]);
        start_y[agent_id] = instance.getRowCoordinate(start_loc);
        start_x[agent_id] = instance.getColCoordinate(start_loc);
        goal_y[agent_id] = instance.getRowCoordinate(goal_loc);
        goal_x[agent_id] = instance.getColCoordinate(goal_loc);
        goal_degree[agent_id] = instance.getDegree(goal_loc);
        delay[agent_id] = static_cast<float>(agent.getNumOfDelays());
        ratio_delay_dist[agent_id] = delay[agent_id] / (graph_dist[agent_id] + 1.0f);

        // Process agents' paths
        agents_output_paths[agent_id].reserve(agent.path.size());
        for (const auto& state : agent.path) 
        {
            countMap[state.location]++;
            //if (state.location != agent.path.back().location) // do not count goal vertices of current agent
            //{
                agents_output_paths[agent_id].push_back(state.location);
                int deg = instance.getDegree(state.location);
                time_steps_vertex_degree[agent_id][deg - 1] += 1;
            //}
        }
    }

    // Process heat values
    for (size_t i = 0; i < agents_output_paths.size(); ++i) 
    {
        seen.clear();
        heat_values[i].reserve(agents_output_paths[i].size());
        for (int loc : agents_output_paths[i]) 
        {
            if (seen.insert(loc).second)
            {
                heat_values[i].push_back(countMap[loc]);
            }
        }
    }

    // Writing results directly into the feature_matrix[seed - 1]
    calcMinMaxSumAvgVec(graph_dist, feature_matrix[seed - 1], 0);     
    calcMinMaxSumAvgVec(start_x, feature_matrix[seed - 1], 4);       
    calcMinMaxSumAvgVec(start_y, feature_matrix[seed - 1], 8);       
    calcMinMaxSumAvgVec(goal_x, feature_matrix[seed - 1], 12);       
    calcMinMaxSumAvgVec(goal_y, feature_matrix[seed - 1], 16);       
    calcMinMaxSumAvgVec(goal_degree, feature_matrix[seed - 1], 20);  
    calcMinMaxSumAvgVec(delay, feature_matrix[seed - 1], 24);        
    calcMinMaxSumAvgVec(ratio_delay_dist, feature_matrix[seed - 1], 28); 
    calcMinMaxSumAvgVecVec(heat_values, feature_matrix[seed - 1], 32);   
    calcMinMaxSumAvgVecVec(time_steps_vertex_degree, feature_matrix[seed - 1], 48); 

    minMaxNormalize(feature_matrix[seed - 1]);
}


void LNS::validateSolution() const
{
    int sum = 0;
    for (const auto& a1_ : agents)
    {
        if (a1_.path.empty())
        {
            cerr << "No solution for agent " << a1_.id << endl;
            exit(-1);
        }
        else if (a1_.path_planner->start_location != a1_.path.front().location)
        {
            cerr << "The path of agent " << a1_.id << " starts from location " << a1_.path.front().location
                << ", which is different from its start location " << a1_.path_planner->start_location << endl;
            exit(-1);
        }
        else if (a1_.path_planner->goal_location != a1_.path.back().location)
        {
            cerr << "The path of agent " << a1_.id << " ends at location " << a1_.path.back().location
                 << ", which is different from its goal location " << a1_.path_planner->goal_location << endl;
            exit(-1);
        }
        for (int t = 1; t < (int) a1_.path.size(); t++ )
        {
            if (!instance.validMove(a1_.path[t - 1].location, a1_.path[t].location))
            {
                cerr << "The path of agent " << a1_.id << " jump from "
                     << a1_.path[t - 1].location << " to " << a1_.path[t].location
                     << " between timesteps " << t - 1 << " and " << t << endl;
                exit(-1);
            }
        }
        sum += (int) a1_.path.size() - 1;
        for (const auto  & a2_: agents)
        {
            if (a1_.id >= a2_.id || a2_.path.empty())
                continue;
            const auto & a1 = a1_.path.size() <= a2_.path.size()? a1_ : a2_;
            const auto & a2 = a1_.path.size() <= a2_.path.size()? a2_ : a1_;
            int t = 1;
            for (; t < (int) a1.path.size(); t++)
            {
                if (a1.path[t].location == a2.path[t].location) // vertex conflict
                {
                    cerr << "Find a vertex conflict between agents " << a1.id << " and " << a2.id <<
                            " at location " << a1.path[t].location << " at timestep " << t << endl;
                    exit(-1);
                }
                else if (a1.path[t].location == a2.path[t - 1].location &&
                        a1.path[t - 1].location == a2.path[t].location) // edge conflict
                {
                    cerr << "Find an edge conflict between agents " << a1.id << " and " << a2.id <<
                         " at edge (" << a1.path[t - 1].location << "," << a1.path[t].location <<
                         ") at timestep " << t << endl;
                    exit(-1);
                }
            }
            int target = a1.path.back().location;
            for (; t < (int) a2.path.size(); t++)
            {
                if (a2.path[t].location == target)  // target conflict
                {
                    cerr << "Find a target conflict where agent " << a2.id << " (of length " << a2.path.size() - 1<<
                         ") traverses agent " << a1.id << " (of length " << a1.path.size() - 1<<
                         ")'s target location " << target << " at timestep " << t << endl;
                    exit(-1);
                }
            }
        }
    }
    if (sum_of_costs != sum)
    {
        cerr << "The computed sum of costs " << sum_of_costs <<
             " is different from the sum of the paths in the solution " << sum << endl;
        exit(-1);
    }
}

void LNS::writeIterStatsToFile(const string & file_name) const
{
    if (init_lns != nullptr)
    {
        init_lns->writeIterStatsToFile(file_name + "-initLNS.csv");
    }
    if (iteration_stats.size() <= 1)
        return;
    string name = file_name;
    if (use_init_lns or num_of_iterations > 0)
        name += "-LNS.csv";
    else
        name += "-" + init_algo_name + ".csv";
    std::ofstream output;
    output.open(name);
    // header
    output << "num of agents," <<
           "sum of costs," <<
           "runtime," <<
           "cost lowerbound," <<
           "sum of distances," <<
           "MAPF algorithm" << endl;

    for (const auto &data : iteration_stats)
    {
        output << data.num_of_agents << "," <<
               data.sum_of_costs << "," <<
               data.runtime << "," <<
               max(sum_of_costs_lowerbound, sum_of_distances) << "," <<
               sum_of_distances << "," <<
               data.algorithm << endl;
    }
    output.close();
}

void LNS::writeResultToFile(const string & file_name) const
{
    if (init_lns != nullptr)
    {
        init_lns->writeResultToFile(file_name + "-initLNS.csv", sum_of_distances, preprocessing_time);
    }
    string name = file_name;
    if (use_init_lns or num_of_iterations > 0)
        name += "-LNS.csv";
    else
        name += "-" + init_algo_name + ".csv";
    std::ifstream infile(name);
    bool exist = infile.good();
    infile.close();
    if (!exist)
    {
        ofstream addHeads(name);
        addHeads << "runtime,solution cost,initial solution cost,lower bound,sum of distance," <<
                 "iterations," <<
                 "group size," <<
                 "runtime of initial solution," <<
                 "runtime of initial solutions generation," <<
                 "runtime of best initial solution prediction," <<
                 "restart times,area under curve," <<
                 "LL expanded nodes,LL generated,LL reopened,LL runs," <<
                 "preprocessing runtime,solver name,instance name" << endl;
        addHeads.close();
    }
    uint64_t num_LL_expanded = 0, num_LL_generated = 0, num_LL_reopened = 0, num_LL_runs = 0;
    for (auto & agent : agents)
    {
        agent.path_planner->reset();
        num_LL_expanded += agent.path_planner->accumulated_num_expanded;
        num_LL_generated += agent.path_planner->accumulated_num_generated;
        num_LL_reopened += agent.path_planner->accumulated_num_reopened;
        num_LL_runs += agent.path_planner->num_runs;
    }
    double auc = 0;
    if (!iteration_stats.empty())
    {
        auto prev = iteration_stats.begin();
        auto curr = prev;
        ++curr;
        while (curr != iteration_stats.end() && curr->runtime < time_limit)
        {
            auc += (prev->sum_of_costs - sum_of_distances) * (curr->runtime - prev->runtime);
            prev = curr;
            ++curr;
        }
        auc += (prev->sum_of_costs - sum_of_distances) * (time_limit - prev->runtime);
    }
    ofstream stats(name, std::ios::app);
    stats << runtime << "," << sum_of_costs << "," << initial_sum_of_costs << "," <<
          max(sum_of_distances, sum_of_costs_lowerbound) << "," << sum_of_distances << "," <<
          iteration_stats.size() << "," << average_group_size << "," <<
          initial_solution_runtime << "," << generate_init_solutions_time << "," << prediction_time << "," << 
          restart_times << "," << auc << "," << num_LL_expanded << "," << 
          num_LL_generated << "," << num_LL_reopened << "," << num_LL_runs << "," <<
          preprocessing_time << "," << getSolverName() << "," << instance.getInstanceName() << endl;
    stats.close();
}

void LNS::writeSolutionFeatureMatrixToFile(const vector<int>& target_ranks) const
{
    // Ensure the sizes of the provided vectors match
    if (target_ranks.size() != feature_matrix.size()) {
        cerr << "Mismatched sizes of input vectors." << endl;
        return;
    }

    // Create the complete path for the file
    string full_fpath = path_train_data + "/" + instance.getInstanceFileName();

    // Open the file for writing
    ofstream file(full_fpath);
    if (!file.is_open()) {
        cerr << "Failed to open file " << full_fpath << endl;
        return;
    }

    // Write the required format to the file
    for (size_t i = 0; i < target_ranks.size(); ++i) {
        file << target_ranks[i];
        for (size_t j = 0; j < feature_matrix[i].size(); ++j) {
            //file << " " << (j + 1) << ":" << feature_matrix[i][j];
            file << " " << feature_matrix[i][j];
        }
        file << "\n";
    }
}

void LNS::writePathsToFile(const string & file_name) const
{
    std::ofstream output;
    output.open(file_name);
    // header
    // output << agents.size() << endl;

    for (const auto &agent : agents)
    {
        output << "Agent " << agent.id << ":";
        for (const auto &state : agent.path)
            output << "(" << instance.getColCoordinate(state.location) << "," <<
                            instance.getRowCoordinate(state.location) << ")->";
        output << endl;
    }
    output.close();
}