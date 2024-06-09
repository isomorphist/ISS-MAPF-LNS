#pragma once
#include "BasicLNS.h"
#include "InitLNS.h"

//pibt related
#include "simplegrid.h"
#include "pibt_agent.h"
#include "problem.h"
#include "mapf.h"
#include "pibt.h"
#include "pps.h"
#include "winpibt.h"

enum destroy_heuristic { RANDOMAGENTS, RANDOMWALK, INTERSECTION, DESTORY_COUNT };

// TODO: adaptively change the neighbor size, that is,
// increase it if no progress is made for a while
// decrease it if replanning fails to find any solutions for several times

class LNS : public BasicLNS
{
public:
    vector<Agent> agents;
    double preprocessing_time = 0;
    double initial_solution_runtime = 0;
    double generate_init_solutions_time = 0;
    int initial_sum_of_costs = -1;
    int sum_of_costs_lowerbound = -1;
    int sum_of_distances = -1;
    int restart_times = 0;
    int num_agents;
    float num_agents_f;
    string path_results;
    string path_train_data;
    int num_initial_solutions;
    int num_lns_runs;
    high_resolution_clock::time_point start_time_feat_calc;
    vector<std::vector<float>> feature_matrix;
    vector<pair<int, double>> target_matrix_per_seed;
    double prediction_time = 0;
    vector<float> all_mins, all_maxs, all_sums, all_avgs;
    vector<int> start_x, start_y, goal_x, goal_y, goal_degree;
    vector<float> graph_dist, delay, ratio_delay_dist;
    vector<vector<int>> time_steps_vertex_degree;
    vector<vector<int>> agents_output_paths;
    vector<vector<int>> heat_values;
    unordered_map<int, int> countMap;
    unordered_set<int> seen;

    LNS(const Instance& instance, double time_limit,
        const string & init_algo_name, const string & replan_algo_name, const string & destory_name,
        int neighbor_size, int num_of_iterations, bool init_lns, const string & init_destory_name, bool use_sipp, const string& path_results,
        const string& path_train_data, int num_initial_solutions,
        int num_lns_runs, int screen, PIBTPPS_option pipp_option);
    ~LNS()
    {
        delete init_lns;
    }
     
    bool generateInitialSolution(bool use_feature_matrix, int seed);
    void generateTrainData();
    bool getInitialSolution();
    bool run();
    bool runML();
    void validateSolution() const;
    void writeIterStatsToFile(const string & file_name) const;
    void writeResultToFile(const string & file_name) const;
    void writePathsToFile(const string &file_name) const;
    string getSolverName() const override { return "LNS(" + init_algo_name + ";" + replan_algo_name + ")"; }


private:
    InitLNS* init_lns = nullptr;
    string init_algo_name;
    string replan_algo_name;
    bool use_init_lns; // use LNS to find initial solutions
    destroy_heuristic destroy_strategy = RANDOMWALK;
    int num_of_iterations;
    string init_destory_name;
    PIBTPPS_option pipp_option;


    PathTable path_table; // 1. stores the paths of all agents in a time-space table;
    // 2. avoid making copies of this variable as much as possible.
    unordered_set<int> tabu_list; // used by randomwalk strategy
    list<int> intersections;

    void runALNS(int run);
    void initializeVariablesForML();
    void initializeAgentRelatedVectors();
    void resetComplexMembers();

    bool runEECBS();
    bool runCBS();
    bool runPP();
    bool runPIBT();
    bool runPPS();
    bool runWinPIBT();


    MAPF preparePIBTProblem(vector<int>& shuffled_agents);
    void updatePIBTResult(const PIBT_Agents& A, vector<int>& shuffled_agents);

    void chooseDestroyHeuristicbyALNS();

    bool generateNeighborByRandomWalk();
    bool generateNeighborByIntersection();

    int findMostDelayedAgent();
    int findRandomAgent() const;
    void randomWalk(int agent_id, int start_location, int start_timestep,
                    set<int>& neighbor, int neighbor_size, int upperbound);

    template <typename T>
    void calcMinMaxSumAvgVec(const vector<T>& vec, vector<float>& features, size_t start_index);

    void calculateMedians(pair<double, double>& target_matrix_medians);
    void calculateRanks(const vector<pair<double, double>>& target_matrix_medians, vector<int>& target_ranks);
    int getBestRankedIndex(const string& model_file);
    void calcMinMaxSumAvgVecVec(const vector<vector<int>>& vec_vec, vector<float>& features, size_t start_index);
    void add_values(int start_idx, int added_num, vector<float>& vec, const vector<float>& added_values);
    void minMaxNormalize(vector<float>& inputVector);
    void calculateSolutionFeatureMatrix(int seed);
    void writeSolutionFeatureMatrixToFile(const vector<int> & target_ranks) const;

};