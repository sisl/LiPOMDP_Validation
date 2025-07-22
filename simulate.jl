using Random
using POMDPs 
using POMDPTools
using LiPOMDPs
using MCTS
using DiscreteValueIteration
using POMCPOW 
using Distributions
using Parameters
using ARDESPOT
using Plots
using Plots.PlotMeasures
using Serialization
using Statistics



function simulate_lipomdp(disturbance::Matrix{Float64})
    rng = MersenneTwister(1)
    pomdp = initialize_lipomdp(disturbance, obj_weights=[0.25, 0.25, 1.0, 1.0, 0.25])
    up = LiBeliefUpdater(pomdp)
    b = initialize_belief(up)
    mdp = GenerativeBeliefMDP(pomdp, up)


    solver = POMCPOW.POMCPOWSolver(
        tree_queries=1000, estimate_value = 0, k_observation=4., 
        alpha_observation=0.06, max_depth=15, enable_action_pw=false,init_N=10  
    ) 

    # solver = DESPOTSolver(bounds=IndependentBounds(-1000.0, 1000.0, check_terminal = true))
    planner = solve(solver, pomdp);
    hr = HistoryRecorder(rng=rng, max_steps=pomdp.time_horizon);


    hist = simulate(hr, pomdp, planner, up, b);


    rewards = get_rewards(pomdp, hist);

    # r1: Domestic mining delay penalty
    # r2: amount of Li mined/imported so far
    # r3: CO2 emissions
    # r4: Demand fulfillment
    # r5: profit

    r1, r2, r3, r4, r5 = rewards[:r1], rewards[:r2], rewards[:r3], rewards[:r4], rewards[:r5]   # shape = horizon - 1
    
    return sum(r5)
end



