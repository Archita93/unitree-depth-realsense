// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <eigen3/Eigen/Dense>
#include <yaml-cpp/yaml.h>
#include "isaaclab/manager/observation_manager.h"
#include "isaaclab/manager/realsense_manager.h"
#include "isaaclab/manager/action_manager.h"
#include "isaaclab/assets/articulation/articulation.h"
#include "isaaclab/algorithms/algorithms.h"
#include <iostream>
#include <fstream>
#include <cstdlib>

namespace isaaclab
{

class ObservationManager;
class ActionManager;

class ManagerBasedRLEnv
{
public:
    // Constructor
    ManagerBasedRLEnv(YAML::Node cfg, std::shared_ptr<Articulation> robot_)
    :cfg(cfg), robot(std::move(robot_))
    {
        // Parse configuration
        this->step_dt = cfg["step_dt"].as<float>();
        robot->data.joint_ids_map = cfg["joint_ids_map"].as<std::vector<float>>();
        robot->data.joint_pos.resize(robot->data.joint_ids_map.size());
        robot->data.joint_vel.resize(robot->data.joint_ids_map.size());

        { // default joint positions
            auto default_joint_pos = cfg["default_joint_pos"].as<std::vector<float>>();
            robot->data.default_joint_pos = Eigen::VectorXf::Map(default_joint_pos.data(), default_joint_pos.size());
        }
        { // joint stiffness and damping
            robot->data.joint_stiffness = cfg["stiffness"].as<std::vector<float>>();
            robot->data.joint_damping = cfg["damping"].as<std::vector<float>>();
        }

        robot->update();

        // load managers
        action_manager = std::make_unique<ActionManager>(cfg["actions"], this);
        observation_manager = std::make_unique<ObservationManager>(cfg["observations"], this);
        realsense_manager = std::make_unique<RealsenseManager>(cfg["realsense"], this);

        std::string home = std::getenv("HOME") ? std::getenv("HOME") : "/tmp";
        log_file.open(home + "/action_log.csv");
        log_file << "step,cmd_x,cmd_y,cmd_z,";
        for (int i = 0; i < 12; i++) log_file << "raw_" << i << ",";
        for (int i = 0; i < 12; i++) log_file << "proc_" << i << (i < 11 ? "," : "\n");
    }

    void reset()
    {
        global_phase = 0;
        episode_length = 0;
        action_manager->reset();
        observation_manager->reset();
    }

    void step()
    {

        static auto last = std::chrono::steady_clock::now();
        auto now = std::chrono::steady_clock::now();
        double dt_ms = std::chrono::duration<double, std::milli>(now - last).count();
        last = now;
        std::cout << "actual dt: " << dt_ms << " ms (expected " << step_dt*1000 << ")\n";
        
        episode_length += 1;
        robot->update();
        last_depth = realsense_manager->compute();
        auto obs = observation_manager->compute();
        std::cout << "[DEBUG] obs.size() = " << obs.size() << std::endl;  
        auto action = alg->act(obs);
        action_manager->process_action(action);

        auto proc = action_manager->processed_actions();
        log_file << episode_length << ","
                  << obs[6] << "," << obs[7] << "," << obs[8] << ",";
        for (auto v : action) log_file << v << ",";              // raw network output
        for (size_t i = 0; i < proc.size(); i++)
            log_file << proc[i] << (i < proc.size()-1 ? "," : "\n");
        log_file.flush();
    }

    float step_dt;
    YAML::Node cfg;

    std::unique_ptr<ObservationManager> observation_manager;
    std::unique_ptr<RealsenseManager> realsense_manager;
    std::unique_ptr<ActionManager> action_manager;
    std::shared_ptr<Articulation> robot;
    std::unique_ptr<Algorithms> alg;
    long episode_length = 0;
    float global_phase = 0.0f;
    cv::Mat last_depth;
    std::ofstream log_file;
};

};
