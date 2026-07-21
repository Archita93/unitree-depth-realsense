// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <eigen3/Eigen/Dense>
#include <yaml-cpp/yaml.h>
#include <unordered_set>
#include "isaaclab/manager/manager_term_cfg.h"
#include <iostream>
#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

namespace isaaclab
{


class RealsenseManager
{
public:
    RealsenseManager(YAML::Node cfg, ManagerBasedRLEnv* env)
    :cfg(cfg), env(env)
    {
      pipe_cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
      profile = pipe.start(pipe_cfg);

      // (Optional) downsample early to save work
      deci.set_option(RS2_OPTION_FILTER_MAGNITUDE, 2); // 640x480 -> 320x240
      depth_scale = profile.get_device().first<rs2::depth_sensor>().get_depth_scale();
    }

    void reset()
    {
    }

    cv::Mat compute()
    {
        rs2::frameset frames = pipe.wait_for_frames();
        rs2::depth_frame depth = frames.get_depth_frame();

        rs2::frame f = deci.process(depth);
        f = spatial.process(f);
        f = temporal.process(f);


        rs2::depth_frame df = f.as<rs2::depth_frame>();
        int w = df.get_width();
        int h = df.get_height();

        cv::Mat z16(h, w, CV_16U, (void*)df.get_data(), cv::Mat::AUTO_STEP);
        cv::Mat depth_m;
        z16.convertTo(depth_m, CV_32F, depth_scale);
        depth_m.setTo(5.0f, depth_m == 0.0f);
        cv::min(depth_m, 5.0f, depth_m);

        // cv::Mat depth_48;
        cv::Mat depth_resized;
        cv::resize(depth_m, depth_resized, cv::Size(128, 96), 0, 0, cv::INTER_AREA);

        return depth_resized;
    }

protected:
    rs2::pipeline pipe;
    rs2::config pipe_cfg;
    rs2::pipeline_profile profile;

    float depth_scale;

    rs2::decimation_filter deci;               // integer reduction
    rs2::spatial_filter spatial;
    rs2::temporal_filter temporal;

    YAML::Node cfg;
    ManagerBasedRLEnv* env;
private:
};

};
