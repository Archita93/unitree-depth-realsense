// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "onnxruntime_cxx_api.h"
#include <mutex>

namespace isaaclab
{

class Algorithms
{
public:
    virtual std::vector<float> act(std::vector<float> obs) = 0;
    virtual std::vector<float> act(std::vector<float> obs, const cv::Mat& image) = 0;
    
    std::vector<float> get_action()
    {
        std::lock_guard<std::mutex> lock(act_mtx_);
        return action;
    }
    
    std::vector<float> action;
protected:
    std::mutex act_mtx_;
};

class OrtRunner : public Algorithms
{
public:
    OrtRunner(std::string model_path, bool use_vision = false)
    {
        // Init Model
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "onnx_model");
        session_options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
        session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);

        is_recurrent = session->GetInputCount() > 1;

        Ort::TypeInfo input_type = session->GetInputTypeInfo(0);
        input_shape = input_type.GetTensorTypeAndShapeInfo().GetShape();

        if (is_recurrent)
        {
            Ort::TypeInfo hidden_type = session->GetInputTypeInfo(1);
            hidden_shape = hidden_type.GetTensorTypeAndShapeInfo().GetShape();
            hidden_state.resize(hidden_shape[1], 0);
        }

        Ort::TypeInfo output_type = session->GetOutputTypeInfo(0);
        output_shape = output_type.GetTensorTypeAndShapeInfo().GetShape();
        action.resize(output_shape[1]);
    }

    std::vector<float> act(std::vector<float> obs)
    {
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, obs.data(), obs.size(), input_shape.data(), input_shape.size());
        
        
        if (is_recurrent)
        {
            auto hidden_state_tensor = Ort::Value::CreateTensor<float>(memory_info, hidden_state.data(), hidden_state.size(), hidden_shape.data(), hidden_shape.size());
            const Ort::Value inputs[] = {
            std::move(input_tensor),
            std::move(hidden_state_tensor)
            };
            auto output_tensor = session->Run(Ort::RunOptions{nullptr}, input_names.data(), inputs, 2, output_names.data(), 2);
            auto floatarr = output_tensor.front().GetTensorMutableData<float>();
            auto hidden_state_out = output_tensor.back().GetTensorMutableData<float>();

            std::lock_guard<std::mutex> lock(act_mtx_);
            std::memcpy(action.data(), floatarr, output_shape[1] * sizeof(float));
            std::memcpy(hidden_state.data(), hidden_state_out, hidden_shape[1] * sizeof(float));
        }

        else{

            const Ort::Value inputs[] = {
            std::move(input_tensor)
            };
            auto output_tensor = session->Run(Ort::RunOptions{nullptr}, input_names.data(), inputs, 1, output_names.data(), 1);
            auto floatarr = output_tensor.front().GetTensorMutableData<float>();

            std::lock_guard<std::mutex> lock(act_mtx_);
            std::memcpy(action.data(), floatarr, output_shape[1] * sizeof(float));
        }
        
            return action;
    }

    std::vector<float> act(std::vector<float> obs, const cv::Mat& depth)
    {
        const cv::Mat depth_cont = depth.isContinuous() ? depth : depth.clone();
        const int64_t H = depth_cont.rows;
        const int64_t W = depth_cont.cols;

        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, obs.data(), obs.size(), input_shape.data(), input_shape.size());
        // auto hidden_state_tensor = Ort::Value::CreateTensor<float>(memory_info, hidden_state.data(), hidden_state.size(), hidden_shape.data(), hidden_shape.size());

        int64_t img_shape[4] = {1, H, W, 1};
        auto image_tensor = Ort::Value::CreateTensor<float>(
          memory_info,
          const_cast<float*>(depth_cont.ptr<float>()),
          static_cast<size_t>(H) * static_cast<size_t>(W),
          img_shape, 4
        );

        if (is_recurrent)
        {
            auto hidden_state_tensor = Ort::Value::CreateTensor<float>(memory_info, hidden_state.data(), hidden_state.size(), hidden_shape.data(), hidden_shape.size());
            const Ort::Value inputs[] = {
              std::move(input_tensor),
              std::move(hidden_state_tensor),
              std::move(image_tensor),
            };

            auto output_tensor = session->Run(
              Ort::RunOptions{nullptr},
              image_input_names.data(),
              inputs,
              3,
              output_names.data(),
              2
            );

            auto floatarr = output_tensor.front().GetTensorMutableData<float>();
            auto hidden_state_out = output_tensor.back().GetTensorMutableData<float>();

            std::lock_guard<std::mutex> lock(act_mtx_);
            std::memcpy(action.data(), floatarr, output_shape[1] * sizeof(float));
            std::memcpy(hidden_state.data(), hidden_state_out, hidden_shape[1] * sizeof(float));
        }
        else
        {
          const Ort::Value inputs[] = {
            std::move(input_tensor),
            std::move(image_tensor),
          };

          auto output_tensor = session->Run(
            Ort::RunOptions{nullptr},
            image_input_names.data(),
            inputs,
            2,
            output_names.data(),
            1
          );

          auto floatarr = output_tensor.front().GetTensorMutableData<float>();

          std::lock_guard<std::mutex> lock(act_mtx_);
          std::memcpy(action.data(), floatarr, output_shape[1] * sizeof(float));
        }
        return action;
    }

private:
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;
    bool is_recurrent = false;
    const std::vector<const char*> input_names = {"obs", "hidden_in"};
    const std::vector<const char*> image_input_names = {"obs", "hidden_in", "image"};
    const std::vector<const char*> output_names = {"actions", "hidden_out"};

    std::vector<int64_t> input_shape;
    std::vector<int64_t> output_shape;
    std::vector<int64_t> hidden_shape;
    std::vector<int64_t> image_shape;

    std::vector<float> hidden_state;
};

};
