import onnxruntime as ort
sess = ort.InferenceSession("/home/sq9025/repo2/logs/rsl_rl/unitree_go2_velocity/2026-07-21_14-02-49/exported/policy.onnx")
print(sess.get_inputs()[0].name, sess.get_inputs()[0].shape)