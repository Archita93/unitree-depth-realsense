#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

files = ["ppo_log.txt", "fasttd3_no_action_scaling_log.txt", "fasttd3_action_scaling_log.txt"]
ranges = {"ppo": None, "fasttd3_no_action_scaling": 0.25, "fasttd3_action_scaling": 1.0}
colors = {
    "ppo": (1.0, 0.0, 0.0),
    "fasttd3_no_action_scaling": (0.0, 1.0, 0.0),
    "fasttd3_action_scaling": (0.0, 0.0, 1.0),
}
NUM_MOTORS = 12
DEFAULT_JOINT_POS = [0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1.0, 1.0, -1.5, -1.5, -1.5, -1.5]

data = {}

for file in files:
    with open("logs/" + file, "r") as f:
        elements = []
        current = []
        first = True
        while line := f.readline().strip():
            if first:
                if line == "---":
                    first = False
                continue

            if line == "---":
                elements.append(current)
                current = []
                continue

            motor_str = "motor: "
            if motor_str in line:
                motor_cmd = float(line[len(motor_str):])
                current.append(np.array(motor_cmd))
    data[file[:-len("_log.txt")]] = np.stack(elements, axis=0)

shortest = min([len(v) for v in data.values()])
start = int(shortest * 0.1)
end = int(shortest * 0.9)
data = {k: v[start:end] for k, v in data.items()}

dictionary = {
    "ppo": r"PPO ($\omega = 0.25$)",
    "fasttd3_action_scaling": r"FastTD3 ($\omega = 1.0$)",
    "fasttd3_no_action_scaling": r"FastTD3 ($\omega = 0.25$)",
}

def plot_motor(data, motor_id: int):
    fig, axs = plt.subplots(len(data), squeeze=True)
    max_ = -float("inf")
    min_ = float("inf")
    for motor_cmds in data.values():
        x = motor_cmds[:, motor_id]
        cmd_min = np.min(x)
        cmd_max = np.max(x)
        if cmd_min < min_:
            min_ = cmd_min
        if cmd_max > max_:
            max_ = cmd_max

    for i, (label, motor_cmds) in enumerate(data.items()):
        x = motor_cmds[:, motor_id]


        axs[i].hist(x, bins=25)

        title = dictionary[label]

        axs[i].set_title(title)
        range_ = ranges[label]
        if range_ is not None:
            cmd_range = (
                DEFAULT_JOINT_POS[motor_id] - range_,
                DEFAULT_JOINT_POS[motor_id] + range_,
            )

            axs[i].vlines(cmd_range, 0.0, 50000.0, linestyles="dotted", color="r")

        axs[i].set_xlim((min_ - 1.0, max_ + 1.0))
        axs[i].set_ylim(0, 60_000)
    fig.suptitle(f"Motor {motor_id}")
    return fig, axs

for i in range(NUM_MOTORS):
    fig, ax = plot_motor(data, i)

    fig.tight_layout()
    fig.savefig(f"imgs/motor_{i}_cmd_histogram.jpg")

