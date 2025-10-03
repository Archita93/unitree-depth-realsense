import torch
from tensordict import TensorDict
import matplotlib.pyplot as plt

teacher_dict: TensorDict = TensorDict.load_memmap("./stored_transitions/")

vision = teacher_dict["vision_observations"]

# clip vision to 2m
vision = torch.clip(vision, 0.0, 2.0)
color = teacher_dict["color_vision_observations"]

for i in range(0, vision.shape[0], 10):
    for j in range(0, vision.shape[1], 100):
        img = vision[i, j]
        color_img = color[i, j] / 255.0
        fig, axs = plt.subplots(2)

        img = axs[0].imshow(img)
        img.set_cmap('hot')

        img = axs[1].imshow(color_img)
        img.set_cmap('hot')

        plt.axis('off')
        fig.savefig(f"./demo_images/{i}_{j}.png", bbox_inches="tight")
        plt.close()
