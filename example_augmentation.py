import stainAugmentation
import cv2
import matplotlib.pyplot as plt

original_image = cv2.cvtColor(cv2.imread('imgs/example1.png'), cv2.COLOR_BGR2RGB)

fig, ax = plt.subplots(4,4, figsize=(10,10))
for i in range(4):
    for j in range(4):
        augmented_image = stainAugmentation.randomHEStaining(original_image)
        ax[i,j].imshow(augmented_image)
        ax[j, i].xaxis.set_major_locator(plt.NullLocator())
        ax[j, i].yaxis.set_major_locator(plt.NullLocator())
plt.tight_layout()
plt.savefig('imgs/example_augmentations.png')
