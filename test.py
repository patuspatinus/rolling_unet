import cv2, numpy as np
m = cv2.imread("/data/phucnd/Bubble_Glare_Baseline/Rolling-Unet/inputs/glare/Lung_cancer_lesions_unclean_glare/train/masks/0/0a0d1b27-4901-4cc1-b29c-d05af821c54f.png", cv2.IMREAD_GRAYSCALE)
print(m.min(), m.max(), np.unique(m)[:10])