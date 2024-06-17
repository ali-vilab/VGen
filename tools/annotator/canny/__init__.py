import cv2
import torch
import numpy as np
from tools.annotator.util import HWC3
# import gradio as gr

class CannyDetector:
    def __call__(self, img, low_threshold = None, high_threshold = None, random_threshold = True):

        ### GPT-4 suggestions
        # In the cv2.Canny() function, the low threshold and high threshold are used to determine the edges based on the gradient values in the image. 
        # There isn't a one-size-fits-all solution for these threshold values, as the optimal values depend on the specific image and the application. 
        # However, there are some general guidelines and empirical values you can use as a starting point:
        #   1. Ratio: A common recommendation is to use a ratio of 1:2 or 1:3 between the low threshold and the high threshold. 
        #       This means if your low threshold is 50, the high threshold should be around 100 or 150.
        #   2. Empirical values: As a starting point, you can use low threshold values in the range of 50-100 and high threshold values in the range of 100-200.
        #       You may need to fine-tune these values based on the specific image and desired edge detection results.
        #   3. Automatic threshold calculation: To automatically calculate the threshold values, you can use the median or mean value of the image's pixel intensities as the low threshold, 
        #       and the high threshold can be set as twice or three times the low threshold.

        ### Convert to numpy
        if isinstance(img, torch.Tensor): # (h, w, c)
            img = img.cpu().numpy()
            img_np = cv2.convertScaleAbs((img * 255.))
        elif isinstance(img, np.ndarray): # (h, w, c)
            img_np = img # we assume values are in the range from 0 to 255.
        else:
            assert False

        ### Select the threshold
        if (low_threshold is None) and (high_threshold is None):
            median_intensity = np.median(img_np)
            if random_threshold is False:
                low_threshold = int(max(0, (1 - 0.33) * median_intensity))
                high_threshold = int(min(255, (1 + 0.33) * median_intensity))
            else:
                random_canny = np.random.uniform(0.1, 0.4)
                # Might try other values
                low_threshold = int(max(0, (1 - random_canny) * median_intensity))
                high_threshold = 2 * low_threshold
                
        ### Detect canny edge
        canny_edge = cv2.Canny(img_np, low_threshold, high_threshold)
        ### Convert to 3 channels
        # canny_edge = HWC3(canny_edge)

        canny_condition = torch.from_numpy(canny_edge.copy()).unsqueeze(dim = -1).float().cuda() / 255.0
        # canny_condition = torch.stack([canny_condition for _ in range(num_samples)], dim=0)
        # canny_condition = einops.rearrange(canny_condition, 'h w c -> b c h w').clone()
        # return cv2.Canny(img, low_threshold, high_threshold)
        return canny_condition