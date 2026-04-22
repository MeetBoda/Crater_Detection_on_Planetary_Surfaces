import cv2
import numpy as np

def detect_craters(prob_map, threshold=0.5, min_radius=5, max_radius=200):
    """
    Detect craters from probability map using connected components.

    Args:
        prob_map : numpy array (H, W) → model output (0–1)
        threshold : segmentation threshold
        min_radius : minimum crater radius (pixels)
        max_radius : maximum crater radius (pixels)

    Returns:
        List of (x, y, r)
    """

    # Step 1: Threshold
    binary = (prob_map > threshold).astype(np.uint8)

    # Step 2: Connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    craters = []

    for i in range(1, num_labels):  # skip background
        x = int(centroids[i][0])
        y = int(centroids[i][1])

        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        # Step 3: Estimate radius
        r = (w + h) / 4.0   # IMPORTANT: from your notebook logic

        # Step 4: Filter by radius
        if min_radius <= r <= max_radius:
            craters.append((x, y, r))

    return craters