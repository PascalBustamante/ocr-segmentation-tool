# import the necessary packages
import cv2
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def merge_components(stats, centroids, threshold):
    merged = []
    merger_data = {}
    n = len(stats)
    
    # Create a list of component indices
    components = list(range(n))
    
    while components:
        current = components.pop(0)
        merged_group = [current]
        
        i = 0
        while i < len(components):
            other = components[i]
            
            # Calculate distance between centroids
            dist = np.linalg.norm(centroids[current] - centroids[other])
            
            # Check if components are close enough to merge
            if dist < threshold:
                merged_group.append(other)
                components.pop(i)
            else:
                i += 1
        
        # Merge the group
        if len(merged_group) > 1:
            merger_data[current] = merged_group
        merged.append(merged_group)
    
    return merged, merger_data
 
def detect_roi_and_bboxes(image, min_area=100, max_area=10000000, aspect_ratio_range=(0.2, 5), merge_threshold=50): 
    ## Add a way to have no max area, would need to implement background filtering (simple)
    # Convert to grayscale if the image is color
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Apply thresholding
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # Apply morphological operations
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.erode(thresh, kernel, iterations=1)
    thresh = cv2.dilate(thresh, kernel, iterations=5)
    thresh = cv2.GaussianBlur(thresh, (5,5), 0)

    # Connected component analysis
    output = cv.connectedComponentsWithStats(thresh, 8, cv.CV_32S)
    (numLabels, labels, stats, centroids) = output



    ## New!!!!

    # Filter components based on area and aspect ratio
    valid_components = []
    for i in range(1, numLabels):  # Skip background
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        aspect_ratio = w / h if h != 0 else 0
        
        if (min_area <= area <= max_area and
            aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1]):
            valid_components.append(i)

    # Merge components
    merged_components, merger_data = merge_components(
        stats[valid_components], 
        centroids[valid_components], 
        merge_threshold
    )

    # Prepare lists to store ROIs and bounding boxes
    rois = []
    bboxes = []

     # Process merged components
    for group in merged_components:
        mask = np.zeros(thresh.shape, dtype=np.uint8)
        x_min, y_min = np.inf, np.inf
        x_max, y_max = 0, 0
        
        for idx in group:
            component_idx = valid_components[idx]
            component_mask = (labels == component_idx).astype(np.uint8) * 255
            mask = cv2.bitwise_or(mask, component_mask)
            
            x, y, w, h = stats[component_idx, cv2.CC_STAT_LEFT], stats[component_idx, cv2.CC_STAT_TOP], \
                         stats[component_idx, cv2.CC_STAT_WIDTH], stats[component_idx, cv2.CC_STAT_HEIGHT]
            x_min, y_min = min(x_min, x), min(y_min, y)
            x_max, y_max = max(x_max, x + w), max(y_max, y + h)
        
        rois.append(mask)
        bboxes.append((x_min, y_min, x_max - x_min, y_max - y_min))

    return rois, bboxes, thresh, merger_data

# Load the input image
image = cv2.imread(r"C:\Users\pasca\Data Science\Math Notes Model\OCR\test_data\math_example.png")

# Detect ROIs and bounding boxes
rois, bboxes, processed, merger_data = detect_roi_and_bboxes(image, merge_threshold=50)

# Visualize results
output = image.copy()
for bbox in bboxes:
    x, y, w, h = bbox
    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display results
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
ax1.set_title('Original Image')
ax2.imshow(processed, cmap='gray')
ax2.set_title('Processed Image')
ax3.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
ax3.set_title('Detected ROIs')
plt.show()

print(f"Number of ROIs detected: {len(rois)}")
print("Merger data:", merger_data)
