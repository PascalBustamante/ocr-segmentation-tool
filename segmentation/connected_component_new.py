import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from skimage.feature import hog
from skimage.transform import resize

class ConnectedComponent:
    def __init__(self, mask, stats, parent_shape):
        self.mask = mask
        self.stats = stats
        self.parent_width, self.parent_height = parent_shape
        self.parent_area = self.parent_width * self.parent_height
        self.x, self.y, self.width, self.height, self.area = stats
        self.black_pixels = np.sum(mask)
        self.perimeter = cv2.arcLength(self.get_contour(), True)
        self.centroid_x, self.centroid_y = self.calculate_centroid()
        self.num_holes = self.count_holes()

    def get_contour(self):
        contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours[0]

    def calculate_centroid(self):
        M = cv2.moments(self.mask)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = self.x + self.width // 2, self.y + self.height // 2
        return cX, cY

    def count_holes(self):
        contours, _ = cv2.findContours(self.mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        return len(contours) - 1

    @property
    def eccentricity(self):
        # Approximate eccentricity using aspect ratio
        return 1 - min(self.width, self.height) / max(self.width, self.height)




'''
    This function is not well implemented since the formation of the CC class,
    comonent merger should be kept track of in either the CC class or 
    a seperate data structure. 

    The CC class might be keeping track of the mergers by keeping track of parents,
    this might be an issue and possibly a bad way to init and store the mergers.
'''
def merge_components(stats, centroids, threshold):
    merged = []
    merger_data = {}
    n = len(stats)
    components = list(range(n))

    while components:
        current = components.pop(0)
        merged_group = [current]

        i = 0
        while i < len(components):
            other = components[i]
            dist = np.linalg.norm(centroids[current] - centroids[other])
            if dist < threshold:
                merged_group.append(other)
                components.pop(i)
            else:
                i += 1

        if len(merged_group) > 1:
            merger_data[current] = merged_group
        merged.append(merged_group)

    return merged, merger_data

def detect_roi_and_bboxes(image, min_area=100, max_area=10000000, aspect_ratio_range=(0.2, 5), merge_threshold=50):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.erode(thresh, kernel, iterations=1)
    thresh = cv2.dilate(thresh, kernel, iterations=5)
    thresh = cv2.GaussianBlur(thresh, (5,5), 0)

    output = cv2.connectedComponentsWithStats(thresh, 8, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output

    valid_components = []
    for i in range(1, numLabels):
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        aspect_ratio = w / h if h != 0 else 0

        if (min_area <= area <= max_area and
            aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1]):
            valid_components.append(i)

    merged_components, merger_data = merge_components(
        stats[valid_components], 
        centroids[valid_components], 
        merge_threshold
    )

    connected_components = []
    bboxes = []

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

        cc = ConnectedComponent(mask, (x_min, y_min, x_max - x_min, y_max - y_min, np.sum(mask)), image.shape[:2])
        connected_components.append(cc)
        bboxes.append((x_min, y_min, x_max - x_min, y_max - y_min))

    return connected_components, bboxes, thresh, merger_data

def extract_features(connected_components):
    features = []
    feature_names = ['aspect_ratio', 'area_ratio', 'density', 'perimeter_density', 
                     'eccentricity', 'rel_width', 'rel_height', 'num_holes', 
                     'rel_x', 'rel_y']

    for cc in connected_components:
        f1 = min(cc.width, cc.height) / max(cc.width, cc.height)
        f2 = cc.area / cc.parent_area
        f3 = cc.black_pixels / cc.area
        f4 = cc.perimeter / cc.black_pixels
        f5 = cc.eccentricity
        f6 = cc.width / cc.parent_width
        f7 = cc.height / cc.parent_height
        f8 = cc.num_holes
        f9 = cc.centroid_x / cc.parent_width
        f10 = cc.centroid_y / cc.parent_height

        hog_features = extract_hog_features(cc.mask)

        combined_features = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10] + list(hog_features)
        features.append(combined_features)

    return np.array(features), feature_names + [f'hog_{i}' for i in range(len(hog_features))]

def extract_hog_features(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    resized = resize(image, (64, 64), anti_aliasing=True)
    hog_features = hog(resized, orientations=orientations, 
                       pixels_per_cell=pixels_per_cell, 
                       cells_per_block=cells_per_block, 
                       block_norm='L2-Hys', 
                       feature_vector=True)
    return hog_features

def visualize_feature_distributions(features, labels, feature_names, max_features_per_plot=20):
    n_features = features.shape[1]
    n_plots = (n_features + max_features_per_plot - 1) // max_features_per_plot

    for plot_num in range(n_plots):
        start_idx = plot_num * max_features_per_plot
        end_idx = min((plot_num + 1) * max_features_per_plot, n_features)
        plot_features = features[:, start_idx:end_idx]
        plot_feature_names = feature_names[start_idx:end_idx]

        n_plot_features = plot_features.shape[1]
        n_rows = (n_plot_features + 1) // 2
        fig, axes = plt.subplots(n_rows, 2, figsize=(15, 3*n_rows))
        axes = axes.flatten()

        for i, (feature, name) in enumerate(zip(plot_features.T, plot_feature_names)):
            if i < len(axes):
                ax = axes[i]
                for label in np.unique(labels):
                    ax.hist(feature[labels == label], bins=50, alpha=0.5, label=f'Class {label}')
                ax.set_title(name)
                ax.legend()

        # Remove any unused subplots
        for i in range(n_plot_features, len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        plt.show()

def visualize_pairwise_features(features, labels, feature_names):
    n_features = min(5, features.shape[1])
    fig, axes = plt.subplots(n_features, n_features, figsize=(15, 15))

    for i in range(n_features):
        for j in range(n_features):
            ax = axes[i, j]
            if i != j:
                for label in np.unique(labels):
                    ax.scatter(features[labels == label, j], features[labels == label, i], 
                               alpha=0.5, label=f'Class {label}')
                ax.set_xlabel(feature_names[j])
                ax.set_ylabel(feature_names[i])
            else:
                ax.hist(features[:, i], bins=50)
                ax.set_title(feature_names[i])

            if i == 0 and j == n_features-1:
                ax.legend()

    plt.tight_layout()
    plt.show()

def visualize_pca(features, labels):
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features)

    plt.figure(figsize=(10, 8))
    for label in np.unique(labels):
        plt.scatter(features_pca[labels == label, 0], features_pca[labels == label, 1], 
                    alpha=0.5, label=f'Class {label}')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.legend()
    plt.title('PCA of Features')
    plt.show()

def visualize_tsne(features, labels):
    tsne = TSNE(n_components=2, random_state=42)
    features_tsne = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    for label in np.unique(labels):
        plt.scatter(features_tsne[labels == label, 0], features_tsne[labels == label, 1], 
                    alpha=0.5, label=f'Class {label}')
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    plt.legend()
    plt.title('t-SNE of Features')
    plt.show()

if __name__ == "__main__":
    # Load the input image
    image_path = r"C:\Users\pasca\Data Science\Math Notes Model\OCR\test_data\math_example.png"
    image = cv2.imread(image_path)

    # Detect ROIs and bounding boxes
    connected_components, bboxes, processed, merger_data = detect_roi_and_bboxes(image, merge_threshold=50)

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

    print(f"Number of ROIs detected: {len(connected_components)}")
    print("Merger data:", merger_data)

    # Extract features
    features, feature_names = extract_features(connected_components)

    # For demonstration purposes, let's assume all components are normal text (label 0)
    # In a real scenario, you would need to label these components manually or use a classifier
    labels = np.zeros(len(connected_components))

    # Visualize feature distributions
    #visualize_feature_distributions(features, labels, feature_names)

    # Visualize pairwise feature relationships
    visualize_pairwise_features(features, labels, feature_names)

    # Visualize PCA
    visualize_pca(features, labels)

    # Visualize t-SNE
    visualize_tsne(features, labels)