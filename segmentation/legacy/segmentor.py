import cv2
import numpy as np
from scipy import ndimage

class SegmentExtractor:
    def __init__(self, min_area=100, max_area=10000, aspect_ratio_range=(0.2, 5)):
        self.min_area = min_area
        self.max_area = max_area
        self.aspect_ratio_range = aspect_ratio_range

    def extract_segments(self, image, mask, threshold=0.5):
        binary_mask = (mask > threshold).astype(np.uint8)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((3,3), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        segments = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            aspect_ratio = w / h if h != 0 else 0

            if (self.min_area <= area <= self.max_area and 
                self.aspect_ratio_range[0] <= aspect_ratio <= self.aspect_ratio_range[1]):
                segment = image[y:y+h, x:x+w]
                rotated_segment, angle = self.correct_skew(segment)
                segments.append({
                    'segment': rotated_segment,
                    'bbox': (x, y, w, h),
                    'rotation_angle': angle
                })

        return segments



    ## perform preprocessing
    def correct_skew(self, image, delta=1, limit=5):
        def determine_score(arr, angle):
            data = ndimage.rotate(arr, angle, reshape=False, order=0)
            histogram = np.sum(data, axis=1)
            score = np.sum((histogram[1:] - histogram[:-1]) ** 2)
            return histogram, score

        scores = []
        angles = np.arange(-limit, limit + delta, delta)
        for angle in angles:
            histogram, score = determine_score(image, angle)
            scores.append(score)

        best_angle = angles[scores.index(max(scores))]
        
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        return rotated, best_angle

    def merge_nearby_segments(self, segments, distance_threshold=20):
        def calculate_distance(bbox1, bbox2):
            x1, y1, w1, h1 = bbox1
            x2, y2, w2, h2 = bbox2
            center1 = (x1 + w1/2, y1 + h1/2)
            center2 = (x2 + w2/2, y2 + h2/2)
            return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

        merged_segments = []
        used_indices = set()

        for i, seg1 in enumerate(segments):
            if i in used_indices:
                continue

            current_group = [seg1]
            used_indices.add(i)

            for j, seg2 in enumerate(segments[i+1:], start=i+1):
                if j in used_indices:
                    continue

                if calculate_distance(seg1['bbox'], seg2['bbox']) < distance_threshold:
                    current_group.append(seg2)
                    used_indices.add(j)

            if len(current_group) > 1:
                merged_segment = self.merge_segment_group(current_group)
                merged_segments.append(merged_segment)
            else:
                merged_segments.append(seg1)

        return merged_segments

    def merge_segment_group(self, segment_group):
        bboxes = [seg['bbox'] for seg in segment_group]
        x = min(bbox[0] for bbox in bboxes)
        y = min(bbox[1] for bbox in bboxes)
        max_x = max(bbox[0] + bbox[2] for bbox in bboxes)
        max_y = max(bbox[1] + bbox[3] for bbox in bboxes)
        w = max_x - x
        h = max_y - y

        merged_segment = np.zeros((h, w), dtype=np.uint8)
        for seg in segment_group:
            sx, sy, sw, sh = seg['bbox']
            merged_segment[sy-y:sy-y+sh, sx-x:sx-x+sw] = seg['segment']

        return {
            'segment': merged_segment,
            'bbox': (x, y, w, h),
            'rotation_angle': np.mean([seg['rotation_angle'] for seg in segment_group])
        }