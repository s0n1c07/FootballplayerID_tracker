# -*- coding: utf-8 -*-
"""
YOLOv11 with RAFT-Enhanced Multi-Object Tracker (Advanced) - ResNet18 Only
"""

import torch
import cv2
import numpy as np
from ultralytics import YOLO
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from torchvision.transforms import functional as F
import torch.nn.functional as torch_F
from scipy.optimize import linear_sum_assignment as hungarian
from torchvision import transforms as tv_transforms
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights

# Check if CUDA is available and set the device
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA device'}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# RAFT Model Setup
try:
    RAFT_MODEL = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=True).to(device)
    RAFT_MODEL = RAFT_MODEL.eval()
    print("RAFT large model loaded successfully.")
except Exception as e:
    print(f"Error loading RAFT large model: {e}")
    print("Falling back to RAFT small model.")
    from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
    RAFT_MODEL = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=True).to(device)
    RAFT_MODEL = RAFT_MODEL.eval()
    print("RAFT small model loaded as a fallback.")

# Re-ID Embedder Setup with ResNet18 only
print("Loading Re-ID embedder (ResNet18)...")
REID_EMBEDDER = resnet18(weights=ResNet18_Weights.DEFAULT).to(device)
REID_EMBEDDER = torch.nn.Sequential(*list(REID_EMBEDDER.children())[:-1])  # Remove the final FC layer
REID_EMBEDDER.eval()
print("Re-ID embedder (ResNet18) loaded successfully.")
    
transform = tv_transforms.Compose([
    tv_transforms.Resize((256, 128)),  # Standard Re-ID input size
    tv_transforms.ToTensor(),
    tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
print("Transforms defined.")

MODEL_PATH = 'best.pt'
VIDEO_PATH = '15sec_input_720p.mp4'
OUTPUT_VIDEO_PATH = 'output_raft_advanced_reid_tracking.mp4'
print(f"Loading YOLOv11 detection model from: {MODEL_PATH}")
YOLO_MODEL = YOLO(MODEL_PATH)
YOLO_MODEL.to(device)
YOLO_MODEL.eval()

if YOLO_MODEL:
    print("YOLOv11 detection model loaded successfully.")
else:
    print("Error: Could not load YOLOv11 detection model.")
    exit()

model_class_names = YOLO_MODEL.names
print("\n--- Model Class Names ---")
print(model_class_names)
print("---------------------------\n")

TARGET_CLASSES = {
    "ball": 0,
    "goalkeeper": 1,
    "player": 2,
    "referee": 3
}

CLASS_DISPLAY_INFO = {
    TARGET_CLASSES["ball"]:       {"name": "Football", "color": (0, 165, 255)},
    TARGET_CLASSES["goalkeeper"]: {"name": "Goalkeeper", "color": (0, 200, 200)},
    TARGET_CLASSES["player"]:     {"name": "Player", "color": (0, 255, 0)},
    TARGET_CLASSES["referee"]:    {"name": "Referee", "color": (255, 0, 0)}
}
CLASSES_TO_PROCESS = [class_id for class_id in TARGET_CLASSES.values() if class_id is not None]

# Video Setup
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Error: Could not open video file {VIDEO_PATH}")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Input video FPS: {fps}, Dimensions: {width}x{height}, Total frames: {total_frames}")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

# Tracking Logic with RAFT-Enhanced Tracker
class Track:
    def __init__(self, track_id, bbox, class_id):
        self.track_id = track_id
        self.bbox = bbox
        self.class_id = class_id
        self.age = 0
        self.miss_count = 0
        self.max_age = 90
        self.kalman_filter = self.init_kalman(bbox)
        self.appearance_embedding = None

    def init_kalman(self, bbox):
        kalman = cv2.KalmanFilter(8, 4)
        kalman.measurementMatrix = np.array([[1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0], [0,0,1,0,0,0,0,0], [0,0,0,1,0,0,0,0]], np.float32)
        kalman.transitionMatrix = np.array([[1,0,0,0,1,0,0,0], [0,1,0,0,0,1,0,0], [0,0,1,0,0,0,1,0], [0,0,0,1,0,0,0,1],
                                            [0,0,0,0,1,0,0,0], [0,0,0,0,0,1,0,0], [0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,1]], np.float32)
        kalman.processNoiseCov = np.eye(8, dtype=np.float32) * 1e-5
        kalman.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-3
        x, y, w, h = (bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2, bbox[2]-bbox[0], bbox[3]-bbox[1]
        kalman.statePost = np.array([[x], [y], [w], [h], [0], [0], [0], [0]], np.float32)
        return kalman

    def predict(self, flow_map=None):
        self.kalman_filter.predict()
        predicted_bbox = self.kalman_filter.statePost[:4].flatten()
        x, y, w, h = predicted_bbox

        if flow_map is not None:
            x_center = int(x)
            y_center = int(y)
            if 0 <= y_center < flow_map.shape[0] and 0 <= x_center < flow_map.shape[1]:
                dx, dy = flow_map[y_center, x_center]
                x += dx
                y += dy

        return [x - w/2, y - h/2, x + w/2, y + h/2]

    def update(self, new_bbox, new_embedding):
        x, y, w, h = (new_bbox[0] + new_bbox[2])/2, (new_bbox[1] + new_bbox[3])/2, new_bbox[2]-new_bbox[0], new_bbox[3]-new_bbox[1]
        measurement = np.array([[x], [y], [w], [h]], np.float32)
        self.kalman_filter.correct(measurement)
        self.bbox = new_bbox
        self.miss_count = 0
        self.appearance_embedding = new_embedding

def get_iou(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    x1_p, y1_p, x2_p, y2_p = bbox2

    x_left = max(x1, x1_p)
    y_top = max(y1, y1_p)
    x_right = min(x2, x2_p)
    y_bottom = min(y2, y2_p)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bbox1_area = (x2 - x1) * (y2 - y1)
    bbox2_area = (x2_p - x1_p) * (y2_p - y1_p)
    union_area = float(bbox1_area + bbox2_area - intersection_area)
    return intersection_area / union_area if union_area > 0 else 0.0

def get_cosine_distance(embed1, embed2):
    norm_embed1 = np.linalg.norm(embed1)
    norm_embed2 = np.linalg.norm(embed2)
    if norm_embed1 == 0 or norm_embed2 == 0:
        return 1.0
    return 1 - np.dot(embed1, embed2) / (norm_embed1 * norm_embed2)

def get_embedding_from_bbox(bbox, frame_rgb, embedder, transforms):
    x1, y1, x2, y2 = [int(i) for i in bbox]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame_rgb.shape[1], x2), min(frame_rgb.shape[0], y2)

    if x2 <= x1 or y2 <= y1:
        return np.zeros(512)

    crop = frame_rgb[y1:y2, x1:x2]
    crop_pil = Image.fromarray(crop)

    crop_tensor = transforms(crop_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = embedder(crop_tensor).flatten()
    return embedding.cpu().numpy()


tracked_objects = []
next_id = 0
prev_frame_rgb_tensor = None
frame_count = 0
print("Processing video frames...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    curr_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    curr_frame_tensor = F.to_tensor(curr_frame_rgb).unsqueeze(0).to(device)

    detections = YOLO_MODEL.predict(
        source=frame,
        conf=0.4,
        iou=0.5,
        imgsz=640,
        device=device,
        verbose=False
    )[0]

    current_detections = []
    if detections.boxes.xyxy.numel() > 0:
        for box, class_id, conf in zip(detections.boxes.xyxy.cpu().numpy(),
                                       detections.boxes.cls.int().cpu().numpy(),
                                       detections.boxes.conf.float().cpu().numpy()):
            if class_id in CLASSES_TO_PROCESS:
                current_detections.append({'bbox': box, 'class_id': class_id, 'conf': conf})

    current_embeddings = [get_embedding_from_bbox(det['bbox'], curr_frame_rgb, REID_EMBEDDER, transform) for det in current_detections]

    if not tracked_objects:
        for det, embed in zip(current_detections, current_embeddings):
            new_track = Track(next_id, det['bbox'], det['class_id'])
            new_track.appearance_embedding = embed
            tracked_objects.append(new_track)
            next_id += 1
    else:
        flow_vectors_np = None
        if prev_frame_rgb_tensor is not None:
            with torch.no_grad():
                h_resized = int(curr_frame_tensor.shape[2] / 8) * 8
                w_resized = int(curr_frame_tensor.shape[3] / 8) * 8
                prev_resized = F.resize(prev_frame_rgb_tensor, size=(h_resized, w_resized))
                curr_resized = F.resize(curr_frame_tensor, size=(h_resized, w_resized))

                flow_vectors = RAFT_MODEL(prev_resized, curr_resized)[-1]
                flow_vectors = F.resize(flow_vectors, size=(height, width))
            flow_vectors_np = flow_vectors[0].cpu().numpy().transpose(1, 2, 0)

        predicted_bboxes = [t.predict(flow_map=flow_vectors_np) for t in tracked_objects]

        iou_matrix = np.zeros((len(predicted_bboxes), len(current_detections)))
        for i, pred_bbox in enumerate(predicted_bboxes):
            for j, det in enumerate(current_detections):
                if tracked_objects[i].class_id == det['class_id']:
                    iou_matrix[i, j] = get_iou(pred_bbox, det['bbox'])

        track_indices, detection_indices = hungarian(-iou_matrix)

        matches = []
        for track_idx, det_idx in zip(track_indices, detection_indices):
            if iou_matrix[track_idx, det_idx] >= 0.5:
                matches.append((track_idx, det_idx))

        matched_tracks_indices = set(m[0] for m in matches)
        matched_detections_indices = set(m[1] for m in matches)

        unmatched_tracks_indices = [i for i in range(len(tracked_objects)) if i not in matched_tracks_indices]
        unmatched_detections_indices = [i for i in range(len(current_detections)) if i not in matched_detections_indices]

        if unmatched_tracks_indices and unmatched_detections_indices:
            reid_cost_matrix = np.zeros((len(unmatched_tracks_indices), len(unmatched_detections_indices)))
            for i, track_idx in enumerate(unmatched_tracks_indices):
                track_embedding = tracked_objects[track_idx].appearance_embedding
                for j, det_idx in enumerate(unmatched_detections_indices):
                    det_embedding = current_embeddings[det_idx]
                    if (track_embedding is not None and det_embedding.shape == track_embedding.shape and
                        tracked_objects[track_idx].class_id == current_detections[det_idx]['class_id']):

                        reid_cost_matrix[i, j] = get_cosine_distance(track_embedding, det_embedding)
                    else:
                        reid_cost_matrix[i, j] = 1.0

            reid_track_map_idx, reid_det_map_idx = hungarian(reid_cost_matrix)
            for i_map, j_map in zip(reid_track_map_idx, reid_det_map_idx):
                if reid_cost_matrix[i_map, j_map] < 0.5:
                    track_idx = unmatched_tracks_indices[i_map]
                    det_idx = unmatched_detections_indices[j_map]
                    matches.append((track_idx, det_idx))
                    matched_tracks_indices.add(track_idx)
                    matched_detections_indices.add(det_idx)

        updated_tracks = []
        for track_idx, det_idx in matches:
            track = tracked_objects[track_idx]
            det = current_detections[det_idx]
            embed = current_embeddings[det_idx]
            track.update(det['bbox'], embed)
            updated_tracks.append(track)

        for i in range(len(tracked_objects)):
            if i not in matched_tracks_indices:
                track = tracked_objects[i]
                track.miss_count += 1
                if track.miss_count <= track.max_age:
                    updated_tracks.append(track)

        for i in range(len(current_detections)):
            if i not in matched_detections_indices and current_detections[i]['class_id'] in CLASSES_TO_PROCESS:
                new_track = Track(next_id, current_detections[i]['bbox'], current_detections[i]['class_id'])
                new_track.appearance_embedding = current_embeddings[i]
                updated_tracks.append(new_track)
                next_id += 1

        tracked_objects = updated_tracks

    for track in tracked_objects:
        if track.miss_count > 0:
            continue

        x1, y1, x2, y2 = track.bbox.astype(int)

        display_info = CLASS_DISPLAY_INFO.get(track.class_id)
        class_name_for_label = display_info["name"] if display_info else "Unknown"
        display_color = display_info["color"] if display_info else (128, 128, 128)

        cv2.rectangle(frame, (x1, y1), (x2, y2), display_color, 2)

        text_color = tuple([int((hash(track.track_id) * 101 + 55) % 255),
                           int((hash(track.track_id) * 212 + 110) % 255),
                           int((hash(track.track_id) * 323 + 165) % 255)])

        text = f"{class_name_for_label} ID: {track.track_id}"
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

    prev_frame_rgb_tensor = curr_frame_tensor

    out.write(frame)

    if frame_count % 50 == 0:
        print(f"Processed {frame_count}/{total_frames} frames...")

cap.release()
out.release()
print(f"Video processing complete. Output saved to {OUTPUT_VIDEO_PATH}")