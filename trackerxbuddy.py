import cv2
import numpy as np
from deepface import DeepFace
from mtcnn import MTCNN
import time
import logging

# Set up logging for debugging and information output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the MTCNN detector for robust face detection
detector = MTCNN()

def create_tracker():
    """
    Create and return a CSRT tracker using either the standard or legacy API.
    """
    if hasattr(cv2, 'TrackerCSRT_create'):
        return cv2.TrackerCSRT_create()
    elif hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerCSRT_create'):
        return cv2.legacy.TrackerCSRT_create()
    else:
        raise AttributeError("CSRT tracker not available. Please install opencv-contrib-python.")

# Settings for detection and emotion analysis intervals (to balance accuracy and performance)
detection_interval = 5   # Run face detection every 5 frames
emotion_interval = 10    # Update emotion analysis every 10 frames
frame_count = 0

# Dictionaries for maintaining trackers, bounding boxes, and emotion labels.
# Internal IDs (used for tracking) will be dynamically mapped to display IDs.
trackers = {}       # internal_face_id -> tracker instance
face_bboxes = {}    # internal_face_id -> (x, y, w, h)
face_emotions = {}  # internal_face_id -> emotion label string

iou_threshold = 0.3
face_id_counter = 0  # Global counter for assigning new internal IDs

# Initialize video capture (webcam)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    logging.error("Unable to open webcam.")
    exit()

prev_frame_time = time.time()

def compute_iou(boxA, boxB):
    """
    Compute the Intersection over Union (IoU) for two bounding boxes.
    Boxes are in the format (x, y, w, h).
    """
    xA, yA, wA, hA = boxA
    xA2, yA2 = xA + wA, yA + hA

    xB, yB, wB, hB = boxB
    xB2, yB2 = xB + wB, yB + hB

    xI1 = max(xA, xB)
    yI1 = max(yA, yB)
    xI2 = min(xA2, xB2)
    yI2 = min(yA2, yB2)

    interWidth = max(0, xI2 - xI1)
    interHeight = max(0, yI2 - yI1)
    interArea = interWidth * interHeight

    areaA = wA * hA
    areaB = wB * hB

    iou = interArea / float(areaA + areaB - interArea) if (areaA + areaB - interArea) > 0 else 0
    return iou

while True:
    ret, frame = cap.read()
    if not ret:
        logging.error("Failed to capture frame.")
        break

    # Resize frame to 640x480 for improved performance
    frame = cv2.resize(frame, (640, 480))
    frame_count += 1

    # ---------------------------
    # Face Detection and Tracker (Re)Initialization
    # ---------------------------
    if frame_count % detection_interval == 0:
        # Convert frame to RGB for MTCNN (which expects RGB images)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = detector.detect_faces(frame_rgb)
        assigned_ids = set()
        
        # Filtering parameters to reduce false positives
        confidence_threshold = 0.90  # Only consider detections above this confidence
        min_box_size = 30            # Minimum width/height for a detection

        for detection in detections:
            # Filter out low-confidence detections
            if detection.get('confidence', 0) < confidence_threshold:
                continue

            box = detection.get('box', [])
            if len(box) != 4:
                continue
            x, y, w, h = box
            # Filter out boxes that are too small (likely false positives)
            if w < min_box_size or h < min_box_size:
                continue

            x, y = max(0, x), max(0, y)
            detection_box = (x, y, w, h)
            best_iou = 0
            best_id = None

            # Compare this detection with existing tracked bounding boxes
            for internal_id, old_box in face_bboxes.items():
                iou_score = compute_iou(detection_box, old_box)
                if iou_score > best_iou:
                    best_iou = iou_score
                    best_id = internal_id

            if best_iou < iou_threshold or best_id is None:
                # New face detected: assign a new internal ID and initialize a tracker
                face_id_counter += 1
                tracker = create_tracker()
                tracker.init(frame, detection_box)
                trackers[face_id_counter] = tracker
                face_bboxes[face_id_counter] = detection_box
                logging.info(f"New face detected with internal ID: {face_id_counter}")
            else:
                # Update existing face tracker with the new detection (to reduce drift)
                if best_id not in assigned_ids:
                    tracker = create_tracker()
                    tracker.init(frame, detection_box)
                    trackers[best_id] = tracker
                    face_bboxes[best_id] = detection_box
                    assigned_ids.add(best_id)
    else:
        # Update trackers on frames where detection is not performed
        remove_ids = []
        for internal_id, tracker in trackers.items():
            success, box = tracker.update(frame)
            if success:
                face_bboxes[internal_id] = tuple(map(int, box))
            else:
                # If a tracker fails, mark it for removal
                remove_ids.append(internal_id)
        for fid in remove_ids:
            logging.info(f"Removing lost tracker for internal face ID: {fid}")
            trackers.pop(fid, None)
            face_bboxes.pop(fid, None)
            face_emotions.pop(fid, None)

    # ---------------------------
    # Emotion Analysis (performed every emotion_interval frames)
    # ---------------------------
    if frame_count % emotion_interval == 0:
        for internal_id, bbox in face_bboxes.items():
            x, y, w, h = bbox
            face_roi = frame[y:y+h, x:x+w]
            if face_roi.size != 0 and w > 20 and h > 20:
                face_roi_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                try:
                    analysis = DeepFace.analyze(
                        face_roi_rgb,
                        actions=['emotion'],
                        detector_backend='mtcnn',
                        enforce_detection=False
                    )
                    if isinstance(analysis, list):
                        analysis = analysis[0]
                    dominant_emotion = analysis.get('dominant_emotion', 'Unknown')
                    emotion_conf = analysis.get('emotion', {}).get(dominant_emotion, 0)
                    face_emotions[internal_id] = f"{dominant_emotion} ({emotion_conf:.2f})"
                except Exception as e:
                    logging.error(f"Error processing emotion for internal face ID {internal_id}: {e}")
                    face_emotions[internal_id] = "Error"
            else:
                face_emotions[internal_id] = "No valid ROI"

    # ---------------------------
    # Dynamic Reassignment of Display IDs
    # ---------------------------
    # For display, sort current faces (for example, by the x-coordinate) and assign sequential IDs
    sorted_faces = sorted(face_bboxes.items(), key=lambda item: item[1][0])
    for display_id, (internal_id, bbox) in enumerate(sorted_faces, start=1):
        x, y, w, h = bbox
        label = f"ID {display_id}: {face_emotions.get(internal_id, 'N/A')}"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # ---------------------------
    # FPS Calculation and Display
    # ---------------------------
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time) if (new_frame_time - prev_frame_time) > 0 else 0
    prev_frame_time = new_frame_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("VisageIntel - Real-Time Face & Emotion Analytics", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        logging.info("Exiting application.")
        break

cap.release()
cv2.destroyAllWindows()

