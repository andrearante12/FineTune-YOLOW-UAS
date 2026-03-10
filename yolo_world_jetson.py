import cv2
import time
from ultralytics import YOLOWorld

# --- MODEL SELECTION ---
#model = YOLOWorld("yolov8s-worldv2.pt")  # SMALLEST
#model = YOLOWorld("yolov8m-worldv2.pt")    # MEDIUM (Current)
#model = YOLOWorld("yolov8x-worldv2.pt")  # LARGEST
model = YOLOWorld("5_epoch.pt") # Finetuned version

# Define your specific objects
classes = ["robot arm", "cactus", "plant", "chess piece"]
model.set_classes(classes)

# Initialize Camera
cap = cv2.VideoCapture(0)

# FPS calculation variables
prev_time = 0
fps_avg = 0  # To store the filtered FPS

print("--- Starting Detection (Press 'q' to quit) ---")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Start timer
    start_time = time.time()

    # 2. Run Inference
    results = model.predict(frame, device=0, conf=0.05, imgsz=640, verbose=False)

    # 3. Calculate FPS
    end_time = time.time()
    current_fps = 1 / (end_time - start_time)
    
    # 4. Apply a simple Low-Pass Filter for smooth readings
    # (90% previous value, 10% new value)
    fps_avg = (0.9 * fps_avg) + (0.1 * current_fps) if fps_avg != 0 else current_fps

    # 5. Draw results and overlay FPS text
    annotated_frame = results[0].plot()
    
    cv2.putText(
        annotated_frame, 
        f"FPS: {fps_avg:.1f}", 
        (20, 50),                 # Position (X, Y)
        cv2.FONT_HERSHEY_SIMPLEX, # Font
        1.0,                      # Font scale
        (0, 255, 0),              # Color (Green)
        2,                        # Thickness
        cv2.LINE_AA
    )

    cv2.imshow("YOLO-World + FPS Counter", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
