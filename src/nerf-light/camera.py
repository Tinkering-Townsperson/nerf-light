from ultralytics import YOLO

model = YOLO('yolov8n.pt')


def mainloop():
	results = model.predict(source=0, show=True, conf=0.5, stream=True)

	# Iterate through the results and extract person detections (class_id=0 for COCO)
	for r in results:
		boxes = r.boxes
		if boxes is not None:
			for box in boxes:
				if box.cls == 0:  # Class ID for "person" in COCO dataset
					x1, y1, x2, y2 = map(int, box.xyxy[0])
					confidence = float(box.conf)
					print(f"Person detected with confidence: {confidence:.2f} at [{x1}, {y1}, {x2}, {y2}]")
