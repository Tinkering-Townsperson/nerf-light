from ultralytics import YOLO


class Camera:
	def __init__(self, model: YOLO, source: int = 0, debug: bool = False):
		self.model: YOLO = model
		self.source: int = source
		self.debug: bool = debug

	def mainloop(self):
		results = self.model.predict(source=self.source, show=self.debug, conf=0.4, stream=True)

		for r in results:
			if r.boxes is None:
				continue

			for box in r.boxes:
				if box.cls not in (0,):
					continue

				confidence = float(box.conf)

				if confidence < 0.4:
					continue

				self.parse_box(box)

	def parse_box(self, box):
		x1, y1, x2, y2 = map(int, box.xyxy[0])
		print(f"Person detected with confidence: {float(box.conf):.2f} at [{x1}, {y1}, {x2}, {y2}]")
