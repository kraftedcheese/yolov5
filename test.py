import torch
import os
from PIL import Image

# For multiple images
# test_dir = "imgs"
# imgs = []
# for filename in os.listdir(test_dir):
#   f = os.path.join(test_dir, filename)
#   imgs.append(Image.open(f))

def predict(img_path):
  # See if possible to dockerize torch.hub.load
  model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/exp/weights/best.pt')
  model.eval()

  # For a single image
  img = Image.open(img_path)

  # Results is models.common.Detections, which is a custom YOLOv5 object.
  results = model(img, size=640)
  results.print()
  # .show() shows the image with bbox overlay (not sure how it works with Docker)
  # results.show()
  df = results.pandas().xyxy[0]
  json_data = df.to_json()
  return json_data

if __name__ == "__main__":
  out = predict("imgs/abc.jpg")
  print(out)