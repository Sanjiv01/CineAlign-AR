import glob
from ultralytics import YOLOWorld

CLASSES = ["person", "bus"]

def main(image_dir: str, model_name: str, classes: list=CLASSES, batch_size: int=32):
    image_list = sorted(glob.glob(image_dir))

    model = YOLOWorld(f"{model_name}.pt")
    model.set_classes(classes)

    for i in range(0, len(image_list), batch_size):
        batch = image_list[i:i+batch_size]
        results = model.predict(batch)
        for r in results:
            boxes = r.boxes.xyxy
            print(boxes)

    for r in results:
        print(r.path)
        print(r.names)
        print(r.boxes.xyxy)