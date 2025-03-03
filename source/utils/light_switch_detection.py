import numpy as np
import cv2
import supervision as sv
from ultralytics import YOLO, YOLOWorld
from ultralytics.data.dataset import YOLODataset
import cv2
import matplotlib.pyplot as plt
from utils.object_detetion import BBox


def _filter_detections_YOLOWorld(detections):

    # squaredness filter
    squaredness = (np.minimum(detections.xyxy[:,2] - detections.xyxy[:,0], detections.xyxy[:,3] - detections.xyxy[:,1])/
                   np.maximum(detections.xyxy[:,2] - detections.xyxy[:,0], detections.xyxy[:,3] - detections.xyxy[:,1]))

    idx_dismiss = np.where(squaredness < 0.95)[0]

    filtered_detections = sv.Detections.empty()
    filtered_detections.class_id = np.delete(detections.class_id, idx_dismiss)
    filtered_detections.confidence = np.delete(detections.confidence, idx_dismiss)
    filtered_detections.data['class_name'] = np.delete(detections.data['class_name'], idx_dismiss)
    filtered_detections.xyxy = np.delete(detections.xyxy, idx_dismiss, axis=0)

    return filtered_detections

def filter_detections_ultralytics(detections, filter_squaredness=True, filter_area=True, filter_within=True):

    detections = detections[0].cpu()
    xyxy = detections.boxes.xyxy.numpy()

    # filter squaredness outliers
    if filter_squaredness:
        squaredness = (np.minimum(xyxy[:, 2] - xyxy[:, 0],
                                  xyxy[:, 3] - xyxy[:, 1]) /
                       np.maximum(xyxy[:, 2] - xyxy[:, 0],
                                  xyxy[:, 3] - xyxy[:, 1]))

        keep_1 = squaredness > 0.5
        xyxy = xyxy[keep_1, :]

    #filter area outliers
    if filter_area:
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        keep_2 = areas < 3*np.median(areas)
        xyxy = xyxy[keep_2, :]

    # filter bounding boxes within larger ones
    if filter_within:
        centers = np.array([(xyxy[:, 0] + xyxy[:, 2]) / 2, (xyxy[:, 1] + xyxy[:, 3]) / 2]).T
        keep_3 = np.ones(xyxy.shape[0], dtype=bool)
        x_in_box = (xyxy[:, 0:1] <= centers[:, 0]) & (centers[:, 0] <= xyxy[:, 2:3])
        y_in_box = (xyxy[:, 1:2] <= centers[:, 1]) & (centers[:, 1] <= xyxy[:, 3:4])
        centers_in_boxes = x_in_box & y_in_box
        np.fill_diagonal(centers_in_boxes, False)
        pairs = np.argwhere(centers_in_boxes)
        idx_remove = pairs[np.where(areas[pairs[:, 0]] - areas[pairs[:, 1]] < 0), 0].flatten()
        keep_3[idx_remove] = False
        xyxy = xyxy[keep_3, :]

    bbox = xyxy
    return bbox


def predict_light_switches(image: np.ndarray, model_type: str = "yolov8", weights_path: str = '', vis_block: bool = False):

    if model_type == "yolo_world":
        model = YOLOWorld("yolov8s-world.pt")
        model.set_classes(["light switch"])

        results_predict = model.predict(image)
        results_predict[0].show()

    elif model_type == "yolov8":
        model = YOLO(weights_path) # conf 0.15
        results_predict = model.predict(source=image, imgsz=1280, conf=0.5, iou=0.4, max_det=9, agnostic_nms=True,
                                        save=False)

        boxes = filter_detections_ultralytics(detections=results_predict)

        if vis_block:
            canv = image.copy()
            for box in boxes:
                xB = int(box[2])
                xA = int(box[0])
                yB = int(box[3])
                yA = int(box[1])

                cv2.rectangle(canv, (xA, yA), (xB, yB), (0, 255, 0), 2)

            plt.imshow(cv2.cvtColor(canv, cv2.COLOR_BGR2RGB))
            plt.show()

        bbs = []
        for box in boxes:
            bbs.append(BBox(box[0], box[1], box[2], box[3]))

        return bbs

def validate_light_switches(data_path, model_type: str = "yolov8"):

    if model_type == "yolov8":
        model = YOLO('../../weights/train27/weights/best.pt')
        metrics = model.val(data=data_path, imgsz=1280)
        result = metrics.results_dict
        a = 2

    elif model_type == "yolo_world":
        model = YOLOWorld("yolov8s-world.pt")
        model.set_classes(["light switch"])
        metrics = model.val(data=data_path, imgsz=1280)
        result = metrics.results_dict
        a = 2

    a = 2

    pass


if __name__ == "__main__":

    data_path = "/home/cvg-robotics/tim_ws/data_switch_detection/20240706_test/data.yaml"
    validate_light_switches(data_path=data_path, model_type="yolov8")
