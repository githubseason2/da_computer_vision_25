import cv2 as cv
import numpy as np
import sys

YOLO_PATH = "C:/Users/SK/Downloads"
SORT_PATH = "C:/Users/SK/Documents/GitHub/Sort/sort"

sys.path.append(SORT_PATH)
from sort import Sort


def construct_yolo_v3():
    with open(f"{YOLO_PATH}/coco.names", 'r') as f:
        class_names = [line.strip() for line in f.readlines()]

    net = cv.dnn.readNet(f"{YOLO_PATH}/yolov3.weights", f"{YOLO_PATH}/yolov3.cfg")
    layer_names = net.getLayerNames()
    out_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    return net, out_layers, class_names


def yolo_detect(img, model, out_layers, conf_threshold=0.5, nms_threshold=0.4):
    height, width = img.shape[:2]
    blob = cv.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    model.setInput(blob)
    outputs = model.forward(out_layers)

    boxes, confidences, class_ids = [], [], []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x, center_y = int(detection[0] * width), int(detection[1] * height)
                w, h = int(detection[2] * width), int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, x + w, y + h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    if isinstance(indices, tuple):
        indices = indices[0] if len(indices) > 0 else []

    indices = np.array(indices).flatten() if len(indices) > 0 else []

    results = []
    for i in indices:
        x1, y1, x2, y2 = boxes[i]
        results.append([x1, y1, x2, y2, confidences[i], class_ids[i]])
    return results


def try_open_camera():
    backends = [cv.CAP_MSMF, cv.CAP_ANY, cv.CAP_DSHOW]
    for backend in backends:
        for idx in range(5):
            cap = cv.VideoCapture(idx, backend)
            if cap.isOpened():
                print(f"카메라 인식 성공: index {idx}, backend {backend}")
                return cap
            cap.release()
    print("카메라 열기 실패: 모든 백엔드 및 인덱스 시도 실패")
    return None


def main():
    model, out_layers, class_names = construct_yolo_v3()
    tracker = Sort()
    colors = np.random.uniform(0, 255, size=(100, 3))

    cap = try_open_camera()
    if cap is None:
        sys.exit("카메라 열기 실패로 프로그램 종료")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임 수신 실패")
            break

        detections = yolo_detect(frame, model, out_layers)
        persons = [d for d in detections if d[5] == 0]  # class_id 0: person

        dets = np.array([[*d[:4], d[4]] for d in persons]) if persons else np.empty((0, 5))
        tracks = tracker.update(dets)

        for d in tracks.astype(int):
            x1, y1, x2, y2, track_id = d
            color = colors[track_id % 100]
            cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv.putText(frame, f"ID {track_id}", (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv.imshow("SORT Person Tracking", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
