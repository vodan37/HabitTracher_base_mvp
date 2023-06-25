import cv2
import pandas as pd
from openvino.runtime import Core

from utils.openvino_utils import detect
from utils.draw_utils import draw_results_with_motion
from utils.model_utils import dict_classes
from utils.motion_finder import MotionFinder
from utils.statistics import print_statistic

df = pd.DataFrame(columns=['x1', 'y1', 'x2', 'y2', 'conf', 'cls', 'is_moving', 'uuid', 'frame_number'])

if __name__ == '__main__':
    motion_finder = MotionFinder()
    ie = Core()
    model = ie.read_model(model="weights/yolov8n_1280_ufo_best_int8_openvino_model/yolov8n_1280_ufo_best.xml")
    compiled_model = ie.compile_model(model=model, device_name="CPU")
    output_layer = compiled_model.output(0)

    # СЮДА НАДО ВСТАВИТЬ ВИДЕО
    cam = cv2.VideoCapture('images_and_videos/video_1min.mp4')
    fps = cam.get(cv2.CAP_PROP_FPS)
    time_per_frame = 1 / fps
    width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)  # float `width`
    height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`

    frame_number = 0
    while True:
        ret, frame = cam.read()
        # print(frame.shape)

        if not ret:
            print("end of video")
            break
        if frame_number % 10 == 0:
            image_orig = frame.copy()
            detections = detect(frame, compiled_model)[0]

            # ЗДЕСЬ ПРОИХОДИТ КЛАССФИКАЦИЯ НА ДВИНАЕТ/СТОИТ
            detections = motion_finder.get_motion_status(detections, image_orig, frame_number)

            for det in detections:
                df = df._append({
                    'x1': float(det[0]),
                    'y1': float(det[1]),
                    'x2': float(det[2]),
                    'y2': float(det[3]),
                    'conf': float(det[4]),
                    'cls': int(det[5]),
                    'is_moving': int(det[6]),
                    'uuid': str(det[7]),
                    'frame_number': int(det[8]),
                }, ignore_index=True)

            image_with_boxes = draw_results_with_motion(detections, frame, dict_classes)

            # cv2.imshow('image', cv2.resize(image_with_boxes, (0,0), fx=0.5, fy=0.5))
            # cv2.waitKey(0)

            # print(df.tail())
        frame_number += 1
    df.to_csv('df_tmp.csv', index=False)
    print_statistic(df)