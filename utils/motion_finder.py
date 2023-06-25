import cv2
import numpy as np
import torch
import imgsim
import uuid


vtr = imgsim.Vectorizer()


def get_iou(bbox_a, bbox_b):
    x1_intersection = max(bbox_a[0], bbox_b[0])
    y1_intersection = max(bbox_a[1], bbox_b[1])
    x2_intersection = min(bbox_a[2], bbox_b[2])
    y2_intersection = min(bbox_a[3], bbox_b[3])
    intersection_area = max(0, x2_intersection - x1_intersection + 1) * max(0, y2_intersection - y1_intersection + 1)
    bbox_a_area = (bbox_a[2] - bbox_a[0] + 1) * (bbox_a[3] - bbox_a[1] + 1)
    bbox_b_area = (bbox_b[2] - bbox_b[0] + 1) * (bbox_b[3] - bbox_b[1] + 1)
    iou = intersection_area / float(bbox_a_area + bbox_b_area - intersection_area)
    return iou


class MotionFinder:
    def __init__(self, history_length=2, thres=20, iou_thres=0.3):
        self.__history_length = history_length
        self.__thres = thres
        self.__iou_thres = iou_thres
        self.__image_prev = None
        self.__bboxes_prev = []

    def get_motion_status(self, bboxes_raw, image, frame_number):
        result = []
        bboxes = bboxes_raw['det']
        for bbox1 in bboxes:
            bbox1_x1, bbox1_y1, bbox1_x2, bbox1_y2, bbox1_conf, bbox1_cls = bbox1
            bbox_ret = [bbox1_x1, bbox1_y1, bbox1_x2, bbox1_y2, bbox1_conf, bbox1_cls, 1, str(uuid.uuid4()), frame_number]
            for bbox2 in self.__bboxes_prev:
                bbox2_x1, bbox2_y1, bbox2_x2, bbox2_y2, bbox2_conf, bbox2_cls, is_moving_prev, id, frame_number = bbox2
                iou = get_iou(bbox1, bbox2)

                if iou > self.__iou_thres and bbox1_cls == bbox2_cls:
                    if self.__image_prev is not None:
                        is_moving = self.object_is_moving(bbox1, bbox2, image)

                        if is_moving:
                            bbox_ret = [bbox1_x1, bbox1_y1, bbox1_x2, bbox1_y2, bbox1_conf, bbox1_cls, 1, id, frame_number]
                            break
                        else:
                            bbox_ret = [bbox1_x1, bbox1_y1, bbox1_x2, bbox1_y2, bbox1_conf, bbox1_cls, 0, id, frame_number]
                    else:
                        bbox_ret = [bbox1_x1, bbox1_y1, bbox1_x2, bbox1_y2, bbox1_conf, bbox1_cls, 1, str(uuid.uuid4()), frame_number]

                else:
                    pass
                    # bbox_ret = [bbox1_x1, bbox1_y1, bbox1_x2, bbox1_y2, bbox1_conf, bbox1_cls, 1, str(uuid.uuid4())]

            result.append(bbox_ret)

        self.__image_prev = image
        self.__bboxes_prev = result
        return result

    def object_is_moving(self, bbox1, bbox2, image):
        x1, y1, x2, y2, conf, label = bbox1
        image1 = image[int(y1):int(y2), int(x1):int(x2)]
        image1_resized = cv2.resize(image1, (32, 32), interpolation=cv2.INTER_AREA)
        # image1_resized = cv2.cvtColor(image1_resized, cv2.COLOR_BGR2GRAY)
        vec0 = vtr.vectorize(image1_resized)

        x1, y1, x2, y2, conf, label, is_moving_prev, id, frame_number = bbox2
        image2 = self.__image_prev[int(y1):int(y2), int(x1):int(x2)]
        image2_resized = cv2.resize(image2, (32, 32), interpolation=cv2.INTER_AREA)
        # image2_resized = cv2.cvtColor(image2_resized, cv2.COLOR_BGR2GRAY)
        vec1 = vtr.vectorize(image2_resized)

        dist = imgsim.distance(vec0, vec1)

        # diff = image1_resized - image2_resized
        # print('mean:', np.mean(diff))
        # print('pers 80:', np.percentile(diff, 80))
        # print('='*50)
        # print(dist)
        # print('='*50)

        # cv2.imshow('now', image1_resized)
        # cv2.imshow('prev', image2_resized)
        # cv2.imshow('diff', diff)
        # cv2.waitKey(0)

        # return np.percentile(diff, 80) > self.__thres
        return dist > self.__thres
