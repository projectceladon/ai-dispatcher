#
# Copyright (C) 2020-2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
#

import os
from concurrent import futures
import argparse
import logging
import sys
import grpc
import datetime
import numpy as np
import cv2
import math
import facemask_detection_pb2
import facemask_detection_pb2_grpc
import adaptors.create_interface as create_interface
import common.inputValidations as inputValidations
from utils.anchor_generator import generate_anchors
from utils.anchor_decode import decode_bbox
from utils.nms import single_class_non_max_suppression

class Detection(facemask_detection_pb2_grpc.DetectionServicer):
    def __init__(self, interface, unix_socket, remote_port, img_height, img_width):
        super().__init__()
        self.remote_port = remote_port
        self.interface = interface
        self.unix_socket = unix_socket
        self.img_height = img_height
        self.img_width = img_width

    def getPredictions(self, request, context):
        #class name -id mapping
        # id2class = {0: 'Mask', 1: 'NoMask'}

        start_time = datetime.datetime.now()
        if not interface.isModelLoaded(2000):#Wait upto 2 seconds for model load
            print("Model Load Failure")
            sys.exit(1)
        input_shape = [1, 3, self.img_width, self.img_height]
        # anchor configuration
        feature_map_sizes = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]]
        anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
        anchor_ratios = [[1, 0.62, 0.42]] * 5
        # generate anchors
        anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)
        anchors_exp = np.expand_dims(anchors, axis=0)
        data = np.fromstring(request.data, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)  # BGR color format, shape HWC
        node_name = "data_1"
        img = cv2.resize(img, (self.img_width, self.img_height))
        image = img / 255.0

        #creating dictionary as required by adapters
        input = {node_name: (image, input_shape)}
        result = interface.run_detection(input)
        end_time = datetime.datetime.now()
        serving_duration = (end_time - start_time).total_seconds() * 1000

        y_bboxes_output = result["loc_branch_concat_1/concat"][0]
        y_cls_output = result["cls_branch_concat_1/concat"][0]
        y_bboxes = decode_bbox(anchors_exp, y_bboxes_output)[0]
        y_cls = y_cls_output[0]
        # To speed up, do single class NMS, not multiple classes NMS.
        bbox_max_scores = np.max(y_cls, axis=1)
        bbox_max_score_classes = np.argmax(y_cls, axis=1)
        # keep_idx is the alive bounding box after nms.
        keep_idxs = single_class_non_max_suppression(y_bboxes,
                                                 bbox_max_scores,
                                                 conf_thresh=0.5,
                                                 iou_thresh=0.4,
                                                 )
        result_coords = []
        for idx in keep_idxs:
            conf = float(bbox_max_scores[idx])
            class_id = bbox_max_score_classes[idx]
            bbox = y_bboxes[idx]
            # clip the coordinate, avoid the value exceed the image boundary.
            xmin = max(0, int(bbox[0] * 600))
            ymin = max(0, int(bbox[1] * 400))
            xmax = min(int(bbox[2] * 600), 600)
            ymax = min(int(bbox[3] * 400), 400)
            result_coords.append(facemask_detection_pb2.Prediction(x_min=xmin, y_min=ymin, x_max=xmax, y_max=ymax, confidence=conf, class_id=class_id))

        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds() * 1000
        print("time in ms run_detection: {} getInferResult: {}".format(serving_duration,
                                                                       duration))
        return facemask_detection_pb2.PredictionsList(predictions=result_coords)


def serve(detection):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    facemask_detection_pb2_grpc.add_DetectionServicer_to_server(detection, server)
    if(detection.unix_socket != ""):
        server.add_insecure_port("unix:" + detection.unix_socket)
        os.chmod(detection.unix_socket, 0o666)
    else:
        server.add_insecure_port('[::]:{}'.format(detection.remote_port))
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    logging.basicConfig()
    parser = argparse.ArgumentParser(description='Face mask detection requests via TFS gRPC API.'
                                                 'analyses input images and detects faces.It relies on '
                                                 'model face_detection...')
    parser.add_argument('--remote_port', required=False, default=50051,
                        help='Specify port to grpc service. default: 50051')
    parser.add_argument('--unix_socket', required=False, default="",
                        help='Specify path to grpc unix socket. default=""')
    parser.add_argument('--serving_address', required=False, default='localhost',
                        help='Specify url to inference service. default:localhost')
    parser.add_argument('--serving_mounted_modelDir', required=True,
                        help='Specify full path to mounted Directory for model loading.')
    parser.add_argument('--serving_port', required=False, default=9000,
                        help='Specify port to inference service. default: 9000')
    parser.add_argument('--serving_model_name', required=False, default='face_mask_detection',
                        help='Specify model name set for inference service.')
    parser.add_argument('--interface', required=False, default='ovms',
                        help='Specify serving interface: currently supported interface \'ovms\'\
                         and \'ovtk\' for dynamically selecting the interface')
    parser.add_argument('--width', required=False, help='How the input image width should be'
                                                    ' resized in pixels', default=1200, type=int)
    parser.add_argument('--height', required=False, help='How the input image width should be'
                                                     ' resized in pixels', default=800, type=int)
    parser.add_argument('--device', required=False, default='AUTO',
                        help='Specify device you want do inference with: currently supported devices \'CPU\'\
                         \'GPU\' and \'GPU.{device # of GPU}\' in case of multiple GPUs for dynamically selecting device')
    args = vars(parser.parse_args(sys.argv[1:]))
    inputValidations.validate(args)
    serving_address = args['serving_address']
    serving_port = args['serving_port']
    dir_path = args['serving_mounted_modelDir']
    serving_model_name = args['serving_model_name']
    adapter = args['interface']
    device = args['device']
    interface = create_interface.createInterfaceObj(adapter, device, serving_address, serving_port,
                                                    serving_model_name, dir_path)
    print("Starting Service")
    serve(Detection(interface, args['unix_socket'], args['remote_port'], args['height'], args['width']))
