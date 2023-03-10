#
# Copyright (C) 2020-2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
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
import object_detection_pb2
import object_detection_pb2_grpc
import adaptors.create_interface as create_interface
import common.inputValidations as inputValidations


class Detection(object_detection_pb2_grpc.DetectionServicer):
    def __init__(self, interface, unix_socket, remote_port, img_height, img_width):
        super().__init__()
        self.remote_port = remote_port
        self.interface = interface
        self.unix_socket = unix_socket
        self.img_height = img_height
        self.img_width = img_width

    def getPredictions(self, request, context):
        start_time = datetime.datetime.now()
        if not self.interface.isModelLoaded(2000):#Wait upto 2 seconds for model load
            print("Model Load Failure")
            sys.exit(1)

        input_shape = [1, 3, self.img_width, self.img_height]
        data = np.fromstring(request.data, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        node_name = "Parameter_0"
        img = cv2.resize(img, (self.img_width, self.img_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.normalize(img.astype('float'), None, -1.0, 1.0, cv2.NORM_MINMAX)

        #creating dictionary as required by adapters
        input = {node_name: (img, input_shape)}
        result = self.interface.run_detection(input)
        end_time = datetime.datetime.now()
        serving_duration = (end_time - start_time).total_seconds() * 1000

        #Modified according to new output format of the adapter
        output_classes = result["Transpose_537"][0]
        output_locations = result["Transpose_535"][0]
        detections = []
        for i in range(0, 1917):  # returns 1917 detections for each class
            classes = output_classes[0,0,i,1:]
            coordinates=output_locations[0,0,i,:]
            top_class_index = np.argmax(classes)+1
            EXP_SCORE = 0.5
            det_score = (1.0/(1.0+math.exp(-classes[top_class_index-1])))
            if (det_score > EXP_SCORE):
                detections.append(object_detection_pb2.Prediction(index0=coordinates[0],
                                                                  index1=coordinates[1],
                                                                  index2=coordinates[2],
                                                                  index3=coordinates[3],
                                                                  confidence=det_score,
                                                                  classIndex=top_class_index,
                                                                  predictIndex=i))

        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds() * 1000
        print("time in ms run_detection: {} getInferResult: {}".format(serving_duration,
                                                                            duration))
        return object_detection_pb2.PredictionsList(predictions=detections)


def serve(detection):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    object_detection_pb2_grpc.add_DetectionServicer_to_server(detection, server)
    if(detection.unix_socket != ""):
        server.add_insecure_port("unix:" + detection.unix_socket)
        os.chmod(detection.unix_socket, 0o666)
    else:
        server.add_insecure_port('[::]:{}'.format(detection.remote_port))
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    logging.basicConfig()
    parser = argparse.ArgumentParser(description='Object detection requests via TFS gRPC API.')
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
    parser.add_argument('--serving_model_name', required=False, default='model_od',
                        help='Specify model name set for inference service.')
    parser.add_argument('--interface', required=False, default='ovms',
                        help='Specify serving interface: currently supported interface \'ovms\'\
                         and \'ovtk\' for dynamically selecting the interface')
    parser.add_argument('--width', required=False,
                        help='How the input image width should be resized in pixels',
                        default=300, type=int)
    parser.add_argument('--height', required=False,
                        help='How the input image width should be resized in pixels',
                        default=300, type=int)
    parser.add_argument('--device', required=False, default='AUTO',
                        help='Specify device you want do inference with: currently supported devices \'CPU\'\
                         \'GPU\' and \'GPU.{device # of GPU}\' in case of multiple GPUs for dynamically selecting device')
    args = vars(parser.parse_args(sys.argv[1:]))
    inputValidations.validate(args)
    serving_address = args['serving_address']
    serving_port = args['serving_port']
    dir_path = args['serving_mounted_modelDir']
    serving_model_name=args['serving_model_name']
    adapter = args['interface']
    device = args['device']
    interface = create_interface.createInterfaceObj(adapter, device, serving_address, serving_port,
                                                    serving_model_name, dir_path)
    print("Starting Service")
    serve(Detection(interface, args['unix_socket'], args['remote_port'], args['height'], args['width']))
