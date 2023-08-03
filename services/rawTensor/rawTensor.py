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
import argparse
import logging as log
import sys
import grpc
import datetime
import numpy as np
from concurrent import futures
import nnhal_raw_tensor_pb2
import nnhal_raw_tensor_pb2_grpc
import adaptors.create_interface as create_interface
import common.inputValidations as inputValidations


class Detection(nnhal_raw_tensor_pb2_grpc.DetectionServicer):
    def __init__(self, adapter, device, dir_path, unix_socket, remote_port, vsock):
        super().__init__()
        self.adapter = adapter
        self.device = device
        self.dir_path = dir_path
        self.unix_socket = unix_socket
        self.remote_port = remote_port
        self.vsock = vsock
        self.interface = {}
        self.shared_model_file = False
        if(serving_model_name == 'shared'):
            self.shared_model_file = True

    def prepare(self, requestStr, context):
        log.info("Preparing model " + str(requestStr.token.data))
        self.interface[requestStr.token.data] = create_interface.createInterfaceObj(self.adapter, self.device, "", "",
                                                             requestStr.token.data, self.dir_path)
        if not self.shared_model_file:
            self.interface[requestStr.token.data].prepareDir()
        return nnhal_raw_tensor_pb2.ReplyStatus(status=True)

    def release(self, requestStr, context):
        if not self.shared_model_file:
            self.interface[requestStr.token.data].cleanUp()
        self.interface[requestStr.token.data] = None
        return nnhal_raw_tensor_pb2.ReplyStatus(status=True)

    def sendXml(self, requestChunks, context):
        for chunk in requestChunks:
            self.interface[chunk.token.data].saveXML(chunk)
        return nnhal_raw_tensor_pb2.ReplyStatus(status=True)

    def sendBin(self, requestChunks, context):
        for chunk in requestChunks:
            self.interface[chunk.token.data].saveBin(chunk)
        return nnhal_raw_tensor_pb2.ReplyStatus(status=True)

    def loadModel(self, request, context):
        #Wait upto 25 seconds for model load
        self.interface[request.token.data].quant_model = request.quant_type
        if not self.interface[request.token.data].isModelLoaded(25000):
            log.error("Model Load Failure")
            return nnhal_raw_tensor_pb2.ReplyStatus(status=False)
        return nnhal_raw_tensor_pb2.ReplyStatus(status=True)

    def getInferResult(self, request, context):
        start_time = datetime.datetime.now()
        reply_data_tensor = nnhal_raw_tensor_pb2.ReplyDataTensors()
        run_start_time = datetime.datetime.now()
        if(request.token.data not in self.interface):
            log.error("Infer failed : Interface not prepared!!!")
            return reply_data_tensor
        #decoding the grpc request and sending to adapter
        input = {}
        for datatensor in request.data_tensors:
            node_name = datatensor.node_name
            input_shape = datatensor.tensor_shape
            img_data = datatensor.data
            data_type = datatensor.data_type
            data = np.frombuffer(img_data, np.dtype(getMappedDatatype(data_type)))
            input[node_name] = (data, input_shape)

        try:
            result = self.interface[request.token.data].run_detection(input)
        except Exception as inst:
            log.warning(inst)
            log.warning("Infer failed")
            return reply_data_tensor
        end_time = datetime.datetime.now()
        serving_duration = (end_time - run_start_time).total_seconds() * 1000
        #Modifed according to the new interface output format
        for key in result.keys():
            output_data_tensor = reply_data_tensor.data_tensors.add()
            output_data_tensor.data = result[key][0].tobytes()
            output_data_tensor.node_name = key
            shape = result[key][1]
            output_data_tensor.tensor_shape.extend(shape)
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds() * 1000
        log.debug("time in ms run_detection: {} getInferResult: {}".format(serving_duration,
                                                                            duration))
        return reply_data_tensor

def getMappedDatatype(type):
    types = {
        0: '<b',
        1: '<f2',
        2: '<f2',
        3: '<f4',
        4: '<f8',
        5: '<i1',
        6: '<i1',
        7: '<i2',
        8: '<i4',
        9: '<i8',
        10: '<u1',
        11: '<u1',
        12: '<u1',
        13: '<u2',
        14: '<u4',
        15: '<u8',
    }
    return types.get(type, 'f4')

def serve(detection):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options = [
        ('grpc.max_send_message_length', 1024*1024*1024),
        ('grpc.max_receive_message_length', 1024*1024*1024)
        ])
    nnhal_raw_tensor_pb2_grpc.add_DetectionServicer_to_server(detection, server)
    if(detection.unix_socket != ""):
        server.add_insecure_port("unix:" + detection.unix_socket)
        os.chmod(detection.unix_socket, 0o666)
        detection.shared_model_file = True
    elif(detection.vsock == "true"):
        server.add_insecure_port("vsock:-1:{}".format(detection.remote_port))
    else :
        server.add_insecure_port('[::]:{}'.format(detection.remote_port))
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    log.basicConfig(format='%(asctime)s %(message)s')
    log.root.setLevel(log.INFO)
    parser = argparse.ArgumentParser(description='Remote Inference from NNHAL')
    parser.add_argument('--remote_port', required=False, default=50051,
                        help='Specify port to grpc service. default: 50051')
    parser.add_argument('--unix_socket', required=False, default="",
                        help='Specify path to grpc unix socket. default=""')
    parser.add_argument('--vsock', required=False, default="false",
                        help='Specify path to grpc vsock socket. default="false"')
    parser.add_argument('--serving_address', required=False, default='localhost',
                        help='Specify url to inference service. default:localhost')
    parser.add_argument('--serving_mounted_modelDir', required=False,
                        help='Specify full path to mounted Directory for model loading.')
    parser.add_argument('--serving_port', required=False, default=9000,
                        help='Specify port to inference service. default: 9000')
    parser.add_argument('--serving_model_name', required=False, default='remote_model',
                        help='Specify model name set for inference service.')
    parser.add_argument('--interface', required=False, default='ovtk',
                        help='Specify serving interface: currently supported interface \
                         \'ovtk\' for dynamically selecting the interface')
    parser.add_argument('--device', required=False, default='AUTO',
                        help='Specify device you want do inference with: currently supported devices \'CPU\'\
                         \'GPU\' and \'GPU.{device # of GPU}\' in case of multiple GPUs for dynamically selecting device')
    args = vars(parser.parse_args(sys.argv[1:]))
    inputValidations.validate(args)
    unix_socket_path = args['unix_socket']
    dir_path = args['serving_mounted_modelDir']
    if(unix_socket_path != ""):
        dir_path = os.path.dirname(unix_socket_path)
    serving_address = args['serving_address']
    serving_port = args['serving_port']
    adapter = args['interface']
    serving_model_name = args['serving_model_name']
    device = args['device']
    log.info("Starting Service")
    log.info("unix_path :" + unix_socket_path)
    log.info("dir_path :" + dir_path)
    serve(Detection(adapter, device, dir_path, unix_socket_path, args['remote_port'], args['vsock']))
