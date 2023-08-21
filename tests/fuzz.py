#
# Copyright (C) 2023 Intel Corporation
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
import struct

import sys
import atheris

import logging as log
from services.rawTensor import nnhal_raw_tensor_pb2

#from services.rawTensor.rawTensor import Detection
with atheris.instrument_imports(exclude=["nnhal_raw_tensor_pb2","nnhal_raw_tensor_pb2_grpc"]):
    from services.rawTensor.rawTensor import Detection

def Test(data) -> None:
    detector = Detection("ovtk", "CPU", "./", "test.socket", "", "")
    detector.shared_model_file = True
    if len(data) != 16:
        return
    full_data = struct.unpack('%sf' % 4, data)
    data1 = full_data[0:2]
    data2 = full_data[2:4]
    expected_result = []
    for i in range(len(data1)):
        expected_result.append(data1[i]+data2[i])
    my_token = nnhal_raw_tensor_pb2.Token(data=2) # remote_model_2 IRs are for Add
    reqStr = nnhal_raw_tensor_pb2.RequestString(token = my_token)
    detector.prepare(reqStr, 1)
    detector.loadModel(reqStr, 1)
    dt1 = nnhal_raw_tensor_pb2.DataTensor(data=struct.pack('%sf' % len(data1), *data1), node_name="0", tensor_shape=[2], data_type=3)
    dt2 = nnhal_raw_tensor_pb2.DataTensor(data=struct.pack('%sf' % len(data2), *data2), node_name="1", tensor_shape=[2], data_type=3)
    dts = [dt1, dt2]
    reqDTs = nnhal_raw_tensor_pb2.RequestDataTensors(data_tensors=dts, token = my_token)
    rpyDTs = detector.getInferResult(reqDTs, 1)
    print("Full Data {} ".format(full_data))
    my_token = nnhal_raw_tensor_pb2.Token(data=200) #
    reqStr = nnhal_raw_tensor_pb2.RequestString(token = my_token)
    detector.prepare(reqStr, 1)
    reqChunk = nnhal_raw_tensor_pb2.RequestDataChunks(data=data, token = my_token)
    detector.sendXml([reqChunk], 1)
    detector.sendBin([reqChunk], 1)
    detector.release(reqStr, 1)

atheris.Setup(sys.argv, Test)
atheris.Fuzz()
