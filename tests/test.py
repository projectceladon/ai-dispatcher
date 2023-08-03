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

import unittest
import grpc
import logging as log
import random
import struct
import numpy as np

from services.rawTensor import nnhal_raw_tensor_pb2
from services.rawTensor import nnhal_raw_tensor_pb2_grpc
from services.rawTensor.rawTensor import getMappedDatatype

class TestRawTensorService(unittest.TestCase):
    def setUp(self) -> None:
        #print("SETUP")
        self.detector = ""
        self.grpc_channel = grpc.insecure_channel('unix:test.socket')
        if(self.grpc_channel):
            self.detector = nnhal_raw_tensor_pb2_grpc.DetectionStub(self.grpc_channel)
            #print("STUB CREATED")
        return super().setUp()

    def tearDown(self) -> None:
        #print("TEARDOWN")
        return super().tearDown()

    def test_valid_model(self):
        log.info("TEST")
        reqStr = nnhal_raw_tensor_pb2.RequestString(token = nnhal_raw_tensor_pb2.Token(data=1))
        self.detector.prepare(reqStr)
        repSts = self.detector.loadModel(reqStr)
        self.assertTrue(repSts.status)

    def test_invalid_model(self):
        log.info("TEST")
        reqStr = nnhal_raw_tensor_pb2.RequestString(token = nnhal_raw_tensor_pb2.Token(data=100))
        self.detector.prepare(reqStr)
        repSts = self.detector.loadModel(reqStr)
        self.assertFalse(repSts.status)

    def test_infer_without_prepare(self):
        log.info("TEST")
        img_data = []
        dts = [nnhal_raw_tensor_pb2.DataTensor(data=struct.pack('%sf' % len(img_data), *img_data), node_name="0", tensor_shape=[1], data_type=3)]
        reqDTs = nnhal_raw_tensor_pb2.RequestDataTensors(data_tensors=dts, token = nnhal_raw_tensor_pb2.Token(data=101))
        rpyDTs = self.detector.getInferResult(reqDTs)
        self.assertEqual(len(rpyDTs.data_tensors), 0)

    def test_infer_without_load(self):
        log.info("TEST")
        reqStr = nnhal_raw_tensor_pb2.RequestString(token = nnhal_raw_tensor_pb2.Token(data=102))
        self.detector.prepare(reqStr)
        img_data = []
        dts = [nnhal_raw_tensor_pb2.DataTensor(data=struct.pack('%sf' % len(img_data), *img_data), node_name="0", tensor_shape=[1], data_type=3)]
        reqDTs = nnhal_raw_tensor_pb2.RequestDataTensors(data_tensors=dts, token = nnhal_raw_tensor_pb2.Token(data=102))
        rpyDTs = self.detector.getInferResult(reqDTs)
        self.assertEqual(len(rpyDTs.data_tensors), 0)

    def test_load_without_prepare(self):
        log.info("TEST")
        reqStr = nnhal_raw_tensor_pb2.RequestString(token = nnhal_raw_tensor_pb2.Token(data=103))
        self.detector.prepare(reqStr)
        repSts = self.detector.loadModel(reqStr)
        self.assertFalse(repSts.status)

    def test_valid_infer_AddRandom(self):
        log.info("TEST")
        img_data1 = [round(random.uniform(-10.0, 10.0),3), round(random.uniform(-10.0, 10.0),3)] # Random input1
        img_data2 = [round(random.uniform(-10.0, 10.0),3), round(random.uniform(-10.0, 10.0),3)] # Random input2
        expected_result = []
        for i in range(len(img_data1)):
            expected_result.append(img_data1[i]+img_data2[i])
        my_token = nnhal_raw_tensor_pb2.Token(data=2) # remote_model_2 IRs are for Add
        reqStr = nnhal_raw_tensor_pb2.RequestString(token = my_token)
        self.detector.prepare(reqStr)
        self.detector.loadModel(reqStr)
        dt1 = nnhal_raw_tensor_pb2.DataTensor(data=struct.pack('%sf' % len(img_data1), *img_data1), node_name="0", tensor_shape=[2], data_type=3)
        dt2 = nnhal_raw_tensor_pb2.DataTensor(data=struct.pack('%sf' % len(img_data2), *img_data2), node_name="1", tensor_shape=[2], data_type=3)
        dts = [dt1, dt2]
        reqDTs = nnhal_raw_tensor_pb2.RequestDataTensors(data_tensors=dts, token = my_token)
        rpyDTs = self.detector.getInferResult(reqDTs)
        self.assertEqual(len(rpyDTs.data_tensors), 1)
        datatensor = rpyDTs.data_tensors[0]
        out_data = datatensor.data
        resData = np.frombuffer(out_data, np.dtype(getMappedDatatype(3)))
        actual_result = []
        for i in range(len(resData)):
            self.assertAlmostEqual(expected_result[i],resData[i], places=3)

    def test_invalid_infer_ResizeBilenear(self):
        log.info("TEST")
        img_data = [1.0, 1.0, 2.0, 2.0]
        expected_result = []
        my_token = nnhal_raw_tensor_pb2.Token(data=1) # remote_model_2 IRs are for ResizeBilenear
        reqStr = nnhal_raw_tensor_pb2.RequestString(token = my_token)
        self.detector.prepare(reqStr)
        self.detector.loadModel(reqStr)
        dts = [nnhal_raw_tensor_pb2.DataTensor(data=struct.pack('%sf' % len(img_data), *img_data), node_name="0", tensor_shape=[1,2,1,1], data_type=3)]
        reqDTs = nnhal_raw_tensor_pb2.RequestDataTensors(data_tensors=dts, token = my_token)
        rpyDTs = self.detector.getInferResult(reqDTs)
        self.assertEqual(len(rpyDTs.data_tensors), 0)

    def test_valid_infer_ResizeBilenear(self):
        log.info("TEST")
        img_data = [1.0, 1.0, 2.0, 2.0]
        expected_result = [1.0, 1.0, 1.0, 1.6666667461395264, 1.6666667461395264, 1.6666667461395264, 2.0, 2.0, 2.0]
        my_token = nnhal_raw_tensor_pb2.Token(data=1)
        reqStr = nnhal_raw_tensor_pb2.RequestString(token = my_token)
        self.detector.prepare(reqStr)
        self.detector.loadModel(reqStr)
        dts = [nnhal_raw_tensor_pb2.DataTensor(data=struct.pack('%sf' % len(img_data), *img_data), node_name="0", tensor_shape=[1,2,2,1], data_type=3)]
        reqDTs = nnhal_raw_tensor_pb2.RequestDataTensors(data_tensors=dts, token = my_token)
        rpyDTs = self.detector.getInferResult(reqDTs)
        self.assertEqual(len(rpyDTs.data_tensors), 1)
        datatensor = rpyDTs.data_tensors[0]
        img_data = datatensor.data
        resData = np.frombuffer(img_data, np.dtype(getMappedDatatype(3)))
        self.assertListEqual(expected_result, resData.tolist())

if __name__ == '__main__':
    log.basicConfig(format='%(funcName)s() %(message)s')
    log.root.setLevel(log.INFO)
    unittest.main()