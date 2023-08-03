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
import logging as log
import shutil
import datetime
import threading

class ModelLoader:
    def __init__(self, model_name):
        self.loaded_flag = False
        self.DIR_PATH = ''
        self.XML_PATH = ''
        self.BIN_PATH = ''
        self.model_name = model_name

    def setModelDir(self, path):
        self.DIR_PATH = path
        self.XML_PATH = self.DIR_PATH + "/remote_model_" + self.model_name + ".xml"
        self.BIN_PATH = self.DIR_PATH + "/remote_model_" + self.model_name + ".bin"

    def prepareDir(self):
        self.cleanUp()

    def cleanUp(self):
        self.loaded_flag = False
        #cleaning if any previous model is loaded
        if os.path.isfile(self.XML_PATH):
            os.remove(self.XML_PATH)
        if os.path.isfile(self.BIN_PATH):
            os.remove(self.BIN_PATH)

    def saveXML(self, chunk):
        with open(self.XML_PATH, 'ab') as out_file:
            log.debug("xml chunk size {}".format(len(chunk.data)))
            out_file.write(chunk.data)
        log.info("saveXML for model {} ".format(self.model_name))

        return True

    def saveBin(self, chunk):
        with open(self.BIN_PATH, 'ab') as out_file:
            log.debug("bin chunk size {}".format(len(chunk.data)))
            out_file.write(chunk.data)
        log.info("saveBin for model {} ".format(self.model_name))

        return True

    def isModelLoaded(self, interface_obj, timeout_in_ms):
        #To check if model is already loaded
        if(self.loaded_flag):
            return True
        result = {self.model_name:False}
        t = threading.Thread(None, interface_obj.load_model, "LoadingModel", (self.XML_PATH, self.model_name, result))
        t.start()
        t.join(timeout=(timeout_in_ms/1000))
        if t.is_alive():
            log.warning("Model load timed out")
            self.loaded_flag = False
        elif result[self.model_name]:
            log.info("Model Loaded successfully")
            self.loaded_flag = True
        else:
            log.info("Model Load Failed")
        return self.loaded_flag
