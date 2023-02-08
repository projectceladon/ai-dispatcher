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
import ipaddress

def validate(args):
    assert os.path.exists(args['serving_mounted_modelDir']), \
        "Invalid path provided to serving_mounted_modelDir: " \
        + args['serving_mounted_modelDir']
    if(args['serving_address'] != "localhost"):
        ipaddress.ip_address(args['serving_address'])
    assert (1 <= int(args['serving_port']) <= 65535 ), "Invalid serving_port provided: " \
        + args['serving_port']
    assert args['interface'] in ["ovms", "ovtk"], "Invalid interface provided: " \
        + args['interface']
    #serving_model_name not validated
    assert args['device'] in ["CPU", "AUTO", "GPU", "GPU.0", "GPU.1"], "Invalid device provided: " \
        + args['device']
    assert (1 <= int(args['remote_port']) <= 65535 ), "Invalid remote_port provided: " \
        + args['remote_port']
    if(args['unix_socket'] != ""):
        assert os.path.exists(os.path.dirname(os.path.abspath(args['unix_socket']))), \
            "Invalid path provided to unix_socket: " + args['unix_socket']
    print("Common Validtion Complete")
