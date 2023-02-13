# AI Dispatcher

## 1. Introduction
AI Dispatcher is a reference solution that provide a list of service-implementations and adaptor-implementations to perform platform-agnostic AI-inference on the given input, using the available underlying Inference Engine.​


## 2. Major functionalities :​

- Accept inputs via gRPC​

- Perform required data conversions on the input​

- Use one of the adaptors** for inference​
    - OpenVINO™ Model Server(OVMS)
    - OpenVino Toolkit(OVTK)

- Format the output and return the result​


## 3. Architecture
<p align="center">
<img src="common/dispatcher_architecture.png"  width="80%" height="40%">
</p>

## 4. Compile and Run 
### 4.1. Pre-requisites
Make sure whichever adaptor you want to use is pre-installed and running

- For openvino toolkit its recommeded to install version 2022.3 or latest

### 4.2. Steps to run

#### 4.2.1 General Setup Steps:
```bash
git clone https://github.com/intel-sandbox/ai-dispatcher.git
cd ai-dispatcher
export PYTHONPATH="$PWD"
python3 -m pip install -r client_requirements.txt
```

#### 4.2.2 Steps For Running Object Detection Service
```bash
# generate proto files
cd services/objectDetection && python3 -m grpc_tools.protoc -I . --python_out=. --grpc_python_out=. *.proto
```

**To start object detection service with ovms**
```bash
#to use ovms adaptor, start ovms server

docker run -d -v $PWD/model/1:/models/model_od/1 -e LOG_LEVEL=DEBUG -p 9000:9000 openvino/ubuntu18_model_server /ie-serving-py/start_server.sh ie_serving model --model_path /models/model_od --model_name model_od --port 9000

# start the service
python3 objectDetection.py --serving_mounted_modelDir $(pwd)/model/ --remote_port 50051 --interface ovms
```
**To start object detection service with ovtk**
```bash
source <open_vino_install_path>/setupenv.sh
python3 objectDetection.py --serving_mounted_modelDir model/ --remote_port 50051 --interface ovtk
```
#### 4.2.3 Steps For Running Raw Tensor service
```bash
# generate proto file
cd services/rawTensor && python3 -m grpc_tools.protoc -I . --python_out=. --grpc_python_out=. *.proto

# create directory where models will be stored
mkdir test_model_name
```

**To start raw service with ovms**
```bash
#to use ovms adaptor, start ovms server
docker run -d -v $(pwd)/test_model_name:/models/remote_model -e LOG_LEVEL=DEBUG -p 9008:9008 openvino/ubuntu18_model_server /ie-serving-py/start_server.sh ie_serving model --model_path /models/remote_model --model_name remote_model --port 9008

#start rawTensor service
python3 rawTensor.py --serving_mounted_modelDir $(pwd)/test_model_name/ --serving_port 9008 --interface ovms
```

**To run  rawTensorservice with ovtk**
```bash
source <open_vino_install_path>/setupenv.sh

python3 rawTensor.py --serving_mounted_modelDir test_model_name/ --interface ovtk --unix_socket ~/ipc/ai.socket

## if you want to pass specific device to be used for inferencing use --device GPU.1 or CPU
```
##### 4.2.4 Steps For Face Mask Detection service:
```bash
#genrate proto
cd services/faceMaskDetection && python3 -m grpc_tools.protoc -I . --python_out=. --grpc_python_out=. *.proto
```

**To start with ovms**
```bash
#start ovms server

docker run -d -v $(pwd)/model/1:/models/face_mask_detection/1 -e LOG_LEVEL=DEBUG -p 9000:9000 openvino/ubuntu18_model_server /ie-serving-py/start_server.sh ie_serving model --model_path /models/face_mask_detection --model_name face_mask_detection --port 9000 --shape auto

#start service
python3 faceMaskDetection.py --width 260 --height 260 --serving_mounted_modelDir $(pwd)/model/ --interface ovms
```

**To start with ovtk**
```bash
source <open_vino_install_path>/setupenv.sh

python3 faceMaskDetection.py --width 260 --height 260 --serving_mounted_modelDir model/ --interface ovtk
```