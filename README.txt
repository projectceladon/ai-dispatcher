Copyright (c) 2022 Intel Corporation

General Setup Steps:
i)cd ai_proxy
ii)export PYTHONPATH="$PWD"

Steps For Runing Object Detection Service:

TO RUN WITH OVMS:
iii)cd services/objectDetection
iv)docker run -d -v $PWD/model/1:/models/model_od/1 -e LOG_LEVEL=DEBUG -p 9000:9000 openvino/ubuntu18_model_server /ie-serving-py/start_server.sh ie_serving model --model_path /models/model_od --model_name model_od --port 9000
v)python3 objectDetection.py --serving_mounted_modelDir $(pwd)/model/ --remote_port 50059 --interface ovms

TO RUN WITH OVTK:
iv)python objectDetection.py --serving_mounted_modelDir model/ --remote_port 50059 --interface ovtk

Steps For Runing Raw Tensor service:

TO RUN WITH OVMS:
iii)cd services/rawTensor
iv)docker run -d -v $(pwd)/test_model_name:/models/remote_model -e LOG_LEVEL=DEBUG -p 9008:9008 openvino/ubuntu18_model_server /ie-serving-py/start_server.sh ie_serving model --model_path /models/remote_model --model_name remote_model --port 9008
v)python3 rawTensor.py --serving_mounted_modelDir $(pwd)/test_model_name/ --serving_port 9008 --interface ovms

TO RUN WITH OVTK:
iv) python rawTensor.py --serving_mounted_modelDir test_model_name/ --serving_port 9008 --interface ovtk

Steps For Face Mask Detection service:

TO RUN WITH OVMS
iii)cd services/faceMaskDetection
iv)docker run -d -v $(pwd)/model/1:/models/face_mask_detection/1 -e LOG_LEVEL=DEBUG -p 9000:9000 openvino/ubuntu18_model_server /ie-serving-py/start_server.sh ie_serving model --model_path /models/face_mask_detection --model_name face_mask_detection --port 9000 --shape auto
v)python3 faceMaskDetection.py --width 260 --height 260 --serving_mounted_modelDir $(pwd)/model/ --interface ovms

TO RUN WITH OVTK
iv)python faceMaskDetection.py --width 260 --height 260 --serving_mounted_modelDir model/ --interface ovtk


