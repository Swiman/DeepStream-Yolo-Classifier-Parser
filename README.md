# Seondary YOLOv8 Classifer for Deepstream 

### Convert model

#### 1. Install Ultralytics library alongside its dependencies

```
pip install ultralytics
pip install onnx onnxslim onnxruntime
```

#### 2. Generate onnx representation from raw .pt model file. 
#### NOTE: Only **STATIC** input dims is supported. For dynamic model shapes, refer to .
```
from ultralytics import YOLO
model = YOLO("<model>.pt", task="classify")
out = model.export(format="onnx", imgsz=640, dynamic=False, verbose=False, simplify=True, batch=32)
```
### Compile the lib

#### 3. Clone this repo inside Secondary_CarMake folder and compile
**NOTE**: Set the `CUDA_VER` inside Makefile according to your DeepStream version 

```
cd /opt/nvidia/deepstream/deepstream/samples/models/<desiredSecondaryFolder>
git clone http://192.168.42.35/zomar/yolov8-classifier-parser.git
make -C nvdsinfer_custom_impl_Yolo clean && make -C nvdsinfer_custom_impl_Yolo
```
### Set model info in config_infer_secondary_carmake_yolo.txt file

```
[property]

net-scale-factor=0.0039215697906911373
model-color-format=0
onnx-file=best.onnx
model-engine-file=best.engine
num-detected-classes=372
```

### Specify the custom parsing and engine creation functions

```
[property]

parse-classifier-func-name=NvDsInferClassicationParseYolo
custom-lib-path=nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so
engine-create-func-name=NvDsInferYoloCudaEngineGet
```

**NOTE**: The **YOLOv8** resizes the input with center padding. To get better accuracy, use

```
[property]

maintain-aspect-ratio=1
symmetric-padding=1
```

**NOTE**: Do Not specify a lables file name in config file. The file name (labels_with_type.txt) is hard-coded, so change its contents to your needs.
```
#labelfile-path=labels_with_type.txt
```


##

### Edit the main config file

```
[secondary-gie1]
config-file=config_infer_secondary_carmake_yolo.txt
```

**NOTE**: The TensorRT engine file may take a very long time to generate (sometimes more than 10 minutes).

**NOTE**: YOLO classifier output is 1xN vector, with N being the number of classes. The vector can be efficiently parsed with one for loop in cpp. Hence, the cuda parser for classification is not implemented!


**NOTE**: For dynamic model converstion refer to [`dynamicModel.md`](dynamicModel.md) file.
