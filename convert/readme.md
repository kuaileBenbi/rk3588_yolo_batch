1. 这个目录存放了在Linux上将```.onnx```转换为多batch推理的```convert.py```，包含yolov5和yolov8。

2. 转换模型直接从[瑞芯微官网](https://github.com/airockchip/rknn_model_zoo)下载的。

3. 转换代码参考的[官网](https://github.com/airockchip/rknn-toolkit2/tree/master/rknn-toolkit2/examples/functions/multi_batch) mobilenet多batch转换。

**关键修改**：
```ret = rknn.build(do_quantization=True, dataset='./dataset.txt', rknn_batch_size=4)```
