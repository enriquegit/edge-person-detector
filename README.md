# Person-detector with Edge Computing and Deep Learning.

This is a prototype of a person detector using deep learning and the Google Coral Edge usb accelerator running on a Raspberry Pi and the [Sense HAT](https://www.raspberrypi.org/products/sense-hat/?resellerType=home).

This demo shows how to use [TensorFlowLite](https://www.tensorflow.org/lite) to implement a real-time person detector running on an Edge device. A pre-trained [mobilenet_ssd_v2_coco_quant_postprocess_edgetpu](https://github.com/google-coral/edgetpu/raw/master/test_data/) deep learning model was used for the detection part. The model is quantized and optimized to run in the Google Coral accelerator.

## Hardware

- Google Coral USB accelerator [link](https://coral.ai/products/accelerator)
- Sense HAT [link](https://www.raspberrypi.org/products/sense-hat/?resellerType=home)
- Raspberry Pi [link](https://www.raspberrypi.org/)
- Camera Module v2 [link](https://www.raspberrypi.org/products/camera-module-v2/?resellerType=home)


**Video DEMO**

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/PpFgAK_Yq1M/0.jpg)](https://www.youtube.com/watch?v=PpFgAK_Yq1M)
