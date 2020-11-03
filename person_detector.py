#!/usr/bin/python3

# This script is based on the file detect_image.py taken from https://github.com/google-coral/tflite/tree/master/python/examples/detection
# The original script mentioned above was modified so the image is read from a camera instead of a file.
# This script also filters the results such that only persons are detected and the resulting person count
# can be sent to a server using mqtt if the variable sendtoserver is set to 'True'.

# The script detect.py was taken from https://github.com/google-coral/tflite/tree/master/python/examples/detection and no modifications were made.

# The script uses the quantized model mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite which can be downloaded from https://github.com/google-coral/edgetpu/raw/master/test_data/
# The coco_labels.txt file is also available at https://github.com/google-coral/edgetpu/raw/master/test_data/

# Please, download the model and the labels and save them in a folder 'models/'

import sys
import platform
import time
import numpy as np
import tflite_runtime.interpreter as tflite
import detect
import paho.mqtt.client as mqtt
from picamera import PiCamera
from sense_hat import SenseHat

EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]

detection_threshold = 0.6
modelpath = 'models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'

sendtoserver = False
clientid = "raspberry1"
brokerip = "localhost"
brokerport = 1883


def load_labels(path, encoding='utf-8'):
  """Loads labels from file (with or without index numbers).

  Args:
    path: path to label file.
    encoding: label file encoding.
  Returns:
    Dictionary mapping indices to labels.
  """
  with open(path, 'r', encoding=encoding) as f:
    lines = f.readlines()
    if not lines:
      return {}

    if lines[0].split(' ', maxsplit=1)[0].isdigit():
      pairs = [line.split(' ', maxsplit=1) for line in lines]
      return {int(index): label.strip() for index, label in pairs}
    else:
      return {index: line.strip() for index, line in enumerate(lines)}


def make_interpreter(model_file):
  #model_file, *device = model_file.split('@')
  return tflite.Interpreter(
      model_path=model_file,
      experimental_delegates=[
          tflite.load_delegate(EDGETPU_SHARED_LIB)
      ])

def get_output(interpreter, score_threshold, image_scale=(1.0, 1.0)):
  """Returns list of detected objects."""
  boxes = output_tensor(interpreter, 0)
  class_ids = output_tensor(interpreter, 1)
  scores = output_tensor(interpreter, 2)
  count = int(output_tensor(interpreter, 3))

  width, height = input_size(interpreter)
  image_scale_x, image_scale_y = image_scale
  sx, sy = width / image_scale_x, height / image_scale_y

  def make(i):
    ymin, xmin, ymax, xmax = boxes[i]
    return Object(
        id=int(class_ids[i]),
        score=float(scores[i]),
        bbox=BBox(xmin=xmin,
                  ymin=ymin,
                  xmax=xmax,
                  ymax=ymax).scale(sx, sy).map(int))

  return [make(i) for i in range(count) if scores[i] >= score_threshold]

def classify(interpreter, input_index, image, labels):

    image = np.expand_dims(image, axis=0).astype(np.uint8)

    interpreter.set_tensor(input_index, image)

    interpreter.invoke()

    objs = detect.get_output(interpreter, detection_threshold)

    person_count = 0 # number of persons.

    if not objs:
        return(person_count)

    for obj in objs:
        label = labels.get(obj.id, obj.id)
        print(label + '  score: ', obj.score)
        if label == "person":
            person_count = person_count + 1

    return(person_count)


def main():

    sense = SenseHat()
    camera = PiCamera()
    camera.resolution = (288, 288)
    camera.start_preview()

    labels = load_labels('models/coco_labels.txt')
    #print(labels)
    interpreter = make_interpreter(modelpath)
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    # Print input image dim info.
    #print(interpreter.get_input_details()) # 300, 300, 3.

    # MQTT client
    if sendtoserver:
        client = mqtt.Client(clientid)
        client.connect(brokerip, brokerport)


    loop = True
    while loop:
        # Generate image with 0s
        tmpimg = np.zeros((300, 300, 3), dtype=np.uint8)

        camimg = np.empty((288, 288, 3), dtype=np.uint8)

        camera.capture(camimg, 'rgb')

        tmpimg[0:288, 0:288, 0] = camimg[0:288, 0:288, 0]
        tmpimg[0:288, 0:288, 1] = camimg[0:288, 0:288, 1]
        tmpimg[0:288, 0:288, 2] = camimg[0:288, 0:288, 2]

        person_count = classify(interpreter, input_index, tmpimg, labels)

        print("Persons count: " + str(person_count))

        sense.show_message(str(person_count))

        # Send result to broker.
        if sendtoserver: client.publish("personCount", clientid + "," + str(person_count))

        time.sleep(3)

        #loop = False # For testing run loop only once.

    camera.stop_preview()

    if sendtoserver: client.disconnect()


if __name__ == '__main__':
  main()
