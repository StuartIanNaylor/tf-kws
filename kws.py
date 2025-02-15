#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#from ai_edge_litert.interpreter import Interpreter
import tflite_runtime.interpreter as tflite
import sounddevice as sd
import numpy as np
import threading
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, help='Record device index --list_devices to list indexes')
parser.add_argument('--model_path', default='stream_state_external.tflite', help='tflite model path default=stream_state_external.tflite')
parser.add_argument('--window_stride', type=float, default=0.020, help='window_stride default=0.020')
parser.add_argument('--sample_rate', type=int, default=16000, help='Sample rate default=16000')
parser.add_argument('--kw_index', type=int, default=0, help='kw label index default=0')
parser.add_argument('--noise_index', type=int, default=1, help='noise label index default=1')
parser.add_argument('--kw_sensitivity', type=float, default=0.70, help='kw_sensitivity default=0.70')
parser.add_argument('--list_devices', help='list input devices', action="store_true")
parser.add_argument('--num_channels', type=int, default=1, help='Device channel number')
parser.add_argument('--noise_sensitivity', type=float, default=0.90, help='noise_sensitivity default=0.90')
args = parser.parse_args()
 
if args.list_devices:
 print(sd.query_devices())
 exit()
 
if args.device:
  sd.default.device = args.device
  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#interpreter = tf.lite.Interpreter(model_path=args.model_path)
interpreter = tflite.Interpreter(model_path=args.model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

inputs = []
for s in range(len(input_details)):
  inputs.append(np.zeros(input_details[s]['shape'], dtype=np.float32))
  
reset_state = False
last_argmax = 0
blocksize = int(args.sample_rate * args.window_stride)

def sd_callback(rec, frames, ftime, status):
    global reset_state, last_argmax
    if reset_state:
      for s in range(len(input_details)):
        inputs[s] = np.zeros(input_details[s]['shape'], dtype=np.float32)
      reset_state = False

    stream_update = np.reshape(rec, (1, blocksize))
    stream_update = stream_update.astype(np.float32)

    # set input audio data (by default input data at index 0)
    interpreter.set_tensor(input_details[0]['index'], stream_update)

    # set input states (index 1...)
    for s in range(1, len(input_details)):
      interpreter.set_tensor(input_details[s]['index'], inputs[s])

    # run inference
    interpreter.invoke()

    # get output: classification
    out_tflite = interpreter.get_tensor(output_details[0]['index'])

    # get output states and set it back to input states
    # which will be fed in the next inference cycle
    for s in range(1, len(input_details)):
      # The function `get_tensor()` returns a copy of the tensor data.
      # Use `tensor()` in order to get a pointer to the tensor.
      inputs[s] = interpreter.get_tensor(output_details[s]['index'])

    out_tflite_argmax = np.argmax(out_tflite)
    #0 kw, 1 falsekw, 2 notkw, 3 noise check labels.txt
    if out_tflite[0][out_tflite_argmax] > 0.90:
      if out_tflite_argmax == 0:
        reset_state = True
        print(out_tflite_argmax, out_tflite[0][out_tflite_argmax])


    
print("Loaded")
    
# Start streaming from microphone
with sd.InputStream(channels=args.num_channels,
                    samplerate=args.sample_rate,
                    blocksize=blocksize,
                    callback=sd_callback):
  threading.Event().wait()
