import glob
import argparse
import sox
import os
import random
import numpy as np
import tflite_runtime.interpreter as tflite
import logging
import soundfile as sf
from playsound import playsound

if not os.path.exists("/tmp/libri"):
  os.makedirs("/tmp/libri")
if not os.path.exists("/tmp/kw"):
  os.makedirs("/tmp/kw")
if not os.path.exists("/tmp/demand"):
  os.makedirs("/tmp/demand")
if not os.path.exists("/tmp/bench"):
  os.makedirs("/tmp/bench")

logging.getLogger('sox').setLevel(logging.ERROR)


def benchmark(libri_dir, demand_dir, kw_dir, target_length, window_stride, sample_rate, kw_index, noise_index, noise_percent, noise_vol, min_vol, model_path):
  libri_files = glob.glob(libri_dir + '/**/*.flac', recursive=True)
  if not libri_files:
    print("No Libri files found", libri_dir)
  else:
    print(libri_files)
  demand_files = glob.glob(demand_dir + '/**/*.wav', recursive=True)
  if not demand_files:
    print("No Demand files found", demand_dir)
  else:
    print(demand_files)
  kw_files = glob.glob(kw_dir + '/*.wav')
  if not kw_files:
    print("No KW files found", kw_dir)
  else:
    print(kw_files)
    
  kw_count = 0
  kw_total = len(kw_files) -1
  demand_count = 0
  demand_total = len(demand_files) -1  
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

  #interpreter = tf.lite.Interpreter(model_path=model_path)
  interpreter = tflite.Interpreter(model_path=model_path)
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  inputs = []
  for s in range(len(input_details)):
    inputs.append(np.zeros(input_details[s]['shape'], dtype=np.float32))
  
  hit_count = 0
  libri_count = 0
  miss_count = 0
  false_hit = 0
  total_length = 0.0
  blocksize = int(sample_rate * window_stride * 3) # 25 rolling windows per 1 sec of 1600khz
  non_stream_blocksize = int(target_length * sample_rate)
  window_count = int((target_length * sample_rate) / blocksize)
  
  for libri in libri_files:
  
    libri_count += 1
  
    tfm1 = sox.Transformer()
    tfm1.clear_effects()
    tfm1.norm(-0.7)
    libri_file = '/tmp/libri/' + os.path.splitext(os.path.basename(libri))[0] + '.wav'
    tfm1.build_file(libri, libri_file)
  
    tfm2 = sox.Transformer()
    tfm2.clear_effects()
    tfm2.pad(0.0, 1.5)
    tfm2.norm(-0.7)
    kw_file = '/tmp/kw/' + os.path.splitext(os.path.basename(kw_files[kw_count]))[0] + '.wav'
    tfm2.build_file(kw_files[kw_count], kw_file)
    kw_count += 1
    if kw_count > kw_total:
      kw_count = 0
      
    cbn1 = sox.Combiner()
    cbn1.set_input_format(["wav", "wav"])
    cbn1.build([libri_file, kw_file], '/tmp/output.wav', 'concatenate')
    output_length = sox.file_info.duration('/tmp/output.wav') 
    offset = (sox.file_info.duration(demand_files[demand_count]) - output_length) * random.random()
    total_length += output_length
    #print(output_length, sox.file_info.duration(demand_files[demand_count]), offset)

    tfm3 = sox.Transformer()
    tfm3.clear_effects()
    tfm3.norm(-0.7)
    tfm3.trim(offset, offset + output_length)
    demand_file = '/tmp/demand/' + os.path.splitext(os.path.basename(demand_files[demand_count]))[0] + '.wav'
    #print(demand_files[demand_count], demand_file)
    tfm3.build_file(demand_files[demand_count], demand_file)
    demand_count += 1
    if demand_count > demand_total:
      demand_count = 0
      
    wav_vol = 1.0 - ((1.0 - min_vol) * random.random())
    noise_lvl = noise_vol * random.random()
    cbn2 = sox.Combiner()
    cbn2.set_input_format(["wav", "wav"])
    cbn2.build(['/tmp/output.wav', demand_file], '/tmp/bench/bench.wav', 'mix', [wav_vol, noise_lvl])
    
    tfm4 = sox.Transformer()
    tfm4.clear_effects()
    tfm4.norm(-0.7)
    bench_wav, samplerate = sf.read('/tmp/bench/bench.wav',dtype='float32')
    bench_len = len(bench_wav)
    #print(bench_len)
    pos = 0
    libri_hit = 0
    lastpos = 0
    reset_state = False
    reset_count = 0
    max_hit = 0
    while True:

      if reset_state:
        for s in range(len(input_details)):
          inputs[s] = np.zeros(input_details[s]['shape'], dtype=np.float32)
        reset_state = False

      if pos + non_stream_blocksize > bench_len:
        break
      if reset_count > 0:
        reset_count -= 1
        rec = np.random.rand(1, non_stream_blocksize)
        rec = rec.astype(np.float32)
      else:
        rec = bench_wav[pos:pos + non_stream_blocksize]
      stream_update = np.reshape(rec, (1, non_stream_blocksize))
       
      
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
      
      out_tflite_argmax = np.argmax(out_tflite[0])
      #0 kw, 1 falsekw, 2 notkw, 3 phonekw, 4 noise check labels.txt
      if out_tflite_argmax == 0 and reset_count == 0:
        if max_hit < out_tflite[0][out_tflite_argmax]:
          max_hit = out_tflite[0][out_tflite_argmax]
        if out_tflite[0][out_tflite_argmax] > 0.99:
          reset_state = True
          reset_count = window_count
          print(out_tflite[0][out_tflite_argmax], pos - lastpos, pos, libri_hit)
          lastpos = pos
          hit_count += 1
          libri_hit += 1

      pos += blocksize
      
    print(max_hit, libri_count, hit_count, false_hit, miss_count, total_length / 360)
    if libri_hit > 1:
      playsound('/tmp/bench/bench.wav')
      false_hit += libri_hit - 1
    elif libri_hit == 0:
      print("No Hit !!!")
      miss_count += 1
      playsound('/tmp/bench/bench.wav')
      
    
    lastpos = 0
     
  ten_hour = total_length / 10
  print("Missed per 10 hour=", miss_count / ten_hour, "False%=", (false_hit / libri_count) * 100, "False% 10hr=", ((false_hit / libri_count) * 100) / ten_hour)
    

def main_body():
  parser = argparse.ArgumentParser()
  parser.add_argument('--libri_dir', default='./LibriSpeech/test-clean', help='source dir location default=./LibriSpeech/test-clean')
  parser.add_argument('--demand_dir', type=str, default='./demand', help='dest dir location default=./demand')
  parser.add_argument('--kw_dir', type=str, default='./kw/hey jarvis', help='dest dir location default=./kw/hey jarvis')
  parser.add_argument('--target_length', type=float, default=1.5, help='Minimum trimmed length default=1.5s')
  parser.add_argument('--window_stride', type=float, default=0.020, help='window_stride default=0.020')
  parser.add_argument('--sample_rate', type=int, default=16000, help='Sample rate default=16000')
  parser.add_argument('--kw_index', type=int, default=0, help='kw label index default=0')
  parser.add_argument('--noise_index', type=int, default=4, help='noise label index default=4')
  parser.add_argument('--kw_sensitivity', type=float, default=0.70, help='kw_sensitivity default=0.70')
  parser.add_argument('--noise_sensitivity', type=float, default=0.90, help='noise_sensitivity default=0.90')
  parser.add_argument('--noise_percent', type=float, default=0.9, help='Percent of KW to add noise to (0.9)')
  parser.add_argument('--noise_vol', type=float, default=0.3, help='Max Vol of noise background mix (0.3)')
  parser.add_argument('--min_vol', type=float, default=0.7, help='Min Vol of foreground (0.7)')
  parser.add_argument('--model_path', default='non_stream.tflite', help='tflite model path default=non_stream.tflite')
  args = parser.parse_args()
 
  benchmark(args.libri_dir, args.demand_dir, args.kw_dir, args.target_length, args.window_stride, args.sample_rate, args.kw_index, args.noise_index, args.noise_percent, args.noise_vol, args.min_vol, args.model_path)

    
if __name__ == '__main__':
  main_body()

