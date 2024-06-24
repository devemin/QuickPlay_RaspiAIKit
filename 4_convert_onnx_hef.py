

# 参考
# https://qiita.com/ysuito/items/a1cdd6e291aa4cb01ca6
# https://www.macnica.co.jp/business/semiconductor/articles/hailo/144988/
# https://www.macnica.co.jp/business/semiconductor/articles/hailo/145097/
# https://www.macnica.co.jp/business/semiconductor/articles/hailo/144843/


from hailo_sdk_client import ClientRunner
import numpy as np
#import os


# DataFlow Compiler Sample. (Ubuntu, Host PC, jupyter-notebook)
# 



onnx_model_name = 'fast_depth_pintosan'
onnx_path = '../models/fast_depth_128x160.onnx'
#onnx_path = '../models/yolov5s_best.onnx'

chosen_hw_arch = 'hailo8l'


input_height = 128
input_width = 160
input_ch = 3

runner = ClientRunner(hw_arch=chosen_hw_arch)
hn, npz = runner.translate_onnx_model(onnx_path, onnx_model_name,
                                      start_node_names=['input.1'],
                                      end_node_names=['Conv_135', 'Conv_24', 'Conv_114'],
                                      net_input_shapes={'input.1': [1, 3, 128, 160]})
#print(hn)
#print(npz)

hailo_model_har_name = f'{onnx_model_name}_hailo_model.har'
runner.save_har(hailo_model_har_name)

#from IPython.display import SVG
#!hailo visualizer {hailo_model_har_name} --no-browser
#SVG('yolov3.svg')

alls_lines = [
    'model_optimization_flavor(optimization_level=0, compression_level=1)\n',
    'resources_param(max_control_utilization=1.0, max_compute_utilization=1.0,max_memory_utilization=1.0)\n',
    'performance_param(fps=250)\n'
]

#Optimize
#TODO koreha random calib nanode, data de yaru
#https://www.macnica.co.jp/business/semiconductor/articles/hailo/144843/
calibData = np.random.randint(0, 255, (1024, input_height, input_width, input_ch))
runner.load_model_script(''.join(alls_lines))
runner.optimize(calibData)
quantized_model_har_path = f'{onnx_model_name}_quantized_model.har'
runner.save_har(quantized_model_har_path)

runner = ClientRunner(har=quantized_model_har_path)
#Compile
hef = runner.compile()
file_name = f'{onnx_model_name}_2.hef'
with open(file_name, 'wb') as f:
    f.write(hef)
compiled_model_har_path = f'{onnx_model_name}_compiled_model.har'
runner.save_har(compiled_model_har_path)
