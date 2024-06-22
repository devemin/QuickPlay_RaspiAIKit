import numpy as np
from multiprocessing import Process
from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams,
    InputVStreamParams, OutputVStreamParams, InputVStreams, OutputVStreams, FormatType, HailoSchedulingAlgorithm)


import copy

# Define the function to run inference on the model
def infer(network_group, input_vstreams_params, output_vstreams_params, input_data):
    rep_count = 3
    with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
        for i in range(rep_count):
            #outdata = copy.deepcopy(infer_pipeline)
            infer_results = infer_pipeline.infer(input_data)
            print(infer_results.keys())


# Loading compiled HEFs:
first_hef_path = '/home/pi/playground/hailo-rpi5-examples/resources/yolov8s_h8l.hef'
second_hef_path = '/home/pi/playground/hailo-rpi5-examples/resources/yolov5n_seg_h8l_mz.hef'
first_hef = HEF(first_hef_path)
second_hef = HEF(second_hef_path)
hefs = [first_hef, second_hef]

# Creating the VDevice target with scheduler enabled
params = VDevice.create_params()
params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
with VDevice(params) as target:
    infer_processes = []

    # Configure network groups
    for hef in hefs:
        configure_params = ConfigureParams.create_from_hef(hef=hef, interface=HailoStreamInterface.PCIe)
        network_groups = target.configure(hef, configure_params)
        network_group = network_groups[0]

        # Create input and output virtual streams params
        input_vstreams_params = InputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
        output_vstreams_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)

        # Define dataset params
        input_vstream_info = hef.get_input_vstream_infos()[0]
        image_height, image_width, channels = input_vstream_info.shape
        num_of_frames = 10
        low, high = 2, 20

        # Generate random dataset
        dataset = np.random.randint(low, high, (num_of_frames, image_height, image_width, channels)).astype(np.float32)
        input_data = {input_vstream_info.name: dataset}

        # Create infer process
        infer_process = Process(target=infer, args=(network_group, input_vstreams_params, output_vstreams_params, input_data))
        infer_processes.append(infer_process)

    print(f'Starting streaming on multiple models using scheduler')
    for infer_process in infer_processes:
        infer_process.start()
    for infer_process in infer_processes:
        infer_process.join()

    print('Done inference')
