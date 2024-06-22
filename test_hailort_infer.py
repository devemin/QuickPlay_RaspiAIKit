#!/usr/bin/env python3

# This sample is from Hailo-Application-Code-Examples/runtime/python/yolox_streaming_inference/yolox_stream_inference.py



# HEF, hailort, user guide, p245 (PDF p249)
# 



# python obj kakunin https://zenn.dev/ynakashi/articles/15b2b7c0a3cd89


import cv2
import os, random, time
import numpy as np
from multiprocessing import Process
#import yolox_stream_report_detections as report
from hailo_platform import (HEF, Device, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams,
    InputVStreamParams, OutputVStreamParams, InputVStreams, OutputVStreams, FormatType)


#from hailo_platform.pyhailort.pyhailort import (Control, InternalPcieDevice, ExceptionWrapper, BoardInformation,  # noqa F401
#                                                CoreInformation, DeviceArchitectureTypes, ExtendedDeviceInformation,  # noqa F401
#                                                HealthInformation, SamplingPeriod, AveragingFactor, DvmTypes, # noqa F401
#                                                PowerMeasurementTypes, MeasurementBufferIndex) # noqa F401
#import hailo_platform.pyhailort._pyhailort as _pyhailort



# yolox_s_leaky input resolution
INPUT_RES_H = 640
INPUT_RES_W = 640

# Loading compiled HEFs to device:
model_name = 'yolox_s_leaky_h8l_mz'
#model_name = 'yolov8s_pose_h8l_pi'
#model_name = 'yolov8s_h8l'
#model_name = 'yolov5n_seg_h8l_mz'
#hef_path =  '/home/pi/playground/Hailo-Application-Code-Examples/runtime/python/yolox_streaming_inference/resources/hefs/{}.hef'.format(model_name)
hef_path = '/home/pi/playground/hailo-rpi5-examples/resources/{}.hef'.format(model_name)
video_dir = '/home/pi/playground/Hailo-Application-Code-Examples/runtime/python/yolox_streaming_inference/resources/video/'
hef = HEF(hef_path)
mp4_files = [f for f in os.listdir(video_dir) if os.path.isfile(os.path.join(video_dir, f)) and f.endswith('.mp4')]

devices = Device.scan()

with VDevice(device_ids=devices) as target:
        configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        network_group = target.configure(hef, configure_params)[0]
        network_group_params = network_group.create_params()
        input_vstream_info = hef.get_input_vstream_infos()[0]
        output_vstream_info = hef.get_output_vstream_infos()[0]
        input_vstreams_params = InputVStreamParams.make_from_network_group(network_group, quantized=False, format_type=FormatType.FLOAT32)
        output_vstreams_params = OutputVStreamParams.make_from_network_group(network_group, quantized=False, format_type=FormatType.FLOAT32)
        height, width, channels = hef.get_input_vstream_infos()[0].shape


        def ShowVstreamInfo(vsinfo, nmsflag=False):
            print("======================= VstreamInfo start.========================")
            print(vsinfo)
            #print(dir(vsinfo))
            print(vsinfo.direction)
            print(vsinfo.format)
            #print(dir(vsinfo.format))
            print(vsinfo.format.equals)
            print(vsinfo.format.flags)
            print(vsinfo.format.order)
            print(vsinfo.format.type)
            print(vsinfo.name)
            print(vsinfo.network_name)
            if (nmsflag):
                print(vsinfo.nms_shape)
            print(vsinfo.quant_info)
            #print(dir(vsinfo.quant_info))
            print(vsinfo.quant_info.limvals_max)
            print(vsinfo.quant_info.limvals_min)
            print(vsinfo.quant_info.qp_scale)
            print(vsinfo.quant_info.qp_zp)
            print(vsinfo.shape)
            print("======================= VstreamInfo end.   ========================")

        def ShowStreamInfo(sinfo, nmsflag=False):
            print("------------------  StreamInfo start.  --------------------------")
            print(sinfo)
            #print(dir(sinfo))
            print(sinfo.data_bytes)
            print(sinfo.direction)
            print(sinfo.format)
            #print(dir(sinfo.format))
            print(sinfo.name)
            if (nmsflag):
                print(sinfo.nms_shape)
            print(sinfo.quant_info)
            #print(dir(sinfo.quant_info))
            print(sinfo.quant_info.limvals_max)
            print(sinfo.quant_info.limvals_min)
            print(sinfo.quant_info.qp_scale)
            print(sinfo.quant_info.qp_zp)
            print(sinfo.shape)
            print(sinfo.sys_index)
            print("------------------  StreamInfo end.    --------------------------")


        print("++++++++++++++++++++++++++++++++++++++++")
        print('hef')
        print(hef)
        print(dir(hef))
        print(hef._hef)
        print(dir(hef._hef))
        #hef.get_hef_device_arch()    # for C++,  not for Python. HailoRT User Guide p193. alternatively, using hailortcli parse-hef ***.hef ? or check error at loading the hef ?
        
        print(hef.bottleneck_fps())
        print(hef.get_all_stream_infos())
        print(hef.get_all_vstream_infos())
        print(hef.get_input_stream_infos())
        print(hef.get_input_vstream_infos())
        print(hef.get_network_group_names())
        print(hef.get_network_groups_infos())
        print(dir(hef.get_network_groups_infos()[0]))
        print(hef.get_network_groups_infos()[0].is_multi_context)
        print(hef.get_networks_names())
        #print(hef.get_original_names_from_vstream_name())
        print(hef.get_output_stream_infos())
        print(hef.get_output_vstream_infos())
        print(hef.get_sorted_output_names())
        #print(hef.get_stream_names_from_vstream_name())
        #print(hef.get_udp_rates_dict())
        #print(hef.get_vstream_name_from_original_name())
        #print(hef.get_vstream_names_from_stream_name())
        print(hef.path)

        print("++++++++++++++++++++++++++++++++++++++++")
        print('configure_params')
        print(configure_params)                                  #{'yolox_s_leaky': <hailo_platform.pyhailort._pyhailort.ConfigureParams object at 0x7fff827a01f0>}
        print(dir(configure_params['yolox_s_leaky']))
        print(configure_params['yolox_s_leaky'].batch_size)
        print(configure_params['yolox_s_leaky'].network_params_by_name)
        print(configure_params['yolox_s_leaky'].power_mode)
        print(configure_params['yolox_s_leaky'].stream_params_by_name)

        print("++++++++++++++++++++++++++++++++++++++++")
        print('network_group')
        print(network_group)                                     #<hailo_platform.pyhailort.pyhailort.ConfiguredNetwork object at 0x7fff827f62c0>
        print(dir(network_group))
        print(network_group.name)
        #print(network_group.get_networks_names)
        #print(dir(network_group.get_networks_names))
        # 'activate', 
        # 'create_params', 
        # 'get_all_stream_infos', 
        # 'get_all_vstream_infos', 
        # 'get_input_stream_infos', 
        # 'get_input_vstream_infos', 
        # 'get_networks_names', 
        # 'get_output_shapes', 
        # 'get_output_stream_infos', 
        # 'get_output_vstream_infos', 
        # 'get_sorted_output_names', 
        # 'get_stream_names_from_vstream_name', 
        # 'get_udp_rates_dict', 
        # 'get_vstream_names_from_stream_name', 
        # 'name', 
        # 'set_scheduler_priority', 
        # 'set_scheduler_threshold', 
        # 'set_scheduler_timeout', 
        # 'wait_for_activation']



        print("++++++++++++++++++++++++++++++++++++++++")
        print('network_group_params')
        print(network_group_params)                              #<hailo_platform.pyhailort._pyhailort.ActivateNetworkGroupParams object at 0x7fff827bc970>
        print(dir(network_group_params))

        print("++++++++++++++++++++++++++++++++++++++++")
        print('input_vstream_info')
        ShowVstreamInfo(input_vstream_info)

        print("++++++++++++++++++++++++++++++++++++++++")
        print('output_vstream_info')
        ShowVstreamInfo(output_vstream_info, True)

        print("++++++++++++++++++++++++++++++++++++++++")
        print('input_vstreams_params')
        print(input_vstreams_params)                             #{'yolox_s_leaky/input_layer1': <hailo_platform.pyhailort._pyhailort.VStreamParams object at 0x7fff8278edf0>}
        print(dir(input_vstreams_params['yolox_s_leaky/input_layer1']))
        print(input_vstreams_params['yolox_s_leaky/input_layer1'].pipeline_elements_stats_flags)
        print(input_vstreams_params['yolox_s_leaky/input_layer1'].queue_size)
        print(input_vstreams_params['yolox_s_leaky/input_layer1'].timeout_ms)
        print(input_vstreams_params['yolox_s_leaky/input_layer1'].user_buffer_format)
        print(dir(input_vstreams_params['yolox_s_leaky/input_layer1'].user_buffer_format))
        print(input_vstreams_params['yolox_s_leaky/input_layer1'].user_buffer_format.equals)
        print(input_vstreams_params['yolox_s_leaky/input_layer1'].user_buffer_format.flags)
        print(input_vstreams_params['yolox_s_leaky/input_layer1'].user_buffer_format.order)
        print(input_vstreams_params['yolox_s_leaky/input_layer1'].user_buffer_format.type)
        print(input_vstreams_params['yolox_s_leaky/input_layer1'].vstream_stats_flags)

        print("++++++++++++++++++++++++++++++++++++++++")
        print('output_vstreams_params')
        print(output_vstreams_params)                            #{'yolox_s_leaky/yolox_nms_postprocess': <hailo_platform.pyhailort._pyhailort.VStreamParams object at 0x7fff8278f4f0>}
        print(dir(output_vstreams_params['yolox_s_leaky/yolox_nms_postprocess']))
        print(output_vstreams_params['yolox_s_leaky/yolox_nms_postprocess'].pipeline_elements_stats_flags)
        print(output_vstreams_params['yolox_s_leaky/yolox_nms_postprocess'].queue_size)
        print(output_vstreams_params['yolox_s_leaky/yolox_nms_postprocess'].timeout_ms)
        print(output_vstreams_params['yolox_s_leaky/yolox_nms_postprocess'].user_buffer_format)
        print(dir(output_vstreams_params['yolox_s_leaky/yolox_nms_postprocess'].user_buffer_format))
        print(output_vstreams_params['yolox_s_leaky/yolox_nms_postprocess'].user_buffer_format.equals)
        print(output_vstreams_params['yolox_s_leaky/yolox_nms_postprocess'].user_buffer_format.flags)
        print(output_vstreams_params['yolox_s_leaky/yolox_nms_postprocess'].user_buffer_format.order)
        print(output_vstreams_params['yolox_s_leaky/yolox_nms_postprocess'].user_buffer_format.type)
        print(output_vstreams_params['yolox_s_leaky/yolox_nms_postprocess'].vstream_stats_flags)

        print("++++++++++++++++++++++++++++++++++++++++")
        print('hef.get_output_vstream_infos()[0].shape') 
        print(hef.get_output_vstream_infos()[0].shape)           #(80, 5, 100)
        print(hef.get_output_vstream_infos()[0])                 #VStreamInfo("yolox_s_leaky/yolox_nms_postprocess")
        ShowVstreamInfo(hef.get_output_vstream_infos()[0])


        print("++++++++++++++++++++++++++++++++++++++++")
        print("++++++++++++++++++++++++++++++++++++++++")
        print("++++++++++++++++++++++++++++++++++++++++")
        print('hef.get_networks_names()')
        print(hef.get_networks_names())                          #['yolox_s_leaky/yolox_s_leaky']
        print("++++++++++++++++++++++++++++++++++++++++")
        print('hef.get_network_group_names()')
        print(hef.get_network_group_names())                     #['yolox_s_leaky']

        print("++++++++++++++++++++++++++++++++++++++++")
        print('hef.get_network_groups_infos()')
        print(hef.get_network_groups_infos())                    #[<hailo_platform.pyhailort._pyhailort.NetworkGroupInfo object at 0x7ffedcccfef0>]
        print(dir(hef.get_network_groups_infos()[0]))
        print(hef.get_network_groups_infos()[0].name)
        print("++++++++++++++++++++++++++++++++++++++++")
        print('hef.get_input_vstream_infos()')
        print(hef.get_input_vstream_infos())                     #[VStreamInfo("yolox_s_leaky/input_layer1")]

        print("++++++++++++++++++++++++++++++++++++++++")
        print('hef.get_output_vstream_infos()')
        print(hef.get_output_vstream_infos())                    #[VStreamInfo("yolox_s_leaky/yolox_nms_postprocess")]

        print("++++++++++++++++++++++++++++++++++++++++")
        print('hef.get_all_vstream_infos()')
        print(hef.get_all_vstream_infos())                       #[VStreamInfo("yolox_s_leaky/input_layer1"), VStreamInfo("yolox_s_leaky/yolox_nms_postprocess")]

        print("++++++++++++++++++++++++++++++++++++++++")
        print('hef.get_input_stream_infos()')
        print(hef.get_input_stream_infos())                      #[StreamInfo("yolox_s_leaky/input_layer1")]
        ShowStreamInfo(hef.get_input_stream_infos()[0])

        print("++++++++++++++++++++++++++++++++++++++++")
        
        print("++++++++++++++++++++++++++++++++++++++++")
        print('hef.get_output_stream_infos()')
        print(hef.get_output_stream_infos())                     #[StreamInfo("yolox_s_leaky/conv56_111"), StreamInfo("yolox_s_leaky/conv55_111"), StreamInfo("yolox_s_leaky/conv54_111"), StreamInfo("yolox_s_leaky/conv68_111"), StreamInfo("yolox_s_leaky/conv70_111"), StreamInfo("yolox_s_leaky/conv69_111"), StreamInfo("yolox_s_leaky/conv83_111"), StreamInfo("yolox_s_leaky/conv82_111"), StreamInfo("yolox_s_leaky/conv81_111")]

        print("++++++++++++++++++++++++++++++++++++++++")
        print('hef.get_all_stream_infos()')
        print(hef.get_all_stream_infos())                        #[StreamInfo("yolox_s_leaky/input_layer1"), StreamInfo("yolox_s_leaky/conv56_111"), StreamInfo("yolox_s_leaky/conv55_111"), StreamInfo("yolox_s_leaky/conv54_111"), StreamInfo("yolox_s_leaky/conv68_111"), StreamInfo("yolox_s_leaky/conv70_111"), StreamInfo("yolox_s_leaky/conv69_111"), StreamInfo("yolox_s_leaky/conv83_111"), StreamInfo("yolox_s_leaky/conv82_111"), StreamInfo("yolox_s_leaky/conv81_111")]

        print("++++++++++++++++++++++++++++++++++++++++")
        print('hef.get_sorted_output_names()')
        print(hef.get_sorted_output_names())                     #['yolox_s_leaky/yolox_nms_postprocess']

        print("++++++++++++++++++++++++++++++++++++++++")
        print('hef.bottleneck_fps()')
        print(hef.bottleneck_fps())                              #393.7697746258695

        print("++++++++++++++++++++++++++++++++++++++++")

        
        
        time.sleep(3)
        
        source = 'camera'
        cap = cv2.VideoCapture(0)

        # check if the camera was opened successfully
        if not cap.isOpened():
            print("Could not open camera")
            exit()

        while True:
            # read a frame from the video source
            ret, frame = cap.read()

            # Get height and width from capture
            orig_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  
            orig_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)        

            # check if the frame was successfully read
            if not ret:
                print("Could not read frame")
                break

            # loop if video source
            if source == 'video' and not cap.get(cv2.CAP_PROP_POS_FRAMES) % cap.get(cv2.CAP_PROP_FRAME_COUNT):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # resize image for yolox_s_leaky input resolution and infer it
            resized_img = cv2.resize(frame, (INPUT_RES_H, INPUT_RES_W), interpolation = cv2.INTER_AREA)
            with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
                input_data = {input_vstream_info.name: np.expand_dims(np.asarray(resized_img), axis=0).astype(np.float32)}    
                with network_group.activate(network_group_params):
                    infer_results = infer_pipeline.infer(input_data)
            # create dictionary that returns layer name from tensor shape (required for postprocessing)
            #layer_from_shape: dict = {infer_results[key].shape:key for key in infer_results.keys()}            

            print("1================================")
            #print(len(infer_results))
            print("2================================")
            #print(list(infer_results.keys()))
            print(infer_results)
            print("3================================")
            print(list(infer_results.values()))
            print("4================================")
            print(len(list(infer_results.values())[0] )    )
            print(len(list(infer_results.values())[0][0] )    )
            print(len(list(infer_results.values())[0][0][0] )    )
            print(len(list(infer_results.values())[0][0][0][0] )    )
            #print(len(list(infer_results.values())[0][0][0][0][0] )    )
            print("5================================")

            #for item in infer_results:
            #    print("#############################################################")
            #    print(item)
            #    print("#############################################################")
            #print(infer_results)
            #print(layer_from_shape)
            #print(layer_from_shape.keys())
            #print(layer_from_shape.values())
            #anchors = {"strides": [32, 16, 8], "sizes": [[1, 1], [1, 1], [1, 1]]}
            #print(anchors)
            
            '''
            endnodes = [infer_results[layer_from_shape[1, 80, 80, 4]],  # stride 8 
                        infer_results[layer_from_shape[1, 80, 80, 1]],  # stride 8 
                        infer_results[layer_from_shape[1, 80, 80, 80]], # stride 8 
                        infer_results[layer_from_shape[1, 40, 40, 4]],  # stride 16
                        infer_results[layer_from_shape[1, 40, 40, 1]],  # stride 16
                        infer_results[layer_from_shape[1, 40, 40, 80]], # stride 16
                        infer_results[layer_from_shape[1, 20, 20, 4]],  # stride 32
                        infer_results[layer_from_shape[1, 20, 20, 1]],  # stride 32
                        infer_results[layer_from_shape[1, 20, 20, 80]]  # stride 32
                    ]
            '''
            #print(endnodes)
            
            #time.sleep(10)


            '''
            from hailo_model_zoo.core.postprocessing.detection import yolo
            # postprocessing info for constructor as recommended in hailo_model_zoo/cfg/base/yolox.yaml
            anchors = {"strides": [32, 16, 8], "sizes": [[1, 1], [1, 1], [1, 1]]}
            yolox_post_proc = yolo.YoloPostProc(img_dims=(INPUT_RES_H,INPUT_RES_W), nms_iou_thresh=0.65, score_threshold=0.01, 
                                                anchors=anchors, output_scheme=None, classes=80, labels_offset=1, 
                                                meta_arch="yolox", device_pre_post_layers=[])                

            # Order of insertion matters since we need the reorganized tensor to be in (BS,H,W,85) shape
            endnodes = [infer_results[layer_from_shape[1, 80, 80, 4]],  # stride 8 
                        infer_results[layer_from_shape[1, 80, 80, 1]],  # stride 8 
                        infer_results[layer_from_shape[1, 80, 80, 80]], # stride 8 
                        infer_results[layer_from_shape[1, 40, 40, 4]],  # stride 16
                        infer_results[layer_from_shape[1, 40, 40, 1]],  # stride 16
                        infer_results[layer_from_shape[1, 40, 40, 80]], # stride 16
                        infer_results[layer_from_shape[1, 20, 20, 4]],  # stride 32
                        infer_results[layer_from_shape[1, 20, 20, 1]],  # stride 32
                        infer_results[layer_from_shape[1, 20, 20, 80]]  # stride 32
                    ]
            hailo_preds = yolox_post_proc.yolo_postprocessing(endnodes)
            num_detections = int(hailo_preds['num_detections'])
            scores = hailo_preds["detection_scores"][0].numpy()
            classes = hailo_preds["detection_classes"][0].numpy()
            boxes = hailo_preds["detection_boxes"][0].numpy()
            if scores[0] == 0:
                num_detections = 0
            preds_dict = {'scores': scores, 'classes': classes, 'boxes': boxes, 'num_detections': num_detections}
            frame = report.report_detections(preds_dict, frame, scale_factor_x = orig_w, scale_factor_y = orig_h)
            '''
            
            if ( ( len(list(infer_results.values())[0][0][0][0]) == 5 ) ):
                posx   = int( list(infer_results.values())[0][0][0][0][0] * 640 )
                posy   = int( list(infer_results.values())[0][0][0][0][1] * 640 )
                width  = int( list(infer_results.values())[0][0][0][0][2] * 640 )
                height = int( list(infer_results.values())[0][0][0][0][3] * 640 )
                value  = ( list(infer_results.values())[0][0][0][0][4] * 100 )
                
                mytext1 = "[%4d, %4d] [%4d, %4d]" % (posx, posy, width, height)
                mytext2 = "[Score: %.3f]" % (value)
                #mytext1 = str( list(infer_results.values())[0][0][0] )
                #mytext1 = str(list(infer_results.values())[0][0][0][0][0])

                
                #print(posx)
                #print(posy)
                #print(width)
                #print(height)
                cv2.putText(resized_img, mytext1, (20, 50), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255))
                cv2.putText(resized_img, mytext2, (20, 100), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255))
                cv2.rectangle(resized_img,  (posy, posx), (height, width),    (255, 0, 0),   2, cv2.LINE_AA)
                #cv2.rectangle(resized_img, (100, 350),   (100+100, 350+100), (255, 0, 255), 2, cv2.LINE_AA, 2)

            
            
            
            #cv2.imshow('frame', frame)
            cv2.imshow('frame', resized_img)
            
            # wait for a key event
            key = cv2.waitKey(1)

            # switch between camera and video source
            if key == ord('c'):
                source = 'camera'
                cap.release()
                cap = cv2.VideoCapture(0)
            elif key == ord('v'):
                source = 'video'
                cap.release()
                random_mp4 = random.choice(mp4_files)
                cap = cv2.VideoCapture(video_dir+random_mp4)
            elif key == ord('q'):
                break

# release the video source and destroy all windows
cap.release()
cv2.destroyAllWindows()



