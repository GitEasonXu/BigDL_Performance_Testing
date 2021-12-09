import argparse
import logging as log
import os
import sys
import cv2
import numpy as np
from utils import preprocess, postprocess, multiclass_nms, vis
from openvino.inference_engine import IECore

COCO_CLASSES = ("person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
                "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
                "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange",
                "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone",
                "microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush",)
def main():
    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)
    
    # ---------------------------Step 1. Initialize inference engine core--------------------------------------------------
    log.info('Creating Inference Engine')
    ie = IECore()
    
    # ---------------------------Step 2. Read a model in OpenVINO Intermediate Representation or ONNX format---------------
    model_xml = "./models/yolox_l.xml"
    model_bin = "./models/yolox_l.bin"
    log.info(f'Reading the network: {model_xml}')
    # (.xml and .bin files) or (.onnx file)
    net = ie.read_network(model=model_xml, weights=model_bin)
    
    # ---------------------------Step 3. Configure input & output----------------------------------------------------------
    log.info('Configuring input and output blobs')
    # Get names of input and output blobs
    input_blob = next(iter(net.input_info))
    out_blob = next(iter(net.outputs))
    
    # Set input and output precision manually
    net.input_info[input_blob].precision = 'FP32'
    net.outputs[out_blob].precision = 'FP16'
    
    # Get a number of classes recognized by a model
    num_of_classes = max(net.outputs[out_blob].shape)
    
    # ---------------------------Step 4. Loading model to the device-------------------------------------------------------
    log.info('Loading the model to the plugin')
    exec_net = ie.load_network(network=net, device_name="CPU")
    
    # ---------------------------Step 5. Create infer request--------------------------------------------------------------
    # load_network() method of the IECore class with a specified number of requests (default 1) returns an ExecutableNetwork
    # instance which stores infer requests. So you already created Infer requests in the previous step.
    
    # ---------------------------Step 6. Prepare input---------------------------------------------------------------------
    origin_img = cv2.imread("./images/street.jpg")
    _, _, h, w = net.input_info[input_blob].input_data.shape
    image, ratio = preprocess(origin_img, (h, w))
    
    # ---------------------------Step 7. Do inference----------------------------------------------------------------------
    log.info('Starting inference in synchronous mode')
    res = exec_net.infer(inputs={input_blob: image}) 
    
    # ---------------------------Step 8. Process output--------------------------------------------------------------------
    res = res[out_blob]
    predicts = postprocess(res, (h,w), p6=False)[0] 
    boxes = predicts[:, :4]
    scores = predicts[:, 4, None] * predicts[:, 5:]

    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
    boxes_xyxy /= ratio
    
    dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
    if dets is not None:
        final_boxes = dets[:, :4]
        final_scores, final_cls_inds = dets[:, 4], dets[:, 5]
        origin_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,
                         conf=0.3, class_names=COCO_CLASSES)
        
    cv2.imwrite("result_street.jpg", origin_img)

if __name__ == "__main__":
    main()
