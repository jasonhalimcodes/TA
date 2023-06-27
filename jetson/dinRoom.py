import cv2
import numpy as np
import mqtt
import time

dispW = 640
dispH = 480
#fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#file = 'trial'                               # File name
#out_ori = cv2.VideoWriter('liveVid/'+file+'_diningRoom.mp4', fourcc, 21, (dispW, dispH))
#out_transformed = cv2.VideoWriter('liveVid/'+file+'_diningRoom_tranformed.mp4', fourcc, 21, (dispW, dispH))

# Constants
INPUT_WIDTH = 416
INPUT_HEIGHT = 416
NMS_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD = 0.2
#camSet='nvarguscamerasrc wbmode3 tnr-mode=2 tnr-strength=1 ee-mode=2 ee-strength=1 !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! videobalance contrast=1.5 brightness=-.2 saturation=1.2 ! appsink drop=true'
#flip = 2

# Text Parameters
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1

# Colors
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)

# Living Room Nando
#topL = (324, 157)
#botL = (29, 278)
#topR = (586, 236)
#botR = (355, 472)

# Dining Room Nando
#topL = (148, 240)
#botL = (208, 480)
#topR = (420, 220)
#botR = (640, 395)

# Dining Room Edward
topL = (148, 234)
botL = (209, 478)
topR = (430, 216)
botR = (640, 396)

# 4 Regions
region_width = dispW // 2
region_height = dispH // 2
regions = [
    (0, 0), # Top-left corner (0,0)
    (region_width, 0), # Top-right corner (0,1)
    (0, region_height), # Bottom-left corner (1,0)
    (region_width, region_height) # Bottom-right corner (1,1)
    ]
region_map = {0: 'H', 1: 'G', 2: 'E', 3: 'F'}

# Hysteresis
hysteresis_x = 16
hysteresis_y = 12       #5% of each region     

# Global variables      
first_iteration = True
prev_region_x = None
prev_region_y = None
prev_region_idx = None

pose_prediction = None
position_prediction = None

# Perspective Transform
src_points = np.float32([topL, botL, topR, botR])
dst_points = np.float32([(0,0), (0, dispH), (dispW, 0), (dispW, dispH)])
matrix = cv2.getPerspectiveTransform(src_points, dst_points)

# record the time last frame was processed
prev_frame_time = 0
new_frame_time = 0

# Utilities
def draw_label(input_image, label, left, top):
    
    # Get text size
    text_size = cv2.getTextSize(label, FONT, FONT_SCALE, THICKNESS)
    dim, baseline = text_size[0], text_size[1]
    
    # Use text size to create black highlight
    cv2.rectangle(input_image, (left, top), (left + dim[0], top + dim[1] + baseline), BLACK, cv2.FILLED)
    
    # Display text inside the highlight
    cv2.putText(input_image, label, (left, top + dim[1]), FONT, FONT_SCALE, YELLOW, THICKNESS, cv2.LINE_AA)


# Pre-Process function
def pre_process(input_image, net):
    # Create a 4D blob from a frame
    blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WIDTH, INPUT_HEIGHT), [0,0,0], 1, crop=False)
    
    # Set the input to the network
    net.setInput(blob)
    
    # Run the forward pass to get the output (final pred) of the output layers
    output_layers = net.getUnconnectedOutLayersNames()
    outputs = net.forward(output_layers)
    
    return outputs  

# Post-Process functions
def pose_detection(input_image, outputs, classes):
    class_ids = []
    confidences = []
    boxes = []
    pose_detected = None
    box = None
    x1 = 0
    y1 = 0
    x2 = 0
    y2 = 0
    x_label = 0
    y_label = 0
    
    height, width = input_image.shape[:2]
    
    # Iterate through all detections
    for out in outputs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > CONFIDENCE_THRESHOLD:
                # Get coordinates of bounding box
                cx, cy, w, h = detection[:4]
                x1 = int((cx - w /2) * width)
                y1 = int((cy - h /2) * height)
                x2 = int((cx + w /2) * width)
                y2 = int((cy + h /2) * height)
                
                # Add to list of boxes
                boxes.append([x1, y1, x2, y2])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-max suppresion to remove redundant overlapping boundary boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        
    for i in indices:
        # Get coordinates of the box and draw on the input_image (img)
        box = boxes[i]
        x1, y1, x2, y2 = box
        cv2.rectangle(input_image, (x1, y1), (x2, y2), RED, 3)
            
        # Draw label on input image with Utilities
        pose_label = "{}:{:.2f}".format(classes[class_ids[i]], confidences[i])
        x_label = x1
        y_label = y1
        
        if x_label < 0:
            x_label = 0
        elif x_label > dispW:
            x_label = dispW
        
        if y_label < 0:
            y_label = 0
        elif y_label > dispH:
            y_label = dispH
        
        draw_label(input_image, pose_label, x_label, y_label)
        pose_detected = classes[class_ids[i]]
    
    return pose_detected, x1, y1, x2, y2

# Other algorithms
def perspective_transform_pt(matrix, x1, y1, x2, y2):
    
    transformed_xy1 = cv2.perspectiveTransform(np.float32([(x1, y1)]).reshape(-1,1,2), matrix)
    transformed_xy1 = tuple(map(np.int_, transformed_xy1[0][0]))
    
    transformed_xy2 = cv2.perspectiveTransform(np.float32([(x2, y2)]).reshape(-1,1,2), matrix)
    transformed_xy2 = tuple(map(np.int_, transformed_xy2[0][0]))
    
    if transformed_xy1[0] < 0:
        transformed_xy1 = (0, transformed_xy1[1])
    if transformed_xy1[0] > dispW:
        transformed_xy1 = (dispW, transformed_xy1[1])
    if transformed_xy1[1] < 0:
        transformed_xy1 = (transformed_xy1[0], 0)
    if transformed_xy1[1] > dispH:
        transformed_xy1 = (transformed_xy1[0], dispH)
        
    if transformed_xy2[0] < 0:
        transformed_xy2 = (0, transformed_xy2[1])
    if transformed_xy2[0] > dispW:
        transformed_xy2 = (dispW, transformed_xy2[1])
    if transformed_xy2[1] < 0:
        transformed_xy2 = (transformed_xy2[0], 0)
    if transformed_xy2[1] > dispH:
        transformed_xy2 = (transformed_xy2[0], dispH)

    x1_t, y1_t = transformed_xy1[0], transformed_xy1[1]
    x2_t, y2_t = transformed_xy2[0], transformed_xy2[1]
    
    transformed_cxy = ((x1_t + x2_t)//2 , (y1_t + y2_t)//2)
    
    return transformed_cxy


def determine_region_location(pose_predicted, cxy_t, transformed_frame):
    global first_iteration
    global prev_region_idx
    global prev_region_x
    global prev_region_y
    
    region_id = None
    region_idx = None
    
    if pose_predicted == 'Baring di Lantai':
        for a, region in enumerate(regions):
            region_x, region_y = region
            x_in_region = region_x <= cxy_t[0] < region_x + region_width
            y_in_region = region_y <= cxy_t[1] < region_y + region_height
            region_detected = a
            if x_in_region and y_in_region:
                if first_iteration:
                    region_idx = region_detected
                    region_id = region_map[region_idx]
                    prev_region_idx = region_idx
                    prev_region_x = region_x
                    prev_region_y = region_y
                    first_iteration = False
                    
                else:
                    region_detected = a
                    x_in_region_h = prev_region_x - hysteresis_x <= cxy_t[0] < prev_region_x + region_width + hysteresis_x
                    y_in_region_h = prev_region_y - hysteresis_y <= cxy_t[1] < prev_region_y + region_height + hysteresis_y
                    if not (x_in_region_h and y_in_region_h):
                        region_idx = region_detected
                        prev_region_idx = region_idx
                        region_id = prev_region_idx
                        prev_region_x = region_x
                        prev_region_y = region_y

                    else:
                        region_idx = prev_region_idx
                        region_id = region_map[region_idx]
                        region_x = prev_region_x
                        region_y = prev_region_y
                            
        if region_idx is not None:
            if region_idx == 0 :
                cv2.rectangle(transformed_frame, (0,0), (dispW//2, dispH//2), YELLOW, -1)
            elif region_idx == 1 :
                cv2.rectangle(transformed_frame, (dispW//2, 0), (dispW, dispH//2), YELLOW, -1)
            elif region_idx == 2 :
                cv2.rectangle(transformed_frame, (0, dispH//2), (dispW//2, dispH), YELLOW, -1)
            elif region_idx == 3 :
                cv2.rectangle(transformed_frame, (dispW//2, dispH//2), (dispW, dispH), YELLOW, -1)
            
            region_label = "Region: {}".format(region_id)
            draw_label(transformed_frame, region_label, 10, 50)
        
        cv2.circle(transformed_frame, cxy_t, 5, GREEN, -1)
        pose_label_xyt = "| {}".format(cxy_t)
        draw_label(transformed_frame, pose_label_xyt, 250, 20)
    
    else:
        region_id = None
        
    pose_label_t = "Pose: {}".format(pose_predicted)
    draw_label(transformed_frame, pose_label_t, 10, 20)
         
    return region_id

def run():
    # Frame
    global new_frame_time
    global prev_frame_time
    
    # MQTT
    global client
    client = mqtt.connect_mqtt()
    client.loop_start()
    
    # Read and get the classes
    classesFile = "utils/obj.names"
    classes = None
    with open(classesFile, "r") as f:
        classes = f.read().rstrip("\n").split("\n")
    cap = cv2.VideoCapture('vidPengujian/piCam_diningRoom.mp4')
    #cap = cv2.VideoCapture(0)
    
    # Give weight files to the model and load the network
    net = cv2.dnn.readNet(model='models/diningRoom_120_best.weights', config='utils/yolov4-tiny-test.cfg')
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (dispW, dispH))
        
        # Apply Perspective Transform
        transformed_frame = cv2.warpPerspective(frame.copy(), matrix, (dispW, dispH))
        
        # Pre-process
        detections = pre_process(frame, net)
        
        # Draw Circles on Img
        cv2.circle(frame, topL, 5, RED, -1)
        cv2.circle(frame, botL, 5, RED, -1)
        cv2.circle(frame, topR, 5, RED, -1)
        cv2.circle(frame, botR, 5, RED, -1)
        
        cv2.line(frame, topL, botL, GREEN, 3)
        cv2.line(frame, topL, topR, GREEN, 3)
        cv2.line(frame, botL, botR, GREEN, 3)
        cv2.line(frame, topR, botR, GREEN, 3)
        
        # Post-process (pose detection, position)
        pose_prediction, x1, y1, x2, y2 = pose_detection(frame, detections, classes)
        cxy_t  = perspective_transform_pt(matrix, x1, y1, x2, y2)
        position_prediction = determine_region_location(pose_prediction, cxy_t, transformed_frame)
        
        # MQTT publish/subscribe
        mqtt.subscribe(client, "robot/docking")
        if pose_prediction != 'Baring di Lantai':
            position_prediction = "E"
        if mqtt.dock == "dock":
            mqtt.publish(client, pose_prediction, position_prediction)
        
        #FPS
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)
        
        t, _ = net.getPerfProfile()
        label = 'Inference time: %2.f ms, FPS: %i' % (t * 1000.0 / cv2.getTickFrequency(), fps)
        #print(label)
        cv2.putText(frame, label, (20,40), FONT, FONT_SCALE, RED, THICKNESS, cv2.LINE_AA)
        
        # Display windows
        cv2.imshow('Output', frame)
        cv2.moveWindow('Output', 10, 10)
        cv2.imshow('Transformed Output', transformed_frame)
        cv2.moveWindow('Transformed Output', 644, 10)
        
        #out_ori.write(frame)
        #out_transformed.write(transformed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    return pose_prediction, position_prediction
        
if __name__ == "__main__":
    run()
        
         
        
    