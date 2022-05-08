import numpy as np
import json
import cv2
import os

def detect_faces(frames_path, output_file, threshold, draw_boxes):
    """
    Detects faces given a directory of video frames in .png, .jpg, or .jpeg format

        Inputs: frames_path - the relative path to a directory of video frames
                output_file - the name of the txt file for the output face dict to be stored in
                threshold   - minimum confidence to be considered a face
                draw_boxes  - bool corresponding to whether bounding boxes should be drawn over the images
                              to the output_frames directory

        Output: a dictionary storing the coordinates of the faces' bounding boxes using 
                the frame number as the key, writes detected face dictionary to output txt file
    """

    # Load face detection model and get frames from directory
    prototxt = "deploy.prototxt"
    model = "res10_300x300_ssd_iter_140000.caffemodel"
    net = cv2.dnn.readNetFromCaffe(prototxt, model)

    faces = {}
    boxes = {}
    frames = os.listdir(frames_path)

    print(f"Beginning face detection for {frames_path}...")

    # for each frame, detect face(s) and add bounding boxes to the face dictionary
    for i in range(len(frames)):
        
        # find index of current image and set up relative path for image to be read
        index = int(os.path.splitext(frames[i])[0].split('_')[-1])
        ext = os.path.splitext(frames[i])[-1].lower()
        img_path = os.path.join(frames_path, frames[i])

        # safeguard for extra files that aren't pictures in the same directory (i.e. .DS_Store)
        if not ext == ".png" and not ext == ".jpg" and not ext == "jpeg":
            continue

        # try to detect faces and update faces and boxes dictionaries if any are found
        bounding_boxes = detect_face(img_path, net, threshold, draw_boxes)

        if len(bounding_boxes) > 0:
            boxes[index] = bounding_boxes
            
            if len(bounding_boxes) == 1:
                faces[index] = 'f'
            else:
                faces[index] = 'ff'

        
    # write the face dictionary to a file
    with open(output_file, 'w') as file:
        file.write(json.dumps(faces))
    
    print(f"Finished face detection for {frames_path}.")

    return boxes


def detect_face(img_path, net, threshold, draw_boxes):
    """
        Adapted from https://pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/

        Takes in an image path, a net, a confidence threshold, and a bool that indicates whether boxes
        should be drawn over each detected face and returns a list of face bounding boxes in the form: 
        (startX, startY, endX, endY).
    """
    bounding_boxes = []

    # Read image and get original dimensions, then resize to 300x300 and convert to blob for detection
    image = cv2.imread(img_path)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, \
        (300, 300), (104.0, 177.0, 123.0))   # weights suggested for best accuracy in OpenCV documentation
    
    # pass the blob through the network and obtain the detections and predictions
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections 
    for i in range(0, detections.shape[2]):

        # confidence has to be above threshold to compute the coordinates of the bounding box
        confidence = detections[0, 0, i, 2]

        if confidence > threshold:
            # resizing box back to original dimensions
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            bounding_boxes += [box.astype("int")]

    if draw_boxes:
        draw_image_with_boxes(img_path, bounding_boxes)  

    return bounding_boxes


def draw_image_with_boxes(filepath, faces):
    """
        Given a list of face bounding boxes, draws a rectangle for each of them on the provided image 
        and saves it as a .jpg under the output_frames directory

        Inputs: filepath - path to the image that boxes should be drawn on
                faces    - a list of dictionaries corresponding to each detected face
        Output: a image (jpg) with bounding boxes drawn onto it saved to the output_frames directory
    """
    image = cv2.imread(filepath)

    # Draw a bounding box for each face detected in the image
    for face in faces:
        startX, startY, endX, endY = face
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)

    # Set up filename for the boxed image to be saved to and write image to .jpg file
    filename = filepath.split("/")[-1]
    without_ext = filename.split(".")[0]
    boxed_filename = os.path.join("output_frames", without_ext, ".jpg")

    cv2.imwrite(boxed_filename, image)
    
