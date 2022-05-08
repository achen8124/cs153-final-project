from mtcnn import MTCNN
import cv2
import json
import os

def detect_faces(frame_dir, output_file, draw_boxes, get_keypoints):
    """
        Detects faces given a directory of video frames in .png, .jpg, or .jpeg format

        Inputs: frames_dir    - the relative path to a directory of video frames
                output_file   - the name of the txt file for the output face dict to be stored in
                draw_boxes    - bool corresponding to whether bounding boxes should be drawn over the images
                                to the output_frames directory
                get_keypoints - bool corresponding to whether facial keypoints in the frames should be saved
                                in the face dictionary
        Output: dictionary containing the bounding boxes of the faces (and the keypoints if chosen) using 
                the frame number as the key. Separately saves face coding dictionary to a text file.
    """
    # Get a list of the frames in the given directory and set up detector
    frames = os.listdir(frame_dir)
    detector = MTCNN()
    face_coding = {}
    face_info = {}
    
    print(f"Beginning face detection for {frame_dir}...")

    for i in range(len(frames)):

        # Find index of current image and set up relative path for image to be read
        index = int(os.path.splitext(frames[i])[0].split('_')[-1])
        ext = os.path.splitext(frames[i])[-1].lower()
        img_path = os.path.join(frame_dir, frames[i])

        # Safeguard for extra files that aren't pictures in the same directory (i.e. .DS_Store)
        if not ext == ".png" and not ext == ".jpg" and not ext == "jpeg":
            continue
        
        # Detect faces and update dictionaries if applicable
        face_list = detect_face(img_path, detector, draw_boxes)

        if len(face_list) > 0:
            face_info[index] = {}
            face_info[index]['box'] = []
            
            # Only store keypoints if told to
            if get_keypoints:
                face_info[index]['keypoints'] = []

            # Update box & keypoint values (if applicable) in nested dictionary
            for face in face_list:
                face_info[index]['box'] += [face['box']]

                if get_keypoints:
                    face_info[index]['keypoints'] += [face['keypoints']]

            # Update face coding dictionary ('f' for single, 'ff' for multiple faces)
            if len(face_info[index]['box']) == 1:
                face_coding[index] = 'f'
            else:
                face_coding[index] = 'ff'
        


    # write the face coding dictionary to a file
    with open(output_file, 'w') as file:
        file.write(json.dumps(face_coding))
    
    print(f"Finished face detection for {frame_dir}.")

    return face_info

def detect_face(filename, detector, draw_boxes):
    """
        Detects faces given an image in .png, .jpg, or .jpeg format

        Inputs: filename    - the relative path to an image to detect faces on
                detector    - the MTCNN detector to be used
                draw_boxes  - bool corresponding to whether bounding boxes should be drawn over the images
                              to the output_frames directory
        Output: a list of dictionaries corresponding to each detected face. Each dictionary contains the keys:
                'box', 'confidence', and 'keypoints.'
    """
    # Read the image and convert it to RGB, then detect faces in the image
    img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(img)
    
    # Draw boxes if applicable
    if draw_boxes:
        draw_image_with_boxes(filename, faces)

    return faces


# draw an image with detected objects
def draw_image_with_boxes(filepath, faces):
    """
        Given a list of face bounding boxes, draws a rectangle for each of them on 
        the image and saves it as a .jpg under the output_frames directory

        Inputs: filepath - path to the image that boxes should be drawn on
                faces    - a list of dictionaries corresponding to each detected face
        Output: a image (jpg) with bounding boxes drawn onto it saved to the output_frames directory
    """
    image = cv2.imread(filepath)

    # Draw a bounding box for each face detected in the image
    for face in faces:
        startX, startY, width, height = face['box']
        endX = startX + width
        endY = startY + height

        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)

    # Set up filename for the boxed image to be saved to and write image to .jpg file
    filename = filepath.split("/")[-1]
    without_ext = filename.split(".")[0]
    boxed_filename = os.path.join("output_frames", without_ext, ".jpg")

    cv2.imwrite(boxed_filename, image)

