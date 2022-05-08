# CS 153 Final Project
Code to detect faces using the [OpenCV Face Detector](https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector) and the [MTCNN Face Detector](https://github.com/ipazc/mtcnn), as well as compare their results to the hand coding from the [Gaze Dataset](http://graphics.stanford.edu/~kbreeden/gazedata.html).

## Usage
You can install the package dependencies for this project with the following command: 
```pip install -r requirements.txt``` 

### Extracting frames from a video clip
Before using either of the face detectors on a video clip, extract the frames and place them in the directory of your choice. One tool to extract the frames of the video is `ffmpeg`, which is included in the dependencies mentioned above. To extract the frames from a video named `myclip.mp4`, you would run the following command:

```ffmpeg -i myclip.mp4 'path/to/where/i/want/frames/myclip_%05d.jpg'```

The `%05d` and `.jpg` can be changed to have more/less zero padding for the index and a different file format if desired. 

### Running the face detectors
The files `dnn.py` and `mtcnn.py` contain the code to run the face detectors on the extracted frames; import them to use any of the functions. To run the face detectors on a directory of extracted frames and get bounding boxes for faces, you can use the `detect_faces()` functions. See the docstrings for more info. 

Note that `deploy.prototxt` and `res10_300x300_ssd_iter_140000.caffemodel` must be in the same directory as your code to call `dnn.detect_faces()` because these are the config and model files for the pre-trained OpenCV face detector. 

### Comparing output to hand coding
You can convert a text file with the hand coding to a dictionary to be compared with output from the steps above by importing `compare_results.py` and calling `read_coding_file()`. Compare the dictionaries using `compare_dict()` or `cmp_dict_files()` if your dictionaries have been saved to a text file. 

Note that text files containing the hand coding for the frames must come in the following format:

![format for hand-coding text files](img_for_docs/hand_coding.png)

### Converting a sequence of images back to a video
After drawing bounding boxes around the detected faces, you can use `ffmpeg` to convert images back to a video with the following command:

```ffmpeg -framerate 24 -i myclip_%05d.jpg .path/to/where/i/want/video/myclip.mp4```


## Example output
See the `detector_outputs` and `video_clips` directories for examples of running the two face detectors on the videos from the Gaze Dataset. While the detectors have been run on all of the clips, there are only a few example videos due to space/memory considerations.

