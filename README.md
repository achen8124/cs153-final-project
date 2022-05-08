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
The files `dnn.py` and `mtcnn.py`



