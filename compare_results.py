import json

def read_coding_file(filename):
    """
        Takes in a text file containing the hand-coding for a video clip and returns a dictionary
        with the frame number as a key and 'f' or 'ff' as the values for single face or multiple faces
        in the frame, respectively. Hand coding must follow same format as the Gaze Dataset 
        (Breeden and Hanrahan, 2017).

        Input:  filename - string containing path to the hand coded text file
        Output: a dictionary with the frame number as a key and 'f' or 'ff', and the number of frames 
                stored with the key 'num_frames'
    """
    face_dict = {}

    with open(filename) as f:
        lines = f.readlines()

        # add entries to the dictionary corresponding to each line in the file
        for line in lines:
            words = line.split()

            if len(words) == 3:
                type, start, end = words

                # only add entries for the codes: f, ff, or end
                if type == 'f' or type == 'ff':
                    for i in range(int(start), int(end)+1):
                        face_dict[i] = type
                elif type == 'end':
                    face_dict['num_frames'] = end

    return face_dict


def cmp_dict_files(detected_file, actual_file, output_file):
    """
        Reads in 2 text files containing dictionaries and compares their values. Writes the
        differences to the given output filename.

        Inputs: detected_file - path to text file containing the coding as detected by face detectors
                actual_file   - path to text file containing the hand coding (ground truth)
                output_file   - the name/path to the text file to write the comparison to
    """

    # opens text files and loads them as dictionaries
    with open(detected_file, 'r') as f1, open(actual_file, 'r') as f2:
        detected = json.loads(f1.read())
        actual = json.loads(f2.read())
    
    # clear the contents of the output file (if any) before writing to it
    open(output_file, "w").close() 
    compare_dict(detected, actual, output_file)

def compare_dict(detected, actual, output_file):
    """
        Compares two dictionaries and writes the differences in their values to an output
        text file. 

        Inputs: detected    - coding dictionary resulting from running OpenCV/MTCNN face detector
                actual      - coding dictionary from the hand coding Gaze Dataset files
                output_file - the name/path to the text file to write the comparison to
    """

    # lists for frame intervals where detected and actual are inconsisent
    f_not_detected = []
    ff_not_detected = []
    f_detected = []
    ff_detected = []

    count = 0
    num_frames = int(actual['num_frames'])

    # compare the values of the dictionaries for each frame 
    for i in range(1, num_frames + 1):
        # the frame index is the key
        key = str(i)
        in_actual = key in actual
        in_detected = key in detected
        
        if in_actual:
            # at least one face wasn't detected (key in detected, but not actual)
            if not in_detected:
                if actual[key] == 'f':
                    update_intervals(f_not_detected, i)
                    curr = 'f_not_detected'
                else:
                    update_intervals(ff_not_detected, i)
                    curr = 'ff_not_detected'
                count += 1
            # there was at least one face detected (key in actual & detected), but results don't match
            elif actual[key] != detected[key]:
                if actual[key] == 'f':
                    update_intervals(ff_detected, i)
                    curr = 'ff_detected'
                else:
                    update_intervals(ff_not_detected, i)
                    curr = 'ff_not_detected'
                count += 1
            else:
                curr = 'correct'

        # at least one false detection since the key was in detected, but not actual
        elif in_detected:
            if detected[key] == 'f':
                update_intervals(f_detected, i)
                curr = 'f_detected'
            else:
                update_intervals(ff_detected, i)
                curr = 'ff_detected'
            count += 1
        else:
            curr = 'correct'

        # print appropriate error message as soon as the mismatch changes type, update prev
        if i != 1 and prev != curr:
            if prev == 'f_detected':
                f_detected = write_and_reset(output_file, f_detected, "face falsely detected")
            elif prev == 'ff_detected':
                ff_detected = write_and_reset(output_file, ff_detected, "multiple faces falsely detected")
            elif prev == 'f_not_detected':
                f_not_detected = write_and_reset(output_file, f_not_detected, "face not detected")
            else:
                ff_not_detected = write_and_reset(output_file, ff_not_detected, "multiple faces not detected")

        prev = curr
    
    # write the total mismatches and overall accuracy to the file
    with open(output_file, 'a') as f:
        f.write(f"\nMismatch in {count} out of {num_frames} frames.\n")
        f.write(f"Accuracy: {(num_frames - count)/num_frames}")

def update_intervals(list, i):
    """
        Helper function for compare_dict(). Takes a list of intervals and a current index,
        and extends the current interval if applicable, or adds a new interval to the list if not. 
    """
    if list != [] and i == list[-1][1] + 1:
        list[-1][1] = i
    else:
        list += [[i,i]]

def write_and_reset(file, list, message):
    """
        Helper function for compare_dict(). Takes a list of intervals and writes
        a new line in the file for every element in the list with the given message.

        Inputs: file    - path to file to append to
                list    - list of frame intervals
                message - the inconsistency between the detected and hand coding to write
                         to write to the file
        Outputs: an empty list to 'reset' the interval list for the next batch of messages
    """
    with open(file, 'a') as f:
        for interval in list:
            f.write(f"Frame {interval[0]}-{interval[1]}: {message}\n")
    return []
