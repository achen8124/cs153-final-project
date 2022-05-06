import json

def read_coding_file(filename):
    """
    
    """
    face_dict = {}

    with open(filename) as f:
        lines = f.readlines()

        for line in lines:
            words = line.split()

            if len(words) == 3:
                type, start, end = words

                if type == 'f' or type == 'ff':
                    for i in range(int(start), int(end)+1):
                        face_dict[i] = type
                elif type == 'end':
                    face_dict['num_frames'] = end

    return face_dict

def update_intervals(list, i):
    if list != [] and i == list[-1][1] + 1:
        list[-1][1] = i
    else:
        list += [[i,i]]

def write_and_reset(file, list, message):
    with open(file, 'a') as f:
        for interval in list:
            f.write(f"Frame {interval[0]}-{interval[1]}: {message}\n")
            print(f"Frame {interval[0]}-{interval[1]}: {message}")
    return []

def compare_dict(detected, actual, output_file):

    f_not_detected = []
    ff_not_detected = []
    f_detected = []
    ff_detected = []

    count = 0
    num_frames = int(actual['num_frames'])

    for i in range(1, num_frames + 1):
        key = str(i)
        in_actual = key in actual
        in_detected = key in detected
        
        if in_actual:
            # if key in actual but not in detected
            if not in_detected:
                if actual[key] == 'f':
                    update_intervals(f_not_detected, i)
                    curr = 'f_not_detected'
                else:
                    update_intervals(ff_not_detected, i)
                    curr = 'ff_not_detected'
                count += 1
            # key in detected, but results don't match
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

        # key not in actual, but in detected (false detection)
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
    
    with open(output_file, 'a') as f:
        f.write(f"\nMismatch in {count} out of {num_frames} frames.\n")
        f.write(f"Accuracy: {(num_frames - count)/num_frames}")


def cmp_dict_files(detected_file, actual_file, output_file):

    with open(detected_file, 'r') as f1, open(actual_file, 'r') as f2:
        detected = json.loads(f1.read())
        actual = json.loads(f2.read())
    
    # clear the contents of the output file (if any) before writing to it
    open(output_file, "w").close() 
    compare_dict(detected, actual, output_file)


# cmp_dict_files('ncc2_half.txt', 'country2.txt', 'compare.txt')
# with open('sc1_hc.txt', 'w') as file:
#    file.write(json.dumps(read_coding_file('hand_coding/shakespeare_clip1_hcode.txt')))
a = 'sc2'
cmp_dict_files(f'{a}_output/{a}_dnn_output.txt', f'{a}_output/{a}_hc.txt', f'{a}_output/compared_{a}_dnn.txt')
cmp_dict_files(f'{a}_output/{a}_mtcnn_output.txt', f'{a}_output/{a}_hc.txt', f'{a}_output/compared_{a}_mtcnn.txt')

a = 'sc1'
cmp_dict_files(f'{a}_output/{a}_dnn_output.txt', f'{a}_output/{a}_hc.txt', f'{a}_output/compared_{a}_dnn.txt')
cmp_dict_files(f'{a}_output/{a}_mtcnn_output.txt', f'{a}_output/{a}_hc.txt', f'{a}_output/compared_{a}_mtcnn.txt')
