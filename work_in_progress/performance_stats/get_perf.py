

if __name__ == '__main__':
    file_name = 'comp_progress_output.txt'
    with open(file_name, 'r') as fid:
        data = fid.read()
    for line in data.split('\n'):
        if 'Capture_Ch2_28072015_140551 Compressing video.' in line:
            print(line)