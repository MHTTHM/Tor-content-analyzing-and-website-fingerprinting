MAX_TRACE_LENGHT = 5000 # 5000
MAX_MATRIX_LEN = 1800
MAX_LOAD_TIME = 45 # 45

def packets_perslot(times, sizes):
    feature = [[0 for _ in range(MAX_MATRIX_LEN)], [0 for _ in range(MAX_MATRIX_LEN)]]
    for i in range(0, len(sizes)):
        if sizes[i] > 0:
            if times[i] >= MAX_LOAD_TIME:
                feature[0][-1] += 1
            else:
                idx = int(times[i] * (MAX_MATRIX_LEN - 1) / MAX_LOAD_TIME)
                feature[0][idx] += 1
        if sizes[i] < 0:
            if times[i] >= MAX_LOAD_TIME:
                feature[1][-1] += 1
            else:
                idx = int(times[i] * (MAX_MATRIX_LEN - 1) / MAX_LOAD_TIME)
                feature[1][idx] += 1

    return feature