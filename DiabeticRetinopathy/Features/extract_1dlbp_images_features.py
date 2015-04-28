from numba import cuda
from timeit import default_timer as timer

import numpy as np
from os import path
from tr_utils import append_to_arr, prep_out_path
from train_files import TrainFiles
from skimage import io
from cv2 import imread

root_path = "/kaggle/retina/train"
inp_path = path.join(root_path, "cropped")
out_path = path.join(root_path, "1dlbp")
ext = ".lbp"

neighborhood = 4

@cuda.jit('void(uint8[:], int32, int32[:], int32[:])')
def lbp_kernel(input, neighborhood, powers, h):
    i = cuda.grid(1)
    r = 0
    if i < input.shape[0] - 2 * neighborhood:
        i += neighborhood
        for j in range(i - neighborhood, i):
            if input[j] >= input[i]:
                r += powers[j - i + neighborhood]
    
        for j in range(i + 1, i + neighborhood + 1):
            if input[j] >= input[i]:
                r += powers[j - i + neighborhood - 1]

        cuda.atomic.add(h, r, 1)

def extract_1dlbp_gpu(input, neighborhood, d_powers):
    maxThread = 512
    blockDim = maxThread
    d_input = cuda.to_device(input)

    hist = np.zeros(2 ** (2 * neighborhood), dtype='int32')
    gridDim = (len(input) - 2 * neighborhood + blockDim) / blockDim

    d_hist = cuda.to_device(hist)

    lbp_kernel[gridDim, blockDim](d_input, neighborhood, d_powers, d_hist)
    d_hist.to_host()
    return hist

def extract_1dlbp_gpu_debug(input, neighborhood):
    res = np.zeros((input.shape[0] - 2 * neighborhood), dtype='int32')
    powers = 2 ** np.array(range(0, 2 * neighborhood), dtype='int32')

    maxThread = 512
    blockDim = maxThread
    gridDim = (len(input) - 2 * neighborhood + blockDim) / blockDim
    
    for block in range(0, gridDim):
        for thread in range(0, blockDim):
            i = blockDim * block + thread
            if i >= res.shape[0]:
                return res

            i += neighborhood
            for j in range(i - neighborhood, i):
                if input[j] >= input[i]:
                    res [i - neighborhood] += powers[neighborhood - (i - j)]
    
            for j in range(i + 1, i + neighborhood + 1):
                if input[j] >= input[i]:
                    res [i - neighborhood] += powers[j - i + neighborhood - 1]
    return res

def extract_1dlbp_cpu(input, neighborhood, p):
    """
    Extract the 1d lbp pattern on CPU
    """
    res = np.zeros((input.shape[0] - 2 * neighborhood))
    for i in range(neighborhood, len(input) - neighborhood):
        left = input[i - neighborhood : i]
        right = input[i + 1 : i + neighborhood + 1]
        both = np.r_[left, right]
        res[i - neighborhood] = np.sum(p [both >= input[i]])
    return res

def file_histogram(lbps, neighborhood):
    """
    Create a histogram out of the exracted pattern
    """
    bins = 2 ** (2 * neighborhood)
    hist = np.zeros(bins, dtype='int')

    for lbp in lbps:
        hist[lbp] += 1

    return hist

def get_1dlbp_features(neighborhood):
    tf = TrainFiles(inp_path, floor = neighborhood * 2 + 1)
    inputs = tf.get_training_inputs()

    start = timer()
    hist = np.array([])
    outs = np.array([])

    i = 0
    writeBatch = 100
    prep_out_path(out_path)

    p = 1 << np.array(range(0, 2 * neighborhood), dtype='int32')
    d_powers = cuda.to_device(p)

    for inp in inputs:

        data_file = path.join(inp_path, inp)

        out_file = path.join(out_path, path.splitext(inp)[0] + ext)
        arr = np.ascontiguousarray(imread(data_file)[:, :, 2].reshape(-1))

        ##GPU##
        file_hist = extract_1dlbp_gpu(arr, neighborhood, d_powers)

        ##CPU##
        #file_hist = extract_1dlbp_cpu(arr, neighborhood, p)
        #file_hist = file_histogram(file_hist, neighborhood)

        i += 1
        hist = append_to_arr(hist, file_hist)
        outs = append_to_arr(outs, out_file)

        if i == writeBatch:
            i = 0
            first = True
            for j in range(0, outs.shape[0]):
                hist[j].tofile(outs[j])
            hist = np.array([])
            outs = np.array([])



    print "==============Done==================="
    print "Elapsed: ", timer() - start

    print "Writing......."

    for i in range(0, outs.shape[0]):
        hist[i].tofile(outs[i])

    print "==============Done==================="

neighborhood = 4
get_1dlbp_features(neighborhood)
tf = TrainFiles(out_path, labels_file='/kaggle/retina/trainLabels.csv', test_size = 0.1)
X, Y, _, _ = tf.prepare_inputs()
tf.dump_to_csv(path.join(root_path, '1dlbp.csv'), X, Y)
