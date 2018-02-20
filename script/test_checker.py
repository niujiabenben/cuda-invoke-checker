import ctypes
import struct
import numpy as np


if __name__ == "__main__":
    MAX_SIZE = 1000

    checker = ctypes.cdll.LoadLibrary('libchecker.so')
    checker.init(MAX_SIZE)
    input = np.random.normal(size=MAX_SIZE).astype(np.float32)
    input = input.tostring()
    output_cpu = ctypes.create_string_buffer(MAX_SIZE)
    output_size_cpu = checker.process_cpu(input, len(input), output_cpu, MAX_SIZE)
    output_gpu = ctypes.create_string_buffer(MAX_SIZE)
    output_size_gpu = checker.process_gpu(input, len(input), output_gpu, MAX_SIZE)
    assert output_size_cpu == output_size_gpu == 4
    output_cpu = struct.unpack("f", output_cpu[:4])[0]
    output_gpu = struct.unpack("f", output_gpu[:4])[0]
    print "output_cpu: %.4f, output_gpu: %.4f" % (output_cpu, output_gpu)
    checker.release()
    print "Done!"
