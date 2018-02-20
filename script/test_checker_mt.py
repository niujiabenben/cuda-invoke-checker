# coding: utf-8

import sys
import time
import Queue
import struct
import ctypes
import threading
import numpy as np


class Producer(threading.Thread):
    def __init__(self, name, samples_queue):
        threading.Thread.__init__(self, name=name)
        self.samples_queue = samples_queue

    def run(self):
        while True:
            sample = np.random.normal(size=int(sys.argv[1]))
            sample = sample.astype(np.float32).tostring()
            self.samples_queue.put(sample, timeout=1)
        print "Thread {} is finished".format(self.name)


class Consumer(threading.Thread):
    def __init__(self, name, samples_queue, checker):
         threading.Thread.__init__(self, name=name)
         self.samples_queue = samples_queue
         self.checker = checker
         self.max_size = 4
         self.output_cpu = ctypes.create_string_buffer(self.max_size)
         self.output_gpu = ctypes.create_string_buffer(self.max_size)

    def run(self):
        while True:
            sample = self.samples_queue.get(timeout=1)
            output_size_cpu = self.checker.process_cpu(
                sample, len(sample), self.output_cpu, self.max_size)
            output_size_gpu = self.checker.process_gpu(
                sample, len(sample), self.output_gpu, self.max_size)
            assert output_size_cpu == output_size_gpu == 4
            output_cpu = struct.unpack("f", self.output_cpu[:4])[0]
            output_gpu = struct.unpack("f", self.output_gpu[:4])[0]
            print "output_cpu: %.4f, output_gpu: %.4f" % (output_cpu, output_gpu)
        self.checker.release()
        print "Thread {} is finished".format(self.name)


if __name__ == "__main__":
    assert len(sys.argv) == 3, "Usage: {} size sequential|parallel"
    checker = ctypes.cdll.LoadLibrary('libchecker.so')
    checker.init(int(sys.argv[1]))

    samples_queue = Queue.Queue(maxsize=1024)
    producer = Producer("producer", samples_queue)
    consumer = Consumer("consumer", samples_queue, checker)
    producer.daemon = True
    consumer.daemon = True
    if sys.argv[2] == "sequential":
        producer.start()
        consumer.run()
    elif sys.argv[2] == "parallel":
        producer.start()
        consumer.start()
    else:
        print "Error: valid mode: [sequential, parallel], given " + sys.argv[2]
        exit(0)

    ### keep main thread alive
    while True: time.sleep(1)
    print "Done!"
