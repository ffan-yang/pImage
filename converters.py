# -*- coding: utf-8 -*-

"""Boilerplate:
A one line summary of the module or program, terminated by a period.

Rest of the description. Multiliner

<div id = "exclude_from_mkds">
Excluded doc
</div>

<div id = "content_index">

<div id = "contributors">
Created on Tue Oct 12 17:34:44 2021
@author: Timothe
</div>
"""

from multiprocessing import Pool, Manager
import sys, time
from video import AutoVideoReader
from video import AutoVideoWriter

def _lambda_transform(item):
    return item

class Standard_Converter:

    def __init__(self,input_path,output_path, **kwargs):

        m = Manager()
        self.read_queue = m.Queue()
        self.transformed_queue = m.Queue()
        self.message_queue = m.Queue()

        self.input_path = input_path
        self.output_path = output_path
        self.kwargs = kwargs

    def start(self):
        with Pool(processes=3) as pool:

            read_process = pool.apply_async(self.read, (AutoVideoReader,self.read_queue,self.message_queue))
            transform_process = pool.apply_async(self.transform, (self.kwargs.pop("transform_function",_lambda_transform),self.read_queue,self.transformed_queue,self.message_queue))
            write_process = pool.apply_async(self.write, (AutoVideoWriter,self.transformed_queue,self.message_queue))

            self.last_update = time.time()
            self.r = self.t = self.w = 0
            self.max_i = 1
            while True :
                msg = self.message_queue.get()
                self.msg_parser(msg)
                if msg == "End of write process":
                    break
        print("\n" + "Conversion done")

    def msg_parser(self,message):

        if message in ("r", "t" ,"w"):
            exec(f"self.{message} += 1")
            if time.time() - self.last_update > 1 :
                self.last_update = time.time()
                message = fr"""Reading : {(self.r/self.max_i)*100:.2f} % - Transforming : {(self.t/self.max_i)*100:.2f} % - Writing : {(self.w/self.max_i)*100:.2f} %"""
                print(message, end = '\r', flush=True)
        elif len(message) >= 7 and message[:7] == "frameno" :
            self.max_i = int(message[7:])
        elif len(message) >= 3 and message[:3] == "End" :
            print("\n" + message, end = '', flush=True)

    def read(self, AutoVideoReader, read_queue , message_queue):

        with AutoVideoReader(self.input_path, **self.kwargs) as vid_read :
            message_queue.put("frameno"+str(vid_read.frames_number))
            for frame in vid_read.frames():
                read_queue.put(frame)
                message_queue.put("r")
        read_queue.put(None)
        message_queue.put("End of read process")
        sys.stdout.flush()

    def transform(self, transform_function, read_queue, transformed_queue , message_queue):
        while True :
            frame = read_queue.get()
            if frame is None:
                transformed_queue.put(None)
                break
            message_queue.put("t")
            transformed_queue.put(transform_function(frame))
        message_queue.put("End of transform process")
        sys.stdout.flush()

    def write(self,AutoVideoWriter, transformed_queue, message_queue):
        with AutoVideoWriter(self.output_path, **self.kwargs) as vid_write :
            while True :
                frame = transformed_queue.get()
                if frame is None:
                    break
                message_queue.put("w")
                vid_write.write(frame)
        message_queue.put("End of write process")
        sys.stdout.flush()

if __name__ == "__main__" :
    test = Standard_Converter( r"C:\Users\Timothe\Desktop\Testzone\Mouse33_2020-07-06T16.15.31.avi" ,  r"C:\Users\Timothe\Desktop\Testzone\TESSSST.avi")
    test.start()






