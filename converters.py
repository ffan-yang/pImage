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

import multiprocessing as mp

class Standard_Converter:

    def __init__(self,input_path,output_path, **kwargs):

        self.read_queue = mp.Queue()
        self.transformed_queue = mp.Queue()

        self.input_path = input_path
        self.output_path = output_path
        self.kwargs = kwargs

        self.read_process = mp.Process(target=self.read, args=())
        self.transform_process = mp.Process(target=self.transform, args=())
        self.write_process = mp.Process(target=self.write, args=())

    def start(self):

        self.read_process.start()
        self.transform_process.start()
        self.write_process.start()

        self.read_process.join()
        self.transform_process.join()
        self.write_process.join()

        print("Done")

    def read(self):
        from video import AutoVideoReader
        with AutoVideoReader(self.input_path, **self.kwargs) as vid_read :
            for frame in vid_read.frames():
                self.read_queue.put(frame)
            self.read_queue.put(None)
            print("Enf of read process")

    def transform(self):
        while True :
            frame = self.read_queue.get()
            self.transformed_queue.put(frame)
            if frame is None:
                break
            print("Enf of transform process")

    def write(self):
        from video import AutoVideoWriter
        with AutoVideoWriter(self.output_path, **self.kwargs) as vid_write :
            while True :
                frame = self.transformed_queue.get()
                if frame is None:
                    break
                vid_write.write(frame)
            print("Enf of write process")


if __name__ == "__main__" :
    test = Standard_Converter( r"C:\Users\Timothe\Desktop\Testzone\Mouse33_2020-07-06T16.15.31.avi" ,  r"C:\Users\Timothe\Desktop\Testzone\TESSSST.avi")
    print(test)
    test.start()






