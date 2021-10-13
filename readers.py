# -*- coding: utf-8 -*-


import os

try :
    import cv2
except ImportError as e :
    cv2 = e

try :
    import hiris
except ImportError as e :
    hiris = e

def select_reader(file_path):
    if os.path.splitext(file_path)[1] == ".seq" :
        if isinstance(hiris, ImportError) :
            raise hiris("hiris.py not available in library folder")
        return hiris.HirisReader
    elif os.path.splitext(file_path)[1] == ".avi" :
        if isinstance(cv2, ImportError) :
            raise cv2("OpenCV2 cannot be imported sucessfully of is not installed")
        return AviReader
    else :
        raise NotImplementedError("File extension/CODEC not supported yet")


class AviReader:
    def __init__(self,file_path):
        self.path = file_path
        self.cursor = 0

    def open(self):
        self.file_handle = cv2.VideoCapture( self.path ,cv2.IMREAD_GRAYSCALE )
        #width  = int(Handlevid.get(cv2.CAP_PROP_FRAME_WIDTH))
        #height = int(Handlevid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def close(self):
        pass

    def frames(self):
        for frame_id in range(self.frames_number) :
            yield self._get_frame()

    @property
    def frames_number(self):
        return int(self.file_handle.get(cv2.CAP_PROP_FRAME_COUNT))

    def _get_frame(self, frame_id = None):
        if frame_id is not None :
            if frame_id != self.cursor :
                self.cursor = frame_id
                self.file_handle.set(cv2.CAP_PROP_POS_FRAMES, self.cursor)

        self.cursor += 1
        _ , temp_frame = self.file_handle.read()
        return temp_frame[:,:,0]