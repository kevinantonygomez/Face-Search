import numpy as np
from numpy.linalg import norm
from numpy import dot
import dlib
import cv2

class Face_Encoder:
    '''
    Handles the extraction and encoding of faces in images
    '''
    def __init__(self) -> None:
        '''
        Uses HOG + Linear SVM face detector
        TODO: CUDA and MMOD CNN face detector options
        '''
        self.face_detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor('data/shape_predictor_68_face_landmarks.dat')
        self.face_recognition_model = dlib.face_recognition_model_v1('data/dlib_face_recognition_resnet_model_v1.dat')
    
    def __type_check(self, obj_name:str, obj, type)->bool:
        '''
        Private method to type check an object
        :param obj_name: variable name
        :param obj: actual obj
        :param type: expected type of obj
        '''
        try:
            if not isinstance(obj, type):
                print(f'!!! {obj_name} should be of type {type} not {type(obj)}')
            else:
                return True
        except Exception as e:
            print(f"!!! Error in __type_check: \n {e}")
        return False
            
    def extract_faces(self, image:np.ndarray, upsample_times:int=0)->list:
        '''
        Extract and return faces (rectangles)
        :param image: image arr 
        :param upsample_times: optionally upsample image prior to encoding  
        '''
        if not self.__type_check('image', image, np.ndarray) or \
            not self.__type_check('upsample_times', upsample_times, int):
            raise TypeError
        if upsample_times < 0:
            print(f'upsample_times must be >= 0')
            raise ValueError
        faces = self.face_detector(image, upsample_times)
        return faces 

    def get_face_encodings(self, file:str, upsample_times:int=0) -> list:
        '''
        Returns a list of computed encodings for face(s) in an image
        :param file: image file path 
        :param upsample_times: optionally upsample image prior to encoding  
        '''
        if not self.__type_check('file', file, str) or \
            not self.__type_check('upsample_times', upsample_times, int):
            raise TypeError
        if upsample_times < 0:
            print(f'upsample_times must be >= 0')
            raise ValueError
        try:
            image = cv2.imread(file)
            faces = self.extract_faces(image, upsample_times)
            face_encodings = list()
            if len(faces) > 0:
                for face in faces:
                    pose_locations = self.shape_predictor(image, face)
                    face_np_arr = dlib.get_face_chip(image, pose_locations)
                    face_encodings.append(np.array(self.face_recognition_model.compute_face_descriptor(face_np_arr)))
            return face_encodings
        except FileNotFoundError:
            print(f"Img file '{file}' not found")
        except Exception as e:
            print(e)