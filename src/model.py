import typing
import numpy as np
from numpy.linalg import norm
from numpy import dot
import dlib
import cv2
import render_html

class Metrics:
    def __init__(self, euclid_dist:np.float64=0.0, cos_sim:np.float64=0.0) -> None:
        for param in [euclid_dist, cos_sim]:
            if type(param) != np.float64:
                print(f'param must be of type np.float64. Got: {type(param)}')
                raise TypeError
            if param is euclid_dist:
                if param < 0.0:
                    print(f'param must be in [0,). Got: {param}')
                    raise ValueError
            elif param is cos_sim:
                if param < 0.0 or param > 1.0:
                    print(f'param must be in [0,1]. Got: {param}')
                    raise ValueError
        self.euclid_dist = euclid_dist
        self.cos_sim = cos_sim
        self.sim_score = self.euclid_dist + abs(self.cos_sim-1)

class FaceData:
    def __init__(self, face_encodings:list=list(), faces:list=list()) -> None:
        self.face_encodings = face_encodings
        self.faces = faces

class Model:
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
        self.renderer = render_html.Renderer()
    
    def _type_check(self, obj_name:str, obj, type)->bool:
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
            print(f"!!! Error in _type_check: \n {e}")
        return False
            
    def extract_faces(self, image:np.ndarray, upsample_times:int=0)->list:
        '''
        Extract and return faces (rectangles)
        :param image: image arr 
        :param upsample_times: optionally upsample image prior to encoding  
        '''
        if not self._type_check('image', image, np.ndarray) or \
            not self._type_check('upsample_times', upsample_times, int):
            raise TypeError
        if upsample_times < 0:
            print(f'upsample_times must be >= 0')
            raise ValueError
        faces = self.face_detector(image, upsample_times)
        return faces 

    def get_face_data(self, file:str, upsample_times:int=0) -> FaceData:
        '''
        Returns a list of computed encodings for face(s) in an image
        :param file: image file path 
        :param upsample_times: optionally upsample image prior to encoding  
        '''
        if not self._type_check('file', file, str) or \
            not self._type_check('upsample_times', upsample_times, int):
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
            return FaceData(face_encodings, faces)
        except FileNotFoundError:
            print(f"Img file '{file}' not found")
        except Exception as e:
            print(e)

    def compute_similarity(self, encodings_1:list, encodings_2:list)->list[Metrics]:
        if not self._type_check('encodings_1', encodings_1, list) or \
            not self._type_check('encodings_2', encodings_2, list):
            raise TypeError
        if len(encodings_1) == 0 or len(encodings_2) == 0:
            return list()
        if len(encodings_1) > 1:
            print(f'query image must contain only one face')
            raise ValueError
        if not self._type_check('encodings_1[0]', encodings_1[0], np.ndarray) or \
            not self._type_check('encodings_2[0]', encodings_2[0], np.ndarray):
            raise TypeError
        metrics_list = list()
        try:
            for encoding_2 in encodings_2:
                euclidean_dist = norm(encodings_1[0]-encoding_2) # 0 is an exact match
                cosine_sim = dot(encodings_1[0], encoding_2)/(norm(encodings_1[0])*norm(encoding_2)) # 1 is an exact match
                metrics_list.append(Metrics(euclidean_dist, cosine_sim))
        except Exception as e:
            print(e)
        return metrics_list

    def test_similarity(self, encodings_1:list, encodings_2:list, euclidean_thres:float = 0.61, \
        cosine_thres:float = 0.92, silent:bool=False)->bool:
        if not self._type_check('encodings_1', encodings_1, list) or \
            not self._type_check('encodings_2', encodings_2, list) or \
            not self._type_check('euclidean_thres', euclidean_thres, float) or \
            not self._type_check('cosine_thres', cosine_thres, float):
            raise TypeError
        if len(encodings_1) == 0 or len(encodings_2) == 0:
            return list()
        if len(encodings_1) > 1:
            print(f'query image: must contain only one face')
            raise ValueError
        if not self._type_check('encodings_1[0]', encodings_1[0], np.ndarray) or \
            not self._type_check('encodings_2[0]', encodings_2[0], np.ndarray):
            raise TypeError
        if euclidean_thres < 0:
            print(f'euclidean_thres must be in [0,)')
            raise ValueError
        if cosine_thres < 0 or cosine_thres > 1:
            print(f'cosine_thres must be in [0, 1]')
            raise ValueError
        try:
            for i, encoding_2 in enumerate(encodings_2):
                euclidean_dist = norm(encodings_1[0]-encoding_2) # 0 is an exact match
                cosine_sim = dot(encodings_1[0], encoding_2)/(norm(encodings_1[0])*norm(encoding_2)) # 1 is an exact match
                if euclidean_dist <= euclidean_thres or cosine_sim >= cosine_thres:
                    if not silent:
                        print(f'encodings_2[{i}] is similar')
                    return True
            if not silent:
                print(f'Dissimilar')
        except Exception as e:
            print(e)
        return False
    
    def find_similarities(self, query:str, data:typing.Dict[str, FaceData]):
        if not self._type_check('query', query, str) or \
            not self._type_check('data', data, dict):
            raise TypeError
        if not query in data:
            print(f"query: {query} not in data")
            raise KeyError
        metrics_dict = dict()
        query_encoding = data[query].face_encodings
        if len(query_encoding) != 1:
            print(f'query image {query}: must contain exactly 1 face')
            raise ValueError
        for key in data.keys():
            if key == query:
                continue
            encoding = data[key].face_encodings
            metrics_list = self.compute_similarity(query_encoding, encoding)
            metrics_dict[(query,key)] = metrics_list
        return metrics_dict


    def render_similar_images(self, query:str, data:typing.Dict[str, FaceData], metrics_dict,
        top_k:int, euclidean_thres:float = 0.6, cosine_thres:float = 0.92):

        if not self._type_check('query', query, str) or \
            not self._type_check('data', data, dict) or \
            not self._type_check('distances', metrics_dict, dict) or \
            not self._type_check('top_k', top_k, int) or \
            not self._type_check('euclidean_thres', euclidean_thres, float) or \
            not self._type_check('cosine_thres', cosine_thres, float):
            raise TypeError
        if euclidean_thres < 0:
            print(f'euclidean_thres must be in [0,1)')
            raise ValueError
        if cosine_thres < 0 or cosine_thres > 1:
            print(f'cosine_thres must be in [0, 1]')
            raise ValueError
        if not query in data:
            print(f"query: {query} not in data")
            raise KeyError
        if top_k > len(data)-1:
            print(f'Cannot retrieve {top_k} similar images from dict of size {len(data)-1}')
            raise ValueError
        similar_images_metric_keys = list()
        for key in metrics_dict.keys():
            metrics_list = metrics_dict[key]
            for face_num, metric in enumerate(metrics_list):
                if metric.euclid_dist <= euclidean_thres or metric.cos_sim >= cosine_thres:
                    similar_images_metric_keys.append((key, face_num, metric.sim_score))

        similar_images_metric_keys = sorted(similar_images_metric_keys, key=lambda tup: tup[2])[:top_k]
        # k[0][1] since key is a tuple with 2nd str being the matched img key
        similar_images_keys = [k[0][1] for k in similar_images_metric_keys] 
        self.renderer.set_query_image(query)
        for i in range(len(similar_images_keys)):
            self.renderer.update_image_tags(similar_images_keys[i],similar_images_metric_keys[i][2])
        self.renderer.render()