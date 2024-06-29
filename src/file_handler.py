import os
import pickle
import bz2file as bz2

class File_Handler:
    '''
    Handles file i/o
    '''
    def __init__(self, face_encodings_pkl:str) -> None:
        '''
        :param face_encodings_pkl: path to the compressed pickled dictionary 
            that stores/should store computed encodings.
        '''
        self._init_face_encodings_pkl(face_encodings_pkl)

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


    def _init_face_encodings_pkl(self, face_encodings_pkl:str):
        '''
        Checks if passed pickled dictionary exists. If yes, loads and
        stores dictionary. Else, attempts to create a new dictonary if the
        passed path is valid.
        :param face_encodings_pkl: path to the compressed pickled dictionary 
            that stores/should store computed encodings.
        '''
        if not self.__type_check('face_encodings_pkl', face_encodings_pkl, str):
            raise TypeError 
        if os.path.isfile(face_encodings_pkl):
            self.face_encodings_pkl = face_encodings_pkl
            self.face_encodings_dict = self._load_face_encodings()
        else:
            head, tail = os.path.split(face_encodings_pkl)
            if os.path.isdir(head):
                tail_split = tail.split('.')
                if tail_split[-1] != 'pbz2':
                    tail = f'{tail}.pbz2'
                self.face_encodings_pkl = f'{head}/{tail}'
                self.face_encodings_dict = dict()
            else:
                print(f'!!! "{face_encodings_pkl}" does not exist and cannot be created')
                raise ValueError


    def save_face_encodings(self, silent:bool=False) -> bool:
        '''
        Pickle dumps dictionary of encodings
        :param silent: print success/failure message if False.
        '''
        if not self.__type_check('silent', silent, bool):
            raise TypeError 
        try:
            with bz2.BZ2File(self.face_encodings_pkl, 'w') as file:
                pickle.dump(self.face_encodings_dict, file)
            if not silent:
                print(f'+++ Saved: {self.face_encodings_pkl}')
            return True
        except Exception as e:
            print(f'!!! Failed to save: {self.face_encodings_pkl}\n', e)
        return False


    def _load_face_encodings(self, silent=False) -> dict:
        '''
        Pickle load dictionary of encodings
        :param silent: print success/failure message if False.
        '''
        if not self.__type_check('silent', silent, bool):
            raise TypeError 
        try:
            data = bz2.BZ2File(self.face_encodings_pkl, 'rb')
            result = pickle.load(data)
            if not silent:
                print(f'+++ Loaded: {self.face_encodings_pkl}')
            return result
        except Exception as e:
            print(f'!!! Failed to load: {self.face_encodings_pkl}\n', e)


    def get_image_files(self, dir_path:str) -> list:
        '''
        Return list of images in the passed directory
        :param dir_path: path to folder of images to be encoded
        '''
        if not self.__type_check('dir_path', dir_path, str):
            raise TypeError 
        if os.path.exists(dir_path):
            files = [f'{dir_path}/{f}' for f in os.listdir(dir_path) if f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".png")]
            return files
        else:
            print(f'!!! {dir_path} does not exist')
            return