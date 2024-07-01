import os
import pickle
import bz2file as bz2
from pathlib import Path

class FileHandler:
    '''
    Handles file i/o
    '''
    def __init__(self, face_data_pkl:str) -> None:
        '''
        :param face_data_pkl: path to the compressed pickled dictionary 
            that stores/should store extracted face data.
        '''
        self._init_face_data_pkl(face_data_pkl)

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


    def _init_face_data_pkl(self, face_data_pkl:str):
        '''
        Checks if passed pickled dictionary exists. If yes, loads and
        stores dictionary. Else, attempts to create a new dictonary if the
        passed path is valid.
        :param face_data_pkl: path to the compressed pickled dictionary 
            that stores/should store extracted face data.
        '''
        if not self._type_check('face_data_pkl', face_data_pkl, str):
            raise TypeError 
        if os.path.isfile(face_data_pkl):
            self.face_data_pkl = face_data_pkl
            self.face_data_dict = self._load_face_data()
        else:
            head, tail = os.path.split(face_data_pkl)
            if head == '':
                head = os.path.abspath('data')
            if os.path.isdir(head):
                tail_split = tail.split('.')
                if tail_split[-1] != 'pbz2':
                    tail = f'{tail}.pbz2'
                if os.path.isfile(f'{head}/{tail}'):
                    self.face_data_pkl = f'{head}/{tail}'
                    self.face_data_dict = self._load_face_data()
                else:
                    self.face_data_pkl = f'{head}/{tail}'
                    self.face_data_dict = dict()
            else:
                print(f'!!! "{face_data_pkl}" does not exist and cannot be created')
                raise ValueError


    def save_face_data(self, silent:bool=False) -> bool:
        '''
        Pickle dumps dictionary of extracted face data
        :param silent: print success/failure message if False.
        '''
        if not self._type_check('silent', silent, bool):
            raise TypeError 
        try:
            with bz2.BZ2File(self.face_data_pkl, 'w') as file:
                pickle.dump(self.face_data_dict, file)
            if not silent:
                print(f'+++ Saved: {self.face_data_pkl}')
            return True
        except Exception as e:
            print(f'!!! Failed to save: {self.face_data_pkl}\n', e)
        return False


    def _load_face_data(self, silent=False) -> dict:
        '''
        Pickle load dictionary of extracted face data
        :param silent: print success/failure message if False.
        '''
        if not self._type_check('silent', silent, bool):
            raise TypeError 
        try:
            data = bz2.BZ2File(self.face_data_pkl, 'rb')
            result = pickle.load(data)
            if not silent:
                print(f'+++ Loaded: {self.face_data_pkl}')
                print(f'    Number of keys: {len(result.keys())}')
            return result
        except Exception as e:
            print(f'!!! Failed to load: {self.face_data_pkl}\n', e)


    def get_image_files(self, dir_path:str, include_sub_dirs=False) -> list:
        '''
        Return list of images in the passed directory
        :param dir_path: path to folder of images to be encoded
        :param include_sub_dirs: optionally recurse into nested folders
        '''
        if not self._type_check('dir_path', dir_path, str):
            raise TypeError 
        if os.path.exists(dir_path):
            if include_sub_dirs:
                if not dir_path.endswith('/'): dir_path = f'{dir_path}/'
                files_gen = (p.resolve() for p in Path(dir_path).glob("**/*") if p.suffix in {".jpeg", ".jpg", ".png", ".webp"})
                files = [f._str for f in list(files_gen)]
            else:
                files = [f'{dir_path}/{f}' for f in os.listdir(dir_path) if f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".png") \
                    or f.endswith(".webp")]
            return files
        else:
            print(f'!!! {dir_path} does not exist')
            return