from tqdm import tqdm
import face_encoder as face_encoder
import file_handler as file_handler

class Driver:
  '''
  Drives program and is handled by user. Provides methods to batch encode multiple images 
  in a given folder at once, and to encode single image files directly
  '''
  def __init__(self, face_encodings_dict_path:str) -> None:
    '''
    :param face_encodings_dict_path: path to the compressed pickled dictionary 
      that stores/should store computed encodings.
    '''
    self.face_encoder = face_encoder.Face_Encoder()
    self.file_handler = file_handler.File_Handler(face_encodings_dict_path)
  
  def encode(self, img_path:str, upsample_times:int=0) -> list:
    '''
    :param img_path: path to image to be encoded
    :param upsample_times: optionally upsample image prior to encoding 
    '''
    return self.face_encoder.get_face_encodings(img_path, upsample_times)

  def batch_encode(self, dir_path:str, upsample_times:int=0) -> None:
    '''
    :param dir_path: path to folder of images to be encoded
    :param upsample_times: optionally upsample images prior to encoding 
    '''
    files = self.file_handler.get_image_files(dir_path)
    for file in tqdm(files):
      face_encodings = self.face_encoder.get_face_encodings(file, upsample_times)
      self.file_handler.face_encodings_dict[file] = face_encodings
    self.file_handler.save_face_encodings()


# user code here
if __name__ == '__main__':
  driver = Driver('data/face_encodings_dict')
  driver.batch_encode('/path/to/image/folder/')
  driver.batch_encode('/path/to/another/image/folder/', upsample_times=1)
  driver.encode('/path/to/image', upsample_times=2)