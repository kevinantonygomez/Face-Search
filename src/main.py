from tqdm import tqdm
import model
import file_handler

class Driver:
  '''
  Drives program and is handled by user. Provides methods to batch encode multiple images 
  in a given folder at once, and to encode single image files directly
  '''
  def __init__(self, face_data_dict:str, detector_type:str='svm') -> None:
    '''
    :param face_data_dict: path to the compressed pickled dictionary 
      that stores/should store extracted face data.
    :param detector_type: "svm" or "cnn"
    '''
    self.model = model.Model(detector_type)
    self.file_handler = file_handler.FileHandler(face_data_dict)
  
  def extract_faces(self, img_path:str, upsample_times:int=0) -> model.FaceData:
    '''
    :param img_path: path to image to extract facial data from
    :param upsample_times: optionally upsample image prior to encoding 
    '''
    return self.model.get_face_data(img_path, upsample_times)

  def batch_extract_faces(self, dir_path:str, upsample_times:int=0, include_sub_dirs:bool=False) -> None:
    '''
    :param dir_path: path to folder of images to extract facial data from
    :param upsample_times: optionally upsample images prior to encoding 
    :param include_sub_dirs: optionally recurse into nested folders
    '''
    files = self.file_handler.get_image_files(dir_path, include_sub_dirs)
    for file in tqdm(files):
      face_data = self.model.get_face_data(file, upsample_times)
      self.file_handler.face_data_dict[file] = face_data
    self.file_handler.save_face_data()

# Example user code
if __name__ == '__main__':
  '''init driver with path to pickled dictionary. Creates one if it does not exist'''
  driver = Driver('path/to/dict/to/use.pbz2')

  '''comparing two images'''
  face_data_1 = driver.extract_faces('/path/to/image/1')
  face_data_2 = driver.extract_faces('/path/to/image/2')
  res = driver.model.test_similarity(face_data_1.face_encodings, face_data_2.face_encodings, euclidean_thres=0.6, cosine_thres=0.92, silent=True)
  if res == True:
    print("Images contain a similar face")
  
  '''extracting facial data in batches'''
  driver.batch_extract_faces('/path/to/image/folder') # extracted data stored in driver.file_handler.face_data_dict
  target_img_key = 'target-img-key' # key to image in driver.file_handler.face_data_dict. Find images with similar faces to this key
  top_k = 10 # attempt to render top_k similar images
  metrics_dict = driver.model.find_similarities(target_img_key, driver.file_handler.face_data_dict) # compare target image to all images in face_data_dict
  driver.model.render_similar_images(target_img_key, driver.file_handler.face_data_dict, metrics_dict, top_k)