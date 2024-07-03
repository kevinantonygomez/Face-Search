# Face-Search
## About
Leverage ML to find images with similar faces on your local machine.

## Set-up
1. Clone this repo
2. Unzip the data/.bz2 files
3. Ensure the following have been installed:
   - dlib (https://github.com/davisking/dlib)
   - tqdm (https://github.com/tqdm/tqdm)
   - OpenCV (https://pypi.org/project/opencv-python/)
   - NumPy (https://github.com/numpy/numpy)
   - bz2file (https://pypi.org/project/bz2file/)
5. Take a look at the example user code in main.py
   - You can choose to use a HOG+SVM or a Max-Margin (MMOD) CNN face detector by setting the 'detector_type' argument to the Driver in main.py.
   - The HOG+SVM detector provides accurate results and is computationally efficient. The CNN provides relatively better results but requires dlib with CUDA enabled to utilize your GPU

## How to run
1. Having filed in the required paths in main.py, run main.py from the root directory of this repo
2. The HTML file containing the results is saved to the data folder by default as 'results.html'.
