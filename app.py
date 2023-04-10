import sys
import os, shutil
import glob
import re
import numpy as np
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Flask utils
from flask import Flask,flash, request, render_template,send_from_directory, url_for, redirect
from werkzeug.utils import secure_filename
# Define a flask app
app = Flask(__name__)

RESULT_FOLDER = os.path.join('static', 'mask_images')
app.config['MASK_FOLDER'] = RESULT_FOLDER
app.config['UPLOAD_FOLDER'] = 'uploads'


def findMask(img):
    quantization = 16 # Quantization is a process to convert the continuous analog signal to the series of discrete values, here 16 bit depth
    tsimilarity = 5 # euclid distance similarity threshhold
    tdistance = 20 # euclid distance between pixels threshold
    vector_limit = 20 # shift vector elimination limit
    block_counter = 0
    block_size = 8
    image = cv2.imread(img)
    # mask = cv2.imread('forged1_mask.png')
    # mask_gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    temp = []
    arr = np.array(gray)
    # mask = np.array(mask_gray)
    prediction_mask = np.zeros((arr.shape[0], arr.shape[1]))
    column = arr.shape[1] - block_size
    row = arr.shape[0] - block_size
    dcts = np.empty((((column+1)*(row+1)), quantization+2))

    # DCT coefficients represent the frequency components of an image. 
    # These coefficients are calculated by applying the DCT to each block of pixels in an image. 
    # The resulting DCT coefficients are used to represent the block in a compressed format, by 
    # retaining only the most significant coefficients and discarding the rest.
    
    print("scanning & dct starting...")

    for i in range(0, row):
        for j in range(0, column):
    
            blocks = arr[i:i+block_size, j:j+block_size]
            imf = np.float32(blocks) / 255.0  # float conversion/scale
            dst = cv2.dct(imf)  # the dct
            blocks = np.uint8(np.float32(dst) * 255.0 ) # convert back
            # zigzag scan
            solution = [[] for k in range(block_size + block_size - 1)]
            for k in range(block_size):
                for l in range(block_size):
                    sum = k + l
                    if (sum % 2 == 0):
                        # add at beginning
                        solution[sum].insert(0, blocks[k][l])
                    else:
                        # add at end of the list
                        solution[sum].append(blocks[k][l])
    
            for item in range(0,(block_size*2-1)):
                temp += solution[item]
    
            temp = np.asarray(temp, dtype=np.float)
            temp = np.array(temp[:16])
            temp = np.floor(temp/quantization)
            temp = np.append(temp, [i, j])
    
            np.copyto(dcts[block_counter], temp)
    
            block_counter += 1
            temp = []
    
    print("scanning & dct over!")
    # correlates the pixel values and reconstructs the correlation
    print("lexicographic ordering starting...")
    
    dcts = dcts[~np.all(dcts == 0, axis=1)]
    dcts = dcts[np.lexsort(np.rot90(dcts))]
    
    print("lexicographic ordering over!")
    
    print("euclidean operations starting...")

    sim_array = []
    for i in range(0, block_counter):
        if i <= block_counter-10:
            for j in range(i+1, i+10):
                pixelsim = np.linalg.norm(dcts[i][:16]-dcts[j][:16])
                pointdis = np.linalg.norm(dcts[i][-2:]-dcts[j][-2:])
                if pixelsim <= tsimilarity and pointdis >= tdistance:
                    sim_array.append([dcts[i][16], dcts[i][17], dcts[j][16], dcts[j][17],dcts[i][16]-dcts[j][16], dcts[i][17]-dcts[j][17]])
        else:
            for j in range(i+1, block_counter):
                pixelsim = np.linalg.norm(dcts[i][:16]-dcts[j][:16])
                pointdis = np.linalg.norm(dcts[i][-2:]-dcts[j][-2:])
                if pixelsim <= tsimilarity and pointdis >= tdistance:
                    sim_array.append([dcts[i][16], dcts[i][17], dcts[j][16], dcts[j][17],dcts[i][16]-dcts[j][16], dcts[i][17]-dcts[j][17]])
    
    print("euclidean operations over!")
    
    print("elimination starting...")

    sim_array = np.array(sim_array)
    delete_vec = []
    vector_counter = 0
    for i in range(0, sim_array.shape[0]):
        for j in range(1, sim_array.shape[0]):
            if sim_array[i][4] == sim_array[j][4] and sim_array[i][5] == sim_array[j][5]:
                vector_counter += 1
        if vector_counter < vector_limit:
            delete_vec.append(sim_array[i])
        vector_counter = 0
    
    delete_vec = np.array(delete_vec)
    delete_vec = delete_vec[~np.all(delete_vec == 0, axis=1)]
    delete_vec = delete_vec[np.lexsort(np.rot90(delete_vec))]
    
    for item in delete_vec:
        indexes = np.where(sim_array == item)
        unique, counts = np.unique(indexes[0], return_counts=True)
        for i in range(0, unique.shape[0]):
            if counts[i] == 6:
                sim_array = np.delete(sim_array,unique[i],axis=0)
    
    print("elimination over!")
    
    print("painting starting...")

    for i in range(0, sim_array.shape[0]):
        index1 = int(sim_array[i][0])
        index2 = int(sim_array[i][1])
        index3 = int(sim_array[i][2])
        index4 = int(sim_array[i][3])
        for j in range(0,7):
            for k in range(0,7):
                prediction_mask[index1+j][index2+k] = 255
                prediction_mask[index3+j][index4+k] = 255
    
    print("painting over!")
    
    return prediction_mask
    

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'static/uploads', secure_filename(f.filename))
        print(file_path)
        f.save(file_path)
        file_name=os.path.basename(file_path)
        
        # reading the uploaded image
        
        # img = cv2.imread(file_path)
        # maskImg = findMask(file_path)
        maskImg = findMask(file_path)
        
        # file_path1 = os.path.join(basepath, 'static/mask_images', secure_filename(maskImg.filename))
        # maskImg.save(file_path1)
        cv2.imwrite("static/mask_images/mask.png",maskImg)
        
        # full_filename = os.path.join(app.config['MASK_FOLDER'], 'mask.png')
        
        return render_template("imageRender.html")
    return render_template("index.html")
        
if __name__ == '__main__':
        app.run(debug=True, host="localhost")