import numpy as np 
import struct 
from PIL import Image 
import os 
#训练集改成   size1 = 7840016, size2 = 10008, root = "test"
def make_img(fn_img = "train-images-idx3-ubyte\\train-images.idx3-ubyte", 
             fn_label = "train-labels-idx1-ubyte\\train-labels.idx1-ubyte", 
             size1 = 47040016, size2 = 60008, root = "train"):
    fmt_img1 = ">IIII"
    offset_img1 = offset_label1= 0
    fmt_label1 = ">II"
    data_file_size = str(size1 - 16) + "B"
    labels_file_size = str(size2 - 8) + "B"
    fmt_img2 = ">" + data_file_size
    offset_img2 = struct.calcsize(fmt_img1)
    fmt_label2 = ">" + labels_file_size
    offset_label2 = struct.calcsize(fmt_label1)
    with open(fn_img, 'rb') as f:
        data_buf = f.read()
    with open(fn_label, 'rb') as f:
        label_buf = f.read()
    magic_img, numImages, numRows, numColumns = struct.unpack_from(fmt_img1, data_buf, offset_img1) 
    datas = struct.unpack_from(fmt_img2, data_buf, offset_img2)
    datas = np.array(datas).astype(np.uint8).reshape(numImages, 1, numRows, numColumns)
    magic_label, numLabels = struct.unpack_from(fmt_label1, label_buf, offset_label1)
    labels = struct.unpack_from(fmt_label2, label_buf, offset_label2)
    labels = np.array(labels).astype(np.int64)
    if not os.path.exists(root): 
        os.mkdir(root) 
    for i in range(10): 
        file_name = root + os.sep + str(i) 
        if not os.path.exists(file_name): 
            os.mkdir(file_name) 
    for ii in range(numLabels): 
        img = Image.fromarray(datas[ii, 0, 0:28, 0:28]) 
        label = labels[ii] 
        file_name = root + os.sep + str(label) + os.sep + str(ii).zfill(5) + '.png' 
        img.save(file_name)
if __name__ == "__main__":
    make_img()
 
