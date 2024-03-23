from flask import Flask, request, send_file, render_template
import os
import cv2
from PIL import Image
import numpy as np
import torch

from DataLoader import SegmentDataset
from torch.utils.data import Subset, Dataset, DataLoader

from multiprocessing import freeze_support

from myUnet import myUnet

import shutil

model_path = 'model/unet_model.pt'
model = torch.load(model_path)

images_path = 'images/'
uploads_path = "uploads/"

app = Flask(__name__, static_folder="uploads")
app.config['UPLOAD_FOLDER'] = 'uploads'


@app.route('/', methods=['GET', 'POST'])
def upload_image():

    shutil.rmtree(images_path)
    shutil.rmtree(uploads_path)
    os.mkdir(images_path)
    os.mkdir(uploads_path)

    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"

        file = request.files['file']

        if file.filename == '':
            return "No selected file"

        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        img = Image.open(filename)
        img2 = img.copy()

        img2.save(os.path.join(images_path, file.filename))

        ims = SegmentDataset(dirPath=r'', imageDir=images_path, masksDir=images_path)
        bb = DataLoader(ims, batch_size=1)
        iterator = iter(bb)

        k = next(iterator)

        mg = model(k[0].to("cuda").float()).argmax(axis=1).cpu().numpy()[0]

        cv2.imwrite((images_path+'mask.jpg'), mg)

        mask = cv2.imread((images_path+'mask.jpg'))

        mask = cv2.resize(mask, img.size)

        mask[mask != 0] = 1

        out_img = np.multiply(np.asarray(img), mask)

        output_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'combined_' + file.filename)

        out = Image.fromarray(out_img)
        out.save(output_filename)

        return render_template('result.html', original_file=file.filename, combined_file='combined_' + file.filename)

    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
        <input type=file name=file>
        <input type=submit value=Upload>
    </form>
    '''


if __name__ == '__main__':
    freeze_support()

    app.run(debug=True)




