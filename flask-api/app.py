from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import base64
from PIL import Image
from io import BytesIO
import cv2
import onnxruntime

app = Flask(__name__)
CORS(app)

ort_session = onnxruntime.InferenceSession("captcha_reader_model.onnx")

@app.route('/', methods=["POST"])
def main_func():
    # get image data
    data = request.get_json()
    base64_data = data.get('image', None)

    if base64_data == None:
        return jsonify({"captcha_value":"Image Not found"})

    # changing base64 img data to image array
    _, encoded_data = base64_data.split(',', 1)
    image_bytes = base64.b64decode(encoded_data)
    try:
        image = Image.open(BytesIO(image_bytes))
    except:
        return jsonify({"captcha_value": "Image Not found"})
    image = np.array(image)

    # preprocess
    process_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshold, process_image = cv2.threshold(process_image, 0, 255, cv2.THRESH_OTSU)
    img_list = separate(process_image)
    # loading classifier

    ans = ""
    for i in range(len(img_list)):

        np_arr = img_list[i].reshape(1, 1, 28, 28)
        ort_inputs = {'input.1': np_arr.astype('float32')}

        val = ort_session.run(None, ort_inputs)
        out = np.argmax(val)
        ans += int2label_dict[out.item()]
    return jsonify({"captcha_value": ans})

    

# ------- function to separate each character ---------
def separate(full_image):
    # vertical cutting
    prev = False
    prev2 = False
    arr = []
    for i in range(full_image[0].shape[0]):

        if np.all(full_image[:, i] < 127) and not prev:
            arr.append(i)
            prev = True
            prev2 = False
        elif not np.all(full_image[:, i] < 127) and not prev2:
            arr.append(i)
            prev2 = True
            prev = False
    # --------------
    crop_img = []

    for i in range((len(arr) - 1) // 2):
        image = full_image[:, arr[2*i + 1]: arr[2*i + 2]]

        # horizontal cutting
        prev = False
        prev2 = False
        arr2 = []
        for j in range(image.shape[0]):
            if np.all(image[j, :] < 127) and not prev:
                arr2.append(j)
                prev = True
                prev2 = False

            elif not np.all(image[j, :] < 127) and not prev2:
                arr2.append(j)
                prev2 = True
                prev = False

        image = image[arr2[1]: arr2[-1], :]
        # --------------

        # padding
        pad_width = int((image.shape[0] - image.shape[1]) / 2)
        pad_size = 5
        if pad_width > 0:
            image = np.pad(image, ((pad_size, pad_size), (pad_width + pad_size, pad_width + pad_size)), mode="constant")
        
        else:
            image = np.pad(image, ((pad_size - pad_width, pad_size - pad_width), (pad_size, pad_size)), mode="constant")
        # ----------
        
        resized_image = cv2.resize(image, (28, 28), cv2.INTER_CUBIC)
        crop_img.append(resized_image)

    return crop_img

# dictionary to change label to int and int to label
label2int_dict = {}
for i in range(62):
    if i < 10:
        label2int_dict[str(i)] = i
    
    elif i <36:
        label2int_dict[chr(97 + i - 10)] = i
    
    else:
        label2int_dict[chr(65 + i - 36)] = i

int2label_dict = {}
for key in label2int_dict:
    int2label_dict[label2int_dict[key]] = key




if __name__ == "__main__":
    app.run(host="0.0.0.0")