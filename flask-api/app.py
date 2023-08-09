from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import torch
import base64
from PIL import Image
from io import BytesIO
import cv2
import numpy as np
import torch.nn as nn
import os

app = Flask(__name__)
CORS(app)


@app.route('/', methods=["POST"])
def main_func():
    print("fd")
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
    print(image.shape)

    # preprocess
    process_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshold, process_image = cv2.threshold(process_image, 0, 255, cv2.THRESH_OTSU)
    img_list = separate(process_image)

    # loading classifier
    classifier = joblib.load("captcha_reader.joblib")

    ans = ""
    print(len(img_list))
    for i in range(len(img_list)):
        val = classifier.output(img_list[i].unsqueeze(0).unsqueeze(0).to(dtype=torch.float32))
        out = torch.argmax(val)
        print(out)
        ans += int2label_dict[out.item()]
    print(ans)
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
    print(len(arr))
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

        final_img = torch.from_numpy(resized_image)
        crop_img.append(final_img)

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


# -------------- classifier model -----------
class Classifier(nn.Module):
    def __init__(self, n_labels, loader_size, lr):
        super().__init__()
        self.lr = lr
        self.n_labels = n_labels
        self.loader_size = loader_size
        # layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1)
        self.pool2 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.pool2 = nn.AvgPool2d(2)

        self.fc1 = nn.Linear(800, 256)

        self.fc2 = nn.Linear(256, n_labels)

        # loss function
        self.lossFn = nn.MSELoss()

        # optimizer
        self.optimizer = torch.optim.SGD(self.parameters(), self.lr)


    # output function
    def output(self,input):
        conv1_out = nn.ReLU()(self.conv1(input))
        pool1_out = self.pool2(conv1_out)
        conv2_out = nn.ReLU()(self.conv2(pool1_out))
        pool2_out = self.pool2(conv2_out)


        flatten_out = pool2_out.view(pool2_out.shape[0], -1)

        out = torch.empty(flatten_out.shape[0], self.n_labels)

        for i in range(flatten_out.shape[0]):
            fc1_out = nn.ReLU()(self.fc1(flatten_out[i]))
            
            fc2_out = self.fc2(fc1_out)

            softmax_out = nn.Softmax(dim=-1)(fc2_out)

            out[i] = softmax_out

        return out
    
    # training function
    def train(self, dataloader, n_epochs):
        for i in range(n_epochs):

            epoch_loss = 0
            

            for inputs, labels in dataloader:            
                inputs = inputs.unsqueeze(1)
                output = self.output(inputs)

                hot_enco = torch.empty(self.loader_size, self.n_labels)
                for j in range(self.loader_size):
                    hot_enco[j] = torch.zeros(self.n_labels)
                    hot_enco[j][labels[j]] = 1

                self.optimizer.zero_grad()

                loss = self.lossFn(hot_enco, output)

                epoch_loss += loss

                loss.backward()

                self.optimizer.step()
            print(f"loss at epoch {i+1}: {epoch_loss.item()}")
# ------------------------------


if __name__ == "__main__":
    app.run(host="0.0.0.0")