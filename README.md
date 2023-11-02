# AIMS_captcha_autofill

Autofills the captcha of IIT Hyderabad AIMS portal using image segmentation followed by letter and digit recognition.


### Installation

Download the aims_captcha_reader folder and load as unpacked in browser.



If you want to test the whole code then clone the repository, load the aims_captcha_download extension in browser and open https://aims.iith.ac.in/aims/ to download the data images.

Then you have to create a virtual environment. You can see online on how to do so

Then run following command to download necessary packages
```bash
pip install -r requirements.txt
```

Then run .ipynb file in flask-api folder which will train the model.

Now run the flask app app.py using python3 app.py. This will start your server.

Now last step is to load the aims_captcha_reader extension.
you are good to go
