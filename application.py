from flask import Flask, request, render_template
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import imutils
import easyocr
import json
import xmltodict
import requests
import glob

app = Flask(__name__)

def get_vehicle_info(plate_number):
    r = requests.get(
        "http://www.regcheck.org.uk/api/reg.asmx/CheckIndia?RegistrationNumber={0}&username=vv2000".format(str(plate_number)))
    data = xmltodict.parse(r.content)
    jdata = json.dumps(data)
    df = json.loads(jdata)
    df1 = json.loads(df['Vehicle']['vehicleJson'])
    return df1

def Key(dict, key):
    if key in dict.keys():
        print("Present, ", end=" ")
        print("value =", dict[key])
        return dict[key]
    else:
        print("Not present")
        return "Not present"

ALLOWED_EXTENSIONS = {'mp4'}
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

path = os.getcwd()
UPLOAD_FOLDER = os.path.join(path, 'uploads')

UPLOAD_FOLDER2 = os.path.join(path, 'output')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['UPLOAD_FOLDER2'] = UPLOAD_FOLDER2

@app.route("/")
def home():
    return render_template('Home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():

    global des, regyr, comp, mod, vid, seat, col, eng, ftype, regd, loc, vari, engs, own, ins, puc, veht


    if request.method == 'POST' and 'file' in request.files:
        for f in request.files.getlist('file'):
            file = secure_filename(f.filename)
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], file))


    for file_name in glob.iglob("uploads/**/*.mp4", recursive=True):
        cap = cv2.VideoCapture(file_name)
        count = 1
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                cv2.imwrite("./output/frame%d.jpg" % count, frame)
                count = count + 1
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            else:
                break
        cap.release()
        cv2.destroyAllWindows()

        car_image = cv2.imread("./output/frame%d.jpg" % (count - 20))
        img = car_image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bfilter = cv2.bilateralFilter(gray, 11, 17, 17)  # Noise reduction
        edged = cv2.Canny(bfilter, 30, 200)  # Edge detection
        keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(keypoints)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        location = None
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 10, True)
            if len(approx) == 4:
                location = approx
                break
        mask = np.zeros(gray.shape, np.uint8)
        (x, y) = np.where(mask == 255)
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))
        cropped_image = gray[x1:x2 + 1, y1:y2 + 1]
        reader = easyocr.Reader(['en'])
        result = reader.readtext(cropped_image)
        text = result[1][-2]
        text.replace(" ", "")

        #########################################################################

        x = get_vehicle_info(text)
        print(x)
        des = Key(x, 'Description')
        regyr = Key(x, 'RegistrationYear')
        comp = Key(x['CarMake'], 'CurrentTextValue')
        mod = Key(x['CarModel'], 'CurrentTextValue')
        vid = Key(x, 'VechileIdentificationNumber')
        seat = Key(x, 'NumberOfSeats')
        col = Key(x, 'Colour')
        eng = Key(x, 'EngineNumber')
        ftype = Key(x, 'FuelType')
        regd = Key(x, 'RegistrationDate')
        loc = Key(x, 'Location')
        vari = Key(x, 'Variant')
        engs = Key(x['EngineSize'], 'CurrentTextValue')
        own = Key(x, 'Owner')
        ins = Key(x, 'Insurance')
        puc = Key(x, 'PUCC')
        veht = Key(x, 'VehicleType')

    # EMPTY UPLOAD FOLDER
    BASE_DIR = os.getcwd()
    dir = os.path.join(BASE_DIR, "uploads")

    for root, dirs, files in os.walk(dir):
        for file in files:
            path = os.path.join(dir, file)
            os.remove(path)

    return render_template('Page-1.html', des=des, regyr=regyr, comp=comp, mod=mod, vid=vid, seat=seat, col=col, eng=eng, ftype=ftype, regd=regd, loc=loc,
                           vari=vari, engs=engs, own=own, ins=ins, puc=puc, veht=veht)


if __name__ == "__main__":
    app.run(debug=True)




