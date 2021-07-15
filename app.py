from flask import Flask, render_template, Response,request
import cv2
import numpy as np
import face_recognition
import os
#Initialize the Flask app
app = Flask(__name__,template_folder = 'tamplates/')

camera = cv2.VideoCapture(0)

path = 'ImagesSet'
images = []         #list of images in the path file
classNames = []     #list of images without extension
myList = os.listdir(path)
#print(myList)
imgName = ""
for name in myList:
    curImg = cv2.imread(f'{path}/{name}')
    images.append(curImg)
    classNames.append(os.path.splitext(name)[0])
#print(classNames)

def get_frames_for_registeration():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def getEncoding(images):
    encodeList =[]
    for i in images:
        i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(i)[0]
        encodeList.append(encode)
    return encodeList

finalEncoding = getEncoding(images) #final encoded list of images in file

def gen_frames():
    while True:
        success, img = camera.read()  # read the camera frame
        if not success:
            break
        else:
            imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)  # 1/4th size
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

            facesCurFrame = face_recognition.face_locations(imgS)  # (top, right, bottom, left)
            encodeCurFace = face_recognition.face_encodings(imgS, facesCurFrame)

            for encodeFace, faceLoca in zip(encodeCurFace, facesCurFrame):
                matches = face_recognition.compare_faces(finalEncoding, encodeFace)
                faceDis = face_recognition.face_distance(finalEncoding, encodeFace)
                matchIndex = np.argmin(faceDis)

                if matches[matchIndex]:
                    person_name = classNames[matchIndex].upper()
                    imgName = person_name
                    #print(imgName)
                    t, r, b, l = faceLoca
                    t, r, b, l = t * 4, r * 4, b * 4, l * 4  # rezise it from 1/4th to full
                    start = (l, t)
                    end = (r, b)
                    cv2.rectangle(img, start, end, (255, 0, 0), 2)
                    cv2.rectangle(img, (l, b - 30), (r, b), (255, 0, 0), cv2.FILLED)
                    cv2.putText(img, person_name, (l + 5, b - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                else:
                    t, r, b, l = faceLoca
                    t, r, b, l = t * 4, r * 4, b * 4, l * 4  # rezise it from 1/4th to full
                    cv2.rectangle(img, (l, t), (r, b), (0, 0, 255), 2)
                    cv2.rectangle(img, (l, b - 30), (r, b), (0, 0, 255), cv2.FILLED)
                    cv2.putText(img, 'Unknown', (l + 5, b - 5), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

            # cv2.imshow('Webcam', img)
            # cv2.waitKey(1)

            ret, buffer = cv2.imencode('.jpg', img)
            img = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')  # concat frame one by one and show result


@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('home.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/register')
def register():
    return render_template('register.html')
    #return Response(get_frames_for_registeration(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/adduser', methods=['GET', 'POST'])
def adduser():
    success, frame = camera.read()
    name = request.form['Uname']
    if name != "":
        if name not in classNames:
            #os.mkdir('E:\ResoluteAI\Face recognition task\ImagesSet/')
            cv2.imwrite('E:\ResoluteAI\Face recognition task\ImagesSet/' + name.lower() + '.jpg', frame)
        return render_template('adduser.html', pred= '{}'.format(name))
    else:
        return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    print(imgName)
    return render_template('login.html',pred= ('WELCOME '+ imgName))

if __name__ == "__main__":
    app.run(debug=True)