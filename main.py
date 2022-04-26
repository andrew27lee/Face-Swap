import os, base64, re
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
from flask import Flask, request, render_template
from flask_sock import Sock

from face_detection import select_face, select_all_faces
from face_swap import face_swap

SOURCE_PATH = 'static/images/input/src.jpg'
IMAGE_DESTINATION_PATH = 'static/images/input/dst.jpg'
IMAGE_RESULT_PATH = 'static/images/output/swapped.jpg'

app = Flask(__name__)
sock = Sock(app)


def face_exists(img_path):
    img = cv2.imread(img_path)
    points, _, _ = select_face(img)

    if points is None:
        return 'No face was detected.'
    return None


def __image_swap():
    src_img = cv2.imread(SOURCE_PATH)
    dst_img = cv2.imread(IMAGE_DESTINATION_PATH)
    
    try:
        src_points, _, src_face = select_face(src_img)
        dst_faceBoxes = select_all_faces(dst_img)

        if src_points is None or dst_faceBoxes is None:
            return 'No face was detected.'

        output = dst_img

        for k, dst_face in dst_faceBoxes.items():
            output = face_swap(src_face, dst_face["face"], src_points,
                            dst_face["points"], dst_face["shape"],
                            output)

        cv2.imwrite(IMAGE_RESULT_PATH, output)
        return None
    except:
        return "An error has occurred."


def __live_swap(dst_img):
    src_img = cv2.imread(SOURCE_PATH)

    try:
        src_points, _, src_face = select_face(src_img)
        dst_faceBoxes = select_all_faces(dst_img)

        output = dst_img

        for k, dst_face in dst_faceBoxes.items():
            output = face_swap(src_face, dst_face["face"], src_points,
                            dst_face["points"], dst_face["shape"],
                            output)
        return output
    except:
        pass


@app.route('/image/swap', methods=['POST'])
def image_swap():
    error = None
    src = request.files['src']
    dst = request.files['dst']

    if src.filename == '' or dst.filename == '':
        return render_template('image_swap.html', error='Input file(s) not selected.')

    src.save(os.path.join(app.root_path, SOURCE_PATH))
    dst.save(os.path.join(app.root_path, IMAGE_DESTINATION_PATH))
    error = __image_swap()
    
    if error:
        return render_template('image_swap.html', error=error)
    return render_template('image_swap.html', image=True)


@app.route('/live/swap', methods=['POST'])
def live_swap():
    error = None
    src = request.files['src']

    if src.filename == '':
        return render_template('live_swap.html', error='Source file not selected.')

    src.save(os.path.join(app.root_path, SOURCE_PATH))
    error = face_exists(SOURCE_PATH)

    if error:
        return render_template('live_swap.html', error=error)
    return render_template('webcam.html')


@sock.route('/echo', methods=['GET'])
def echo(ws):
    while True:
        dst_data = ws.receive()

        if 'base64,' in dst_data:
            try:
                dst_base64_string = re.sub('^data:image/.+;base64,', '', dst_data)
                dst_img = Image.open(BytesIO(base64.b64decode(dst_base64_string))).convert('RGB')
                swapped = __live_swap(np.asarray(dst_img))
                output = Image.fromarray(swapped)
                buffered = BytesIO()
                output.save(buffered, format='png')
                ws.send(base64.b64encode(buffered.getvalue()).decode('ascii'))
            except:
                pass
        else:
            continue


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/blog', methods=['GET'])
def blog():
    return render_template('blog.html')


@app.route('/image', methods=['GET'])
def image():
    return render_template('image_swap.html')


@app.route('/live', methods=['GET'])
def live():
    return render_template('live_swap.html')


if __name__ == '__main__':
    app.run(port=5000, host='0.0.0.0')
