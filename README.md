# FaceSwap
[![Run on Ainize](https://ainize.ai/static/images/run_on_ainize_button.svg)](https://ainize.ai/andrew27lee/FaceSwap?branch=master)

Swap the face from one image onto the face(s) of another image or your own face using Python 3 with OpenCV and dlib.

## Deploy Using Docker
```
$ docker build -t <CUSTOM_IMAGE_NAME> .
$ docker run -d -p 5000:5000 <CUSTOM_IMAGE_NAME>
```

## Examples
<table>
    <thead>
        <tr>
            <td>Source</td>
            <td>Destination</td>
            <td>Result</td>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><img src="static/images/input/test1.jpg" width="180" height="240"/></td>
            <td><img src="static/images/input/test2.jpg" width="180" height="240"/></td>
            <td><img src="static/images/output/test1_2.jpg" width="180" height="240"/></td>
        </tr>
        <tr>
            <td><img src="static/images/input/test3.jpg" width="180" height="240"/></td>
            <td><img src="static/images/input/test4.jpg" width="180" height="240"/></td>
            <td><img src="static/images/output/test3_4.jpg" width="180" height="240"/></td>
        </tr>
        <tr>
            <td><img src="static/images/input/test5.jpg" width="180" height="240"/></td>
            <td><img src="static/images/input/test6.jpg" width="180" height="240"/></td>
            <td><img src="static/images/output/test5_6.jpg" width="180" height="240"/></td>
        </tr>
    </tbody>
</table>