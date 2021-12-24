from flask import Flask,jsonify,request
from yolov3_ANPR_flask import ANPR
from flask_ngrok import run_with_ngrok
import json

app = Flask(__name__, static_url_path = "", static_folder = r"C:\Users\Student\Desktop\ANPR\yolov3_ANPR")
app.config['JSON_AS_ASCII'] = False
@app.route('/')
def detect():
    img_path=request.args['image']
    results=ANPR(img_path)
    return json.dumps(results,ensure_ascii=False)
    # return jsonify(results)

run_with_ngrok(app)

if __name__ == "__main__":
    app.run()