from flask import Flask, render_template, request
# from werkzeug import secure_filename
from detect_license_plate import read_license_plate 

app = Flask(__name__)

@app.route("/handle_data", methods=["POST"])
def handle_data():
    if request.method == 'POST': 
        # fetch data from request form
        
        f = request.files['imgfile']
        f.save(f.filename)
        print(f.filename)
        imgfile = f.filename
        
        method = request.form["method"]

        license_plate = ""

        if method == "1":
            if imgfile is not None:
                license_plate = read_license_plate(imgfile)
    
        return render_template("index.html", license_plate=license_plate)
    else:
        return render_template("index.html", minerals="", chemical_composition="")

@app.route("/")
def index():
    return render_template("index.html", minerals="", chemical_composition="")