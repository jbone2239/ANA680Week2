from flask import Flask, render_template, request
import pickle

# Load the trained KNN model
model = pickle.load(open("knnmodel.pkl", "rb"))

# create my Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get input values from the form
    clump_thickness = int(request.form["clump_thickness"])
    uniformity_of_cell_size = int(request.form["uniformity_of_cell_size"])
    uniformity_of_cell_shape = int(request.form["uniformity_of_cell_shape"])
    marginal_adhesion = int(request.form["marginal_adhesion"])
    single_epithelial_cell_size = int(request.form["single_epithelial_cell_size"])
    bare_nuclei = int(request.form["bare_nuclei"])
    bland_chromatin = int(request.form["bland_chromatin"])
    normal_nucleoli = int(request.form["normal_nucleoli"])
    mitoses = int(request.form["mitoses"])

    # Organize inputs into a 2D list
    input_features = [[
        clump_thickness,
        uniformity_of_cell_size,
        uniformity_of_cell_shape,
        marginal_adhesion,
        single_epithelial_cell_size,
        bare_nuclei,
        bland_chromatin,
        normal_nucleoli,
        mitoses
    ]]

    # Predict using the model
    prediction = model.predict(input_features)[0]

    # Interpret the prediction
    result = "Benign" if prediction == 2 else "Malignant"

    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True, port=8000)
