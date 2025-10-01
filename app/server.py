from flask import Flask, request, render_template_string
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model")))

from model.ascii_converter import image_to_ascii

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    ascii_art = ""
    if request.method == "POST":
        file = request.files["image"]
        path = "uploaded_image.jpg"
        file.save(path)
        ascii_art = image_to_ascii(path, new_width=80)
    return render_template_string("""
        <h1>ASCII Art Generator</h1>
        <form method="post" enctype="multipart/form-data">
            <input type="file" name="image">
            <input type="submit" value="Convert">
        </form>
        <pre>{{ascii_art}}</pre>
    """, ascii_art=ascii_art)

if __name__ == "__main__":
    app.run(debug=True)
