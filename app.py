from flask import Flask, render_template
from generate import generate

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/services')
def services():

    return render_template("services.html")

@app.route('/single')
def single():
    generate()
    return render_template("single.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

if __name__ == "__main__":
    app.run()