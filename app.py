from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")  # ✅ 正确返回 `text/html`

if __name__ == '__main__':
    app.run(debug=True, port=8000)
