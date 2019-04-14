from flask import Flask, render_template, request

# Configure application
app = Flask(__name__)

# Routing
@app.route('/')
def index():
    return render_template('index.html')

#TODO change this for production! (debug=True only for dev mode)
if __name__ == '__main__':
    app.run(debug=True)

