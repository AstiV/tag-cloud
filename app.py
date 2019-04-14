from flask import Flask, render_template, request
import csv

# Configure application
app = Flask(__name__)

# Routing
@app.route('/')
def index():
    # register a new dialect to define custom delimiter that is not ',' (so ';' is considered as column separator)
    csv.register_dialect('myDialect', delimiter = ';')
    
    # read csv data
    with open('./Datensatz_Coding_Challenge.csv', 'r', encoding='latin1') as csvFile:
        reader = csv.reader(csvFile, dialect='myDialect')
        for row in reader:
            print(row)

    csvFile.close()
    return render_template('index.html')

#TODO change this for production! (debug=True only for dev mode)
if __name__ == '__main__':
    app.run(debug=True)

