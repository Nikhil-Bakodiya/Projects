from flask import Flask
import pip
app=Flask(__name__)

@app.route('/') #('/hello'(name))
def hello_world():
    return 'Hello Minkowski'
if __name__=='__main__':
    app.run()

