from flask import Flask,url_for,redirect,request
app = Flask(__name__)

@app.route('/admin')
def hello_admin():
    return 'Hello Admin'

@app.route('/guest/<name>')
def hello_user(name):
    return f'Hello {name} as guest'

@app.route('/user/<name>')
def hello(name):
    if name == 'admin':
        return redirect(url_for('hello_admin'))
    else:
        return redirect(url_for('hello_user',name=name))

@app.route('/sucess/<name>')
def sucess(name):
    return f'welcome {name}'

@app.route('/login',methods = ['POST','GET '])
def login():
    if request.method == 'POST':
        user = request.form['nm']
        return redirect(url_for('sucess',name=user))
    else:
        user = request.args.get('nm')
        return redirect(url_for('sucess',name=user))


if __name__ == '__main__':
   app.run(debug=True)