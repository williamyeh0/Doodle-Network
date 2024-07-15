from flask import render_template
from app import app

@app.route('/')
@app.route('/paint')
def index():
    user = {'username': 'Miguel'}
    return render_template('paint.html', title='Home', user=user)