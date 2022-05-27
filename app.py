### IMPORTS ###
from flask import Flask, request, jsonify
import json
from flask import render_template,flash,redirect,session,url_for,abort
from flask_wtf import FlaskForm
from wtforms import StringField,SubmitField
from wtforms.validators import DataRequired
from flask_sqlalchemy import SQLAlchemy
from bs4 import BeautifulSoup
from markupsafe import Markup

import os
import time
import numpy as np
import pandas as pd
from logic import preprocess,create_tfidf_features,calculate_similarity,show_similar_documents,df,html_codes,titles,urls
import html

basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)

app.config['SECRET_KEY']='mysecret'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir,'data.sqlite')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db=SQLAlchemy(app)

### FORMS ###
class MyForm(FlaskForm):
    search_query = StringField(validators=[DataRequired()])
    submit = SubmitField('submit')

### MODELS ###
class Example(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String())
    keywords = db.Column(db.String())
    url = db.Column(db.String())
    html = db.Column(db.String())


### VIEWS ###
#Homepage
@app.route('/', methods=['GET','POST'])
def index():
    if request.method=='POST':
        val = request.form.get('search_query')
        return redirect('/search/'+val)
    return render_template('index.html')

#Search page
# @app.route('/search',methods=['GET','POST'])
# def search():
#     form = MyForm()
#     if form.validate_on_submit():
#         # flash('Query Submitted!')
#         query = form.search_query.data
#         session['query']=query
#         return redirect('/search/'+query)
#     # form = MyForm(FlaskForm)
#     # if request.method == 'POST' and form.validate():
#     #     query = form.name.search_query
#     #     print(query)
#     return render_template('search.html',form=form)

#Search results page
@app.route('/search/<qry>',methods=['GET', 'POST'])
def searching(qry):
    print(qry)
    print(preprocess(qry))
    print('Reading the corpus...')
    print('done')
    # sample_index = np.random.randint(len(df))
    # sample = df.loc[sample_index,['Title', 'Keywords']]
    # print('title: {}, \nbody: {}'.format(sample['Title'],sample['Keywords']))
    print('creating tfidf matrix...')
    data = [(body+' '+title).lower() for title, body in zip(df['title'], df['keywords'])]
    # Learn vocabulary and idf, return term-document matrix
    X,v = create_tfidf_features(data)
    features = v.get_feature_names()
    user_question = [preprocess(qry)]
    search_start = time.time()
    sim_vecs, cosine_similarities = calculate_similarity(X, v, user_question)
    search_time = time.time() - search_start
    print("search time: {:.2f} ms".format(search_time * 1000))
    # print()
    d = show_similar_documents(data, cosine_similarities, sim_vecs)
    # return render_template('searching.html',qry=qry)
    print(d)
    return render_template('searching.html',qry=qry,d=d,search_time=search_time)

@app.route('/problem/<problem_id>')
def problem_page(problem_id):
    if(int(problem_id)<0 or int(problem_id)>len(titles)):
        abort(404, description="Resource not found")
    html_code = Markup(html_codes[int(problem_id)])
    return render_template('page.html',title=titles[int(problem_id)],html_code=html_code,url=urls[int(problem_id)])


# Error Pages
@app.errorhandler(404)
def page_not_found(e):
    return render_template('error_pages/404.html'),404

@app.errorhandler(403)
def page_not_found(e):
    return render_template('error_pages/403.html'),403

if __name__ == '__main__':
    # Bind to PORT if defined, otherwise default to 5000.
    # port = int(os.environ.get('PORT', 5000))
    # app.run(host='0.0.0.0', port=port)
    app.run()
