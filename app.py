

import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import sqlite3
import flask
import pandas as pd
import numpy as np
import os
import json
from pickle import loads 

server = flask.Flask('app')
server.secret_key = os.environ.get('secret_key', 'secret')

def load_data(file='summary-led7digit.db'):
    path="./Script/database"
    db = sqlite3.connect(os.path.join(path,file))
    with db:
        df=pd.read_sql_query("SELECT heuristic,evaluation,sampling,weight,balance,test_avg FROM heuristic_info",db)
        _metric_ = ['recall', 'auc', 'sar', 'gm']
        df.loc[df.evaluation == df.heuristic, 'heuristic'] = 'direct'
        df = df[~df['heuristic'].isin(_metric_)]
    db.close()
    return df

def load_process(file, heuristic, balance, weight, sample, evaluation):
    if heuristic == 'direct': heuristic = evaluation
    path="./Script/database"
    db = sqlite3.connect(os.path.join(path,file))
    with db:
        _data_=pd.read_sql_query("""
            SELECT object FROM heuristic_info where heuristic = '%s'
            and balance = %d and weight = %d and sampling = '%s' and evaluation='%s'
        """ %(heuristic, balance, weight, sample, evaluation),db)
    db.close()
    tmp = np.zeros((10,2,30))
    for i, data in enumerate(_data_.values):
        data = loads(data[0])
        tmp[i][0] = data.cv
        tmp[i][1] = data.test
    return tmp.mean(axis=0),tmp.std(axis=0), data.key

app = dash.Dash('app', server=server)

# app.css.config.serve_locally = True
# app.scripts.config.serve_locally = True
dcc._js_dist[0]['external_url'] = 'https://cdn.plot.ly/plotly-basic-latest.min.js'

app.layout = html.Div([
    html.Div([
        
        html.Div([
            html.P('File'),
            dcc.Dropdown(
                id='dash-file',
                options=[
                    {'label': 'glass-1 (1.82)', 'value': 'summary-glass-1.db'},
                    {'label': 'ecoli-01 (1.86)', 'value': 'summary-ecoli-01.db'},
                    {'label': 'yeast-1 (2.46)', 'value': 'summary-yeast-1.db'},
                    {'label': 'yeast-3 (8.11)', 'value':'summary-yeast-3.db'},
                    {'label': 'ecoli-0675 (10)', 'value':'summary-ecoli-0675.db'},
                    {'label': 'led7digit (10.97)', 'value':'summary-led7digit.db'},
                    {'label': 'yeast-21897 (30.56)', 'value':'summary-yeast-21897.db'},
                    {'label': 'yeast-5 (32.78)', 'value':'summary-yeast-5.db'},
                    {'label': 'yeast-6 (39.15)', 'value':'summary-yeast-6.db'},
                    {'label': 'abalone-19 (128.87)', 'value':'summary-abalone-19.db'}],
                value='summary-yeast-21897.db'
            )
        ],
        style={'width': '15%', 'display': 'inline-block','vertical-align': 'middle','padding':'15px'}),
        
        html.Div([
            html.P('Evaluation'),
            dcc.Dropdown(
                id='dash-evaluation',
                options=[
                    {'label': 'Geometric Mean', 'value': 'gm'},
                    {'label': 'AUC', 'value': 'auc'},
                    {'label': 'Minority Recall', 'value': 'recall'},
                    {'label': 'SAR', 'value':'sar'}],
                value='auc'
            )
        ],
        style={'width': '9%', 'display': 'inline-block','vertical-align': 'middle','padding':'5px'}),

        html.Div([
            html.P('Sampling'),
            dcc.Dropdown(
                id='dash-sampling',
                options=[
                    {'label': 'Rus', 'value': 'rus'},
                    {'label': 'No', 'value': 'no'},
                    {'label': 'Smote', 'value': 'smote'}],
                value='no'
            )
        ],
        style={'width': '9%', 'display': 'inline-block','vertical-align': 'middle','padding':'5px'}),
        
        html.Div([
            html.P('Balance'),
            dcc.Dropdown(
                id='dash-balance',
                options=[
                    {'label': 'Yes', 'value': 1},
                    {'label': 'No', 'value': 0}],
                value=0
            )
        ],
        style={'width': '9%', 'display': 'inline-block','vertical-align': 'middle','padding':'5px'}),
        
        html.Div([
            html.P('Weight'),
            dcc.Dropdown(
                id='dash-weight',
                options=[
                    {'label': 'Yes', 'value': 1},
                    {'label': 'No', 'value': 0}],
                value=0
            )
        ],
        style={'width': '9%', 'display': 'inline-block','vertical-align': 'middle','padding':'5px'}),
        
        html.Div([
            html.P('Comparison'),
            dcc.Dropdown(
                id='dash-compare',
                options=[
                    {'label': 'Weight', 'value': 'weight'},
                    {'label': 'Balance', 'value': 'balance'},
                    {'label': 'Sample', 'value': 'sampling'}],
                value='sampling'
            )
        ],
        style={'width': '9%', 'display': 'inline-block','vertical-align': 'middle','padding':'5px'}),
    ], style={
        'borderBottom': 'thin lightgrey solid',
        'backgroundColor': 'rgb(250, 250, 250)',
        'padding': '5px 3px'
    }),
    
    html.Div([
        dcc.Graph(
            id='dash-main-plot')
    ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'}),
    
    html.Div([
            dcc.Graph(id='dash-process-plot')
    ], style={'width': '49%', 'float': 'right', 'display': 'inline-block'}),
    
    html.Div(id='dash-signal', style={'display': 'none'})
])

@app.callback(
    dash.dependencies.Output('dash-main-plot', 'figure'),
    [dash.dependencies.Input('dash-evaluation', 'value'),
     dash.dependencies.Input('dash-sampling', 'value'),
     dash.dependencies.Input('dash-balance', 'value'),
     dash.dependencies.Input('dash-weight', 'value'),
     dash.dependencies.Input('dash-compare', 'value'),
     dash.dependencies.Input('dash-signal', 'children')])
def update_graph(eva, sample, balance, weight, compare, newdf):
    
    df = pd.read_json(newdf)
    if compare == 'sampling':
        lst = ['no','rus','smote']
        dff=df[(df.evaluation==eva) & (df.balance==balance) & (df.weight==weight)]
    if compare == 'balance':
        lst = [0,1]
        dff=df[(df.evaluation==eva) & (df.sampling==sample) & (df.weight==weight)]
    if compare == 'weight':
        lst = [0,1]
        dff=df[(df.evaluation==eva) & (df.sampling==sample) & (df.balance==balance)]
    order = ['direct','kappa','boost','uwa','dwa','udwa','udwa_enf','random']
    return {
        'data': [{
            'x': order,
            'y': dff[dff[compare]==val].groupby('heuristic').mean()['test_avg'].reindex(order),
            'error_y': {'type': 'data', 
                        'array': dff[dff[compare]==val].groupby('heuristic').std()['test_avg'].reindex(order), 
                        'visible':True},
            'name': val,
            'line': {'width': 3}
        }  for val in lst],
        'layout': {
            'margin': {'l': 40,'r': 30,'b': 20,'t': 15},
            'height':600,
            'hovermode':'closest',
            'legend':{'orientation': 'h'}
        }
    }



@app.callback(Output('dash-signal', 'children'),
              [Input('dash-file', 'value')])
def update_df(value):
    # it is not safe for multi user purpose
    newdf =  load_data(value)
    return newdf.to_json()

@app.callback(
    dash.dependencies.Output('dash-process-plot', 'figure'),
    [dash.dependencies.Input('dash-evaluation', 'value'),
     dash.dependencies.Input('dash-sampling', 'value'),
     dash.dependencies.Input('dash-balance', 'value'),
     dash.dependencies.Input('dash-weight', 'value'),
     dash.dependencies.Input('dash-main-plot', 'hoverData'),
     dash.dependencies.Input('dash-compare', 'value'),
     dash.dependencies.Input('dash-file', 'value')])
def update_process(eva, sample, balance, weight, hoverData, compare, file):
    heuristic = hoverData['points'][0]['x']
    pos = hoverData['points'][0]['curveNumber']
    if compare == 'sampling':
        sample = ['no','rus','smote'][pos]
    if compare == 'balance':
        balance = [0,1][pos]
    if compare == 'weight':
        weight = [0,1][pos]
    y, error_y, name = load_process(file, heuristic, balance, weight, sample, eva)
    title = '<b>metric={metric}, heuristic={heuristic}, sample={sample}, balance={balance}, weight={weight}</b>'.format(**name)
    return {
        'data': [{
            'x': list(range(1,31)),
            'y': y[0],
            'error_y': {'type': 'data', 
                        'array': error_y[0],
                        'visible':True},
            'mode': 'lines+markers',
            'name': 'prune score',
            'line': {'width': 3}
        },
        {
            'x': list(range(1,31)),
            'y': y[1],
            'error_y': {'type': 'data', 
                        'array': error_y[1],
                        'visible':True},
            'mode': 'lines+markers',
            'name': 'test score',
            'line': {'width': 3}
        }],
        'layout': {
            'height': 600,
            "titlefont": {"size": 16},
            'margin': {'l': 40,'r': 30,'b': 20,'t': 55},
             'title': title,
            'legend':{'orientation': 'h'}
        }
    }

if __name__ == '__main__':
    app.run_server()