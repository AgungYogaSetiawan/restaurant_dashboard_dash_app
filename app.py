# Import Library & Prepare Data

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from dateutil import parser
from datetime import date



# Read dataset
resto_df = pd.read_excel('Q3_competition_detail_dataset.xlsx')


# Cleaning data
resto_df['latitude'] = resto_df['latitude'].astype('float')
resto_df['longitude'] = resto_df['longitude'].astype('float')
resto_df = resto_df.reset_index()
resto_df = resto_df.drop(['index','cross_streets'],axis=1)
resto_df['Price'] = resto_df['price'].apply(lambda x: len(str(x)))



# Cleaning data column city

resto_df['City'] = resto_df.city.apply(lambda x: x.strip().lower())

# Column name changed
resto_df['City'] = resto_df.City.apply(lambda x: x[:-2] if x[-2:]=='ca' else x)
resto_df['City'] = resto_df.City.apply(lambda x: ' '.join(x.split()))
resto_df['City'] = resto_df.City.replace('lost angeles', 'los angeles')
resto_df['City'] = resto_df.City.replace('longbeach', 'long beach')
resto_df['City'] = resto_df.City.replace('rowland hghts', 'rowland heights')
resto_df['City'] = resto_df.City.replace('rowland heightes', 'rowland heights')
resto_df['City'] = resto_df.City.replace('santa fe spring', 'santa fe springs')
resto_df['City'] = resto_df.City.replace('shermanoaks', 'sherman oaks')
resto_df['City'] = resto_df.City.replace('canyon cntry', 'canyon country')
resto_df['City'] = resto_df.City.replace('studiocity', 'studio city')
resto_df['City'] = resto_df.City.replace('santa moni', 'santa monica')

# Set up capital letters for first letter

resto_df['City'] = resto_df.City.apply(lambda x: str(x)[0].upper()+ str(x)[1:])
resto_df['categories01'] = resto_df.categories01.apply(lambda x: str(x)[0].upper()+ str(x)[1:])
resto_df['categories02'] = resto_df.categories02.apply(lambda x: str(x)[0].upper()+ str(x)[1:])
resto_df['categories03'] = resto_df.categories03.apply(lambda x: str(x)[0].upper()+ str(x)[1:])


# Merge with data reviews

resto_reviews_df = pd.read_excel('Q3_competition_review_dataset.xlsx')
df = resto_reviews_df.merge(resto_df, left_on='id', right_on='id')


# review_time_created change to datetime

df['review_time_created'] = pd.to_datetime(df['review_time_created'])



import re
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


data = df.review_text.values.tolist()
eng = stopwords.words('english')
data = [word for word in data if not word in eng]
df['reviews_text_clean'] = data



# Make column categorizing the key words into specific values cared by customers
df['value_review'] = df['review_rating']
df['value_review'].replace({
    0.0: 'Food', 
    1.0: 'Environment',
    2.0: 'Experience',
    3.0: 'Service',
    4.0: 'Waiting time',
    5.0: 'Others'
}, inplace=True)



# make city & category list
city_list = df.groupby('city').count().sort_values(by='id', ascending=False).index.to_list()
# city_list[:11]

categories1 = df.groupby('categories01').count().sort_values(by='id', ascending=False).index.to_list()
categories2 = df.groupby('categories02').count().sort_values(by='id', ascending=False).index.to_list()
categories3 = df.groupby('categories03').count().sort_values(by='id', ascending=False).index.to_list()


# Data Covid-19


df_covid = pd.read_csv('latimes-place-totals.csv')
df_covid = df_covid.drop('note', axis=1)
df_covid = df_covid[df_covid.county == 'Los Angeles']
df_covid['date'] = pd.to_datetime(df_covid['date'])
df_covid = df_covid[df_covid['date'] > '2021-01-01']
df_covid.rename(columns={'name':'city'}, inplace=True)
df_covid['city'] = df_covid['city'].apply(lambda x: x.split(':')[-1].strip())


covid_2021 = df_covid[['city','date','fips','confirmed_cases','population']].copy()
covid_2021 = covid_2021.sort_values(by=['city','date']).reset_index().drop('index', axis=1)


new_cases = pd.DataFrame(None)

for city in covid_2021.city.unique():
    covid_data = covid_2021[covid_2021['city'] == city]
    covid_data = covid_data.sort_values(by='date')
    covid_data['next_day_cases'] = covid_data['confirmed_cases'].shift(-1)
    new_cases = pd.concat([covid_data, new_cases])

new_cases['new_daily_cases'] = new_cases['next_day_cases'] - new_cases['confirmed_cases']
new_cases = new_cases[new_cases['new_daily_cases']>=0]
data_covid = new_cases.sort_values(by=['city','date']).reset_index().drop('index',axis=1)


# save dataframe 
# main_data = df.to_excel('main_data.xlsx', index=False)
# covid_df = data_covid.to_excel('data_covid.xlsx', index=False)
# main_data = df.to_csv('main_data.csv', index=False)
# covid_df = data_covid.to_csv('data_covid.csv', index=False)


# Build Dashboard using Dash Plotly

# Import library
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input
import plotly.express as px
import plotly.graph_objects as go


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
# server = app.server

# layout
app.layout = html.Div([
    html.H2('Analytical Dashboard To Improve Restaurant Performance', style={'textAlign':'center'}),
    html.Hr(),
    html.P('Choose Food Category:'),
    html.Div(html.Div([
        dcc.Dropdown(id='food_list', clearable=False,
                     options=[{'label':x, 'value':x} for x in df['categories01'].unique()],
                     value='Pizza'),
    ], className='twelve columns'), className='row'),
    
    html.Div(id='output_graph', children=[]),
    
    
    html.Div([html.Div([
        html.P('Choose City:'),
        dcc.Dropdown(id='city_list', clearable=False,
                     options=[{'label':x, 'value':x} for x in df['city'].unique()],
                     value='Los Angeles'),
    ], className='six columns'),
            html.Div([
        html.P('Choose One or Many Cities:'),
        dcc.Dropdown(id='cov_city_list', multi=True,
                     options=[{'label':x, 'value':x} for x in df['city'].unique()],
                     value=['Hollywood','Beverly Hills']),
    ], className='six columns')], className='row'),
    
    html.Div([
        html.Div([
            dcc.Graph(id='out_graphs2', figure={})
        ], className='six columns'),
        
        html.Div([
            dcc.Graph(id='out_line_chart', figure={})
        ], className='six columns'),
    ]),
    
    
], style={'textAlign':'center'})


# callback
@app.callback(
    Output(component_id='output_graph', component_property='children'),
    [Input(component_id='food_list', component_property='value')]
)

def make_graph(food_choosen):
    # Make dash datatable cheap price
    df_table = df[df['categories01'] == food_choosen]
    df_table = df_table.sort_values(by=['Price'], ascending=True)
    df_table = df_table.groupby(['name','Price','City','review_rating']).size().reset_index(name='count').drop('count', axis=1).sort_values(by='Price')[:10]
    df_table.rename(columns={'name':'Restaurants'}, inplace=True)
    fig_table_cheap = go.Figure(data=[
        go.Table(
            header=dict(values=list(df_table.columns), fill_color='skyblue', align='left'),
            cells=dict(values=[df_table.Restaurants, df_table.Price, df_table.City, df_table.review_rating], fill_color='lavender', align='left')
        )
    ], layout=go.Layout(title=go.layout.Title(text=f'10 Cheapest {food_choosen} Restaurants')))
    
    # Make dash datatable cheap price
    df_table2 = df[df['categories01'] == food_choosen]
    df_table2 = df_table2.sort_values(by=['Price'], ascending=False)
    df_table2 = df_table2.groupby(['name','Price','City','review_rating']).size().reset_index(name='count').drop('count', axis=1).sort_values(by='Price', ascending=False)[:10]
    df_table2.rename(columns={'name':'Restaurants'}, inplace=True)
    fig_table_expensive = go.Figure(data=[
        go.Table(
            header=dict(values=list(df_table2.columns), fill_color='paleturquoise', align='left'),
            cells=dict(values=[df_table2.Restaurants, df_table2.Price, df_table.City, df_table2.review_rating], fill_color='lavender', align='left')
        )
    ], layout=go.Layout(title=go.layout.Title(text=f'10 Expensive {food_choosen} Restaurants')))
    
    # Make scatter mapbox chart
    df_map = df.copy()
    map_df = df_map[df_map['categories01'] == food_choosen]
    map_df.rename(columns={'review_rating':'Rating'}, inplace=True)
    fig_map = px.scatter_mapbox(map_df, lat='latitude', lon='longitude', color='Rating', size='Price', hover_name='name',
                                title=f'where is the location of the restaurant that serves {food_choosen} <br> based on the map?', 
                                color_continuous_scale=px.colors.sequential.Viridis, mapbox_style="carto-positron", height=500, size_max=6, zoom=8)
    
    
    # Make bar chart average review rating consumer
    df_review = df.copy()
    reviews = df_review[df_review['categories01'] == food_choosen]
    bar_fig = px.bar(x=reviews.groupby('price')['review_count'].mean().index,
                     y=reviews.groupby('price')['review_count'].mean(),
                     title='How many Average Reviews? <br> #Reviews by $ Level')
    bar_fig.update_layout(xaxis_title="Price Level", yaxis_title="Average Reviews")
    
    # Make pie chart
    df_pie = df.copy()
    pie_df = df_pie[df_pie['categories01'] == food_choosen]
    pie_df.rename(columns={'name':'count'}, inplace=True)
    pie_df['transactions'] = pie_df['transactions'].replace(['[]'], ["['pickup', 'delivery']"])
    pie_fig = px.pie(
        data_frame=pie_df,
        values=pie_df.groupby('transactions')['count'].count().sort_values(ascending=False),
        title='what restaurant provides the most take out food',
        names=pie_df.groupby('transactions').count().sort_values(by='count', ascending=False).index,
        color_discrete_sequence=px.colors.qualitative.G10,
        hole=.3,
        width=650, 
        height=400
    )
    

    return [
        html.Div([
            html.Div([dcc.Graph(figure=fig_table_cheap)], className='six columns'),
            html.Div([dcc.Graph(figure=fig_table_expensive)], className='six columns')
        ], className='row'),
        html.Div([
            html.Div([dcc.Graph(figure=fig_map)], className='twelve columns') 
        ], className='row'),
        html.Div([
            html.Div([dcc.Graph(figure=bar_fig)], className='six columns'),
            html.Div([dcc.Graph(figure=pie_fig)], className='six columns')
        ], className='row'),
    ]



# callback 2 for output_graphs2
@app.callback(
    Output(component_id='out_graphs2', component_property='figure'),
    [Input(component_id='city_list', component_property='value')]
)

def make_graphs2(drop_city):
    # Make bar chart for the most high rating restaurant
    df_table_city = df[df['city'] == drop_city]
    df_table_city = df_table_city.groupby(['name', 'categories01', 'Price','City','review_rating']).size().reset_index(name='count').drop('count', axis=1).sort_values(by='review_rating', ascending=False)[:10]
    df_table_city.rename(columns={'name':'Restaurants', 'categories01':'Food'}, inplace=True)
    fig_table_city = go.Figure(data=[
        go.Table(
            header=dict(values=list(df_table_city.columns), fill_color='skyblue', align='left'),
            cells=dict(values=[df_table_city.Restaurants, df_table_city.Food, df_table_city.Price, df_table_city.City, df_table_city.review_rating], fill_color='lavender', align='left')
        )
    ], layout=go.Layout(title=go.layout.Title(text=f'The Most High Rating Review Restaurants in {drop_city}')))
    
    return fig_table_city



# Callback for line chart
@app.callback(
    Output(component_id='out_line_chart', component_property='figure'),
    [Input(component_id='cov_city_list', component_property='value')]
)

def fig_line(multi_city):
    # Make line chart for cases in city about covid-19
#     df_covid = pd.read_csv('data_covid.csv')
    df_covid = data_covid[data_covid['city'].isin(multi_city)]
    df_covid.rename(columns={'date':'Date', 'new_daily_cases':'New Confirmed Cases', 'city':'City'}, inplace=True)
    fig_line_cov = px.line(df_covid, x='Date', y='New Confirmed Cases', color='City', title='Confirmed Cases by City',
                           color_discrete_sequence=px.colors.qualitative.G10)
    fig_line_cov.update_xaxes(side="bottom")
    
    return fig_line_cov



# Run app
if __name__ == "__main__":
    app.run_server(debug=False)