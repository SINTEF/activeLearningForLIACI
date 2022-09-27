# from distutils.log import debug
from dash import Dash, html, dcc, get_asset_url
import plotly.express as px
import pandas as pd
from PIL import Image

img_dir = 'assets/image_0002.jpg'

app = Dash(__name__)

# app.css.append_css({'external_url': 'reset.css'})
# app.server.static_folder = ''

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

df = pd.DataFrame({
    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
    "Amount": [4,1,2,2,4,5],
    "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
})

fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")
fig.update_layout(
    plot_bgcolor=colors['background'],
    paper_bgcolor=colors['background'],
    font_color=colors['text'],)

app.layout = html.Div(children=[
    html.H1(children="hello Dash",
        style={
            'textAlign': 'center',
            'color': colors['text']
            }
    ),
    html.Div(children=''' 
        Dash: Al web app frema
    ''',
        style={
            'textAlign': 'center',
            'color': colors['text']
            }),
    dcc.Graph(
        id='example-graph',
        figure=fig
    ),
    
    # html.Img(src=img_dir),
    html.Img(src=get_asset_url('images/image_0008.jpg')),
    ],
    # style={'color':colors['background']}
)



if __name__ == '__main__':
    app.run_server(debug=True)