# import libraries
import matplotlib.pylab as plt
import pandas as pd
import numpy as np

np.random.seed(3123) # set random seed for reproducibility

df1 = pd.read_csv('./climate_change.csv',header=2)
df2 = pd.read_csv('./infrastructure.csv',header=2)

for idc in df2["Indicator Code"].unique():
    df1.drop(df1[df1["Indicator Code"].eq(idc)].index, inplace=True)

df = pd.DataFrame(np.concatenate((df1,df2),axis=0),columns=df1.columns)
df = df.drop(df.columns[-1],axis=1) # drop unnamed column with nans

df_adj = pd.DataFrame(None,columns= df["Indicator Code"].unique())
df_adj.insert(0,"Country Code",None)
df_adj.insert(0,"Country Name",None)
df_adj.insert(0,"Year",None)

for c in df["Country Code"].unique():

    df_c = df[df["Country Code"].eq(c)]
    df_c = df_c.drop(["Country Name","Country Code","Indicator Name"],axis=1)
    df_c = pd.DataFrame(np.array(df_c.iloc[:,1:]).transpose(),columns=df_c["Indicator Code"],index=df_c.columns[1:])
    df_c.insert(0,"Country Code",np.repeat(c,df_c.shape[0],axis=0))
    df_c.insert(0,"Country Name",np.repeat(df[df["Country Code"] == c]["Country Name"].iloc[0],df_c.shape[0],axis=0))
    df_c.insert(0,"Year",df_c.index)


    df_adj = pd.DataFrame(np.concatenate((df_adj,df_c),axis=0),columns=df_adj.columns)
    del df_c

df_adj.rename(index=df_adj["Year"],inplace=True)
df_adj = df_adj.drop("Year",axis=1)

# create by country data frame for visulisation
df_emiss = df_adj[["Country Code", "Country Name", "EN.ATM.GHGT.KT.CE"]]
# create world data frame for analysis
df_world = df_adj.loc[df_adj["Country Code"].eq("WLD")]
df_world = df_world.drop(["Country Code", "Country Name"],axis=1)

# drop features which are a function of the greenhouse gas emissions indicator (e.g. measures of specific greenhouse gases and transformations)
df_world = df_world.drop(["EN.ATM.SF6G.KT.CE", "EN.ATM.PFCG.KT.CE", "EN.ATM.NOXE.ZG", "EN.ATM.NOXE.KT.CE", "EN.ATM.METH.ZG", "EN.ATM.METH.KT.CE", "EN.ATM.HFCG.KT.CE", "EN.ATM.GHGT.ZG", "EN.ATM.GHGO.ZG", "EN.ATM.GHGO.KT.CE", "EN.ATM.CO2E.SF.ZS", "EN.ATM.CO2E.SF.KT", "EN.ATM.CO2E.PP.GD.KD", "EN.ATM.CO2E.PP.GD", "EN.ATM.CO2E.PC", "EN.ATM.CO2E.LF.ZS", "EN.ATM.CO2E.LF.KT", "EN.ATM.CO2E.KT", "EN.ATM.CO2E.KD.GD", "EN.ATM.CO2E.GF.ZS", "EN.ATM.CO2E.GF.KT", "EN.ATM.CO2E.EG.ZS"],axis=1)

# deal with nans (for analysis and plotting separately)
nan_idx = df_world.index[df_world["EN.ATM.GHGT.KT.CE"].isna()]
df_world = df_world.drop(nan_idx,axis=0)
df_emiss = df_emiss.drop(nan_idx,axis=0) # also apply to country df

# drop countries with missing values (for plotting)
def check_na(x):
    return x.isna().any()

miss_c = df_emiss.groupby(by="Country Code").agg(check_na)


incl_idx=np.array(df_emiss["Country Code"].map(lambda x: x not in miss_c.index[miss_c["EN.ATM.GHGT.KT.CE"]]))

df_emiss = df_emiss.loc[incl_idx]
del incl_idx, miss_c

# add column to express change as % from 1990
df_emiss_change = df_emiss.groupby(by="Country Code")["EN.ATM.GHGT.KT.CE"].apply(lambda x: (x / x.iloc[0])*100)
df_emiss_change.index = df_emiss.index

df_emiss["EN.ATM.GHGT.KT.CE_perc"] = df_emiss_change

# change dtype to numeric
df_emiss["EN.ATM.GHGT.KT.CE"] = pd.to_numeric(df_emiss["EN.ATM.GHGT.KT.CE"])
df_emiss["EN.ATM.GHGT.KT.CE_perc"] = pd.to_numeric(df_emiss["EN.ATM.GHGT.KT.CE_perc"])

df_world.dropna(axis=1,thresh=np.round(df_world.shape[0]*0.75),inplace=True)

# Visualization
from dash import Dash, html, dash_table, dcc, Input, Output, callback
import plotly.express as px

#---define some variables helpful for plotting---

# year index
year = pd.to_numeric(df_emiss.index.unique().tolist())

# country code mapping
country_list = df_emiss["Country Name"].copy()
country_list.index=df_emiss["Country Code"]
country_list.drop_duplicates(keep='first',inplace=True)
country_list = country_list.to_dict()

# indicator key mapping
indicator_dict = df[df["Country Code"] == 'WLD'][["Indicator Code", "Indicator Name"]].copy()
indicator_dict.index=df[df["Country Code"] == 'WLD']["Indicator Code"]
indicator_dict = indicator_dict.loc[df_world.columns]
indicator_dict = indicator_dict.to_dict('records')

#---Configure the app---
app = Dash()
server = app.server

# create choropleth map object
fig_worldMap = px.choropleth(df_emiss, locations='Country Code', color="EN.ATM.GHGT.KT.CE_perc", hover_name='Country Name', range_color=[min(df_emiss["EN.ATM.GHGT.KT.CE_perc"]), np.mean(df_emiss["EN.ATM.GHGT.KT.CE_perc"])+np.std(df_emiss["EN.ATM.GHGT.KT.CE_perc"])],
                    projection='natural earth', animation_frame=df_emiss.index, labels = {'EN.ATM.GHGT.KT.CE_perc': 'GHG%', 'index': 'year'})

app.layout = html.Div(style={'font-family': 'arial', 'max-width': '95%', 'margin': 'auto'}, children=[
    
    # title and info text
    html.H1(children="🌎 Climate Change", style={'textAlign': 'center'}),
    dcc.Markdown('''
                This dashboard allows you to explore the development of greenhouse gas (GHG) emissions of different countries and indicators used to forecast changes in worldwide GHG emissions in [this project](https://github.com/AlexSkowron/climate_change_forecasting).
                '''),
    html.Br(),

    # greenhouse gas emissions by country plots
    html.Div([
        html.Div([
            "World-wide greenhouse gas emissions as % change from 1990 (GHG%)",
            dcc.Graph(figure=fig_worldMap)
        ], style={'width': '60%', 'display': 'inline-block', 'vertical-align': 'top'}),
        html.Div([
            "Select countries from the drop-down menu:",
            dcc.Dropdown(id = 'country_select', options=country_list, value=['WLD'], multi=True),
            dcc.Graph(id = 'countryGHG_fig'),
            html.Br()
        ], style={'width': '35%', 'display': 'inline-block', 'padding-left': '4%'})
    ]),

    # indicator plots
    html.Div([
        html.Div([
            "Select forecast indicator from the drop-down menu:",
            dcc.Dropdown(id = 'indicator_select', options=df_world.columns, value='EN.ATM.GHGT.KT.CE'),
            dcc.Graph(id = 'indicator_fig')
        ], style={'width': '50%', 'display': 'inline-block', 'vertical-align': 'top'}),
        html.Div([
            dash_table.DataTable(data=indicator_dict, page_size=14, style_cell={'textAlign': 'left', 'font-family':'arial', 'font-size': 12}),
        ], style={'width': '40%', 'display': 'inline-block', 'padding-left': '4%'})
    ], style={'padding-top': 60}),
])
@callback(
    Output('countryGHG_fig', 'figure'),
    Input('country_select', 'value')
)
def update_figure(selected_country):

    df_plot = pd.DataFrame(None,columns=selected_country)

    for c in selected_country:
        df_plot[c] = df_emiss[df_emiss["Country Code"] == c]['EN.ATM.GHGT.KT.CE']

    fig_countryGHG = px.line(df_plot, x=year, y=selected_country, labels={'x': 'year', 'value': 'Total greenhouse gas emissions<br>(kt of CO2 equivalent)', 'variable': 'country'})

    return fig_countryGHG
@callback(
    Output('indicator_fig', 'figure'),
    Input('indicator_select', 'value')
)
def update_figure(selected_indicator):

    fig_indicator = px.line(df_world, x=pd.to_numeric(df_world.index.to_list()), y=selected_indicator, labels={'x': 'year'})

    return fig_indicator


## Run the app
#if __name__ == '__main__':
#    app.run(jupyter_mode="tab",port=8060)