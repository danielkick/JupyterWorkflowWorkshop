# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import os
# os.system('pip install palmerpenguins')
# os.system('pip install plotly')

# Data
from palmerpenguins import load_penguins 

# Data Wrangling
import numpy as np
import pandas as pd

# Data Modeling
import statsmodels.api as sm          
import statsmodels.formula.api as smf # tilde formulas (y ~ x1 + x2)

# Graphing
import plotly.express as px         # Main interface
import plotly.figure_factory as ff  # Specialized plots (dendrograms, density plots)
import plotly.graph_objects as go   # Lower level interface
# -

penguins = load_penguins()
penguins

# # Standard Plots

# +
# Scatterplot
# > #

# +
# Distribution Plot
penguin_sp = ['Adelie', 'Gentoo', 'Chinstrap']
hist_data = [penguins.loc[((penguins['species'] == sp) & (penguins['flipper_length_mm'].notna()) ), 'flipper_length_mm'] for sp in penguin_sp]

group_labels = penguin_sp

# > #
# > #
# -
# Scatterplot 2
# > # fig = 
fig 

# # Customizing a Plot

# +
# Recreate Scatterplot 2

color_mapping = {
    'Adelie':'#636EFA', 
    'Gentoo':'#EF553B', 
    'Chinstrap':'#00CC96'}

color_list = [color_mapping[i] for i in penguins['species']]

# > #
# > #
# > #
# > #
# > #
# > #
# > #
# > #
# > #
# > #

fig.update_layout(scene = dict(
                    xaxis_title='bill_length_mm',
                    yaxis_title='bill_depth_mm',
                    zaxis_title='flipper_length_mm'))
fig.show()



# +
# Model fitting:
penguins_nona = penguins[['species', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm']].dropna()
fm = smf.ols(formula='flipper_length_mm ~ species + bill_length_mm + bill_depth_mm', data=penguins_nona)

res = fm.fit()

print(res.summary())
# -

# Model parameter
res.params


# +
# Transparant predictions 
# Note: relying on a global variable like this is bad practice.
def predict_flipper_len(species, bill_length_mm, bill_depth_mm):
    yHat = res.params['Intercept']

    if species == 'Chinstrap':
        yHat += res.params['species[T.Chinstrap]']
    elif species == 'Gentoo':
        yHat += res.params['species[T.Gentoo]']

    yHat += res.params['bill_length_mm'] * bill_length_mm
    yHat += res.params['bill_depth_mm'] * bill_depth_mm

    return(yHat)


def mk_plane(species = 'Adelie'):

    # get corners of the plane to define
    min_length = penguins_nona.loc[penguins_nona['species'] == species, 'bill_length_mm'].min()
    max_length = penguins_nona.loc[penguins_nona['species'] == species, 'bill_length_mm'].max()

    min_depth  = penguins_nona.loc[penguins_nona['species'] == species, 'bill_depth_mm'].min()
    max_depth  = penguins_nona.loc[penguins_nona['species'] == species, 'bill_depth_mm'].max()


    temp = pd.DataFrame({'bill_length_mm' : [min_length, min_length, max_length, max_length],
                         'bill_depth_mm'  : [min_depth, max_depth, min_depth, max_depth]})

    temp['species'] = species
    temp['flipper_length_mm'] = np.nan

    # calculate the z for each
    for i in temp.index:
        temp.loc[i, 'flipper_length_mm'] = predict_flipper_len(species = temp.loc[i, 'species'], 
                                                               bill_length_mm = temp.loc[i, 'bill_length_mm'], 
                                                               bill_depth_mm = temp.loc[i, 'bill_depth_mm'])
    return(temp)

# +
for sp in ['Adelie', 'Gentoo', 'Chinstrap']:

    prediction_plane = mk_plane(species = sp)

    color_list = [color_mapping[i] for i in prediction_plane['species']]

# > #
# > #
# > #
# > #
# > #

fig
