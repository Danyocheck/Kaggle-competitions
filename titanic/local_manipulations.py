import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import plotly.express as px


def plot_scores(model, X_tr, y_tr, X_val, y_val, params):
    y_pred_tr_expit = model.predict(X_tr)
    y_pred_tr_raw = model.predict(X_tr, raw_score=True)
    y_pred_val_expit = model.predict(X_val)
    y_pred_val_raw = model.predict(X_val, raw_score=True)

    mosaic = [
        ['tr_expit', 'tr_raw'],
        ['val_expit', 'val_raw'],
        ['isalone=0. raw', 'isalone=1. raw']
    ]
        
    fig, ax = plt.subplot_mosaic(mosaic, figsize=(15, 10))
    for key in ax:
        ax[key].set_title(key, fontsize=15)

    sns.histplot(x=y_pred_tr_expit, hue=y_tr, bins=10, ax=ax['tr_expit'])
    sns.histplot(x=y_pred_tr_raw, hue=y_tr, bins=10, ax=ax['tr_raw'])
    sns.histplot(x=y_pred_val_expit, hue=y_val, bins=10, ax=ax['val_expit'])
    sns.histplot(x=y_pred_val_raw, hue=y_val, bins=10, ax=ax['val_raw'])

    for value in X_val.isalone.unique():
        cond = X_val.isalone == value
        sns.histplot(x=y_pred_val_raw[cond], hue=y_val[cond], bins=10, ax=ax[f'isalone={value}. raw'])
    
    fig.tight_layout()
    plt.show()
    
    
def plot_tree_info(t, ntrees=None, figsize=(15, 25)):
    sns.set()
    mpl.rcParams['legend.title_fontsize'] = 15
    mpl.rcParams['legend.fontsize'] = 15
    
    fig, ax = plt.subplots(figsize=figsize)

    if ntrees is None:
        ntrees = t.tree_index.max() + 1
    sns.stripplot(t.query('node_depth < 4 and ~split_feature.isnull() and tree_index < @ntrees'),
                  y='split_feature', x='tree_index', hue='node_depth', jitter=False, dodge=True,
                  size=11, edgecolor='black', linewidth=0.5)

    ax.tick_params('both', labelsize=15)
    ax.grid(True)
    ax.legend_.set_bbox_to_anchor((-0.15, 1.))

    ax.set_ylabel('')
    ax.set_xlabel('номер дерева', fontsize=15)
    ax.xaxis.set_label_position('top') 
    ax.xaxis.tick_top()
    
    plt.show()


def plot_feature_info(t):
    t_prep = (
        t.query('~split_feature.isnull()')
        .groupby(['split_feature', 'node_depth'])
        .size()
        .to_frame('cnt')
        .reset_index()
    )
    t_prep.cnt = t_prep.cnt / t_prep.node_depth + 1
    fig = px.scatter(t_prep, x='split_feature', y='node_depth', size='cnt')
    fig.show()