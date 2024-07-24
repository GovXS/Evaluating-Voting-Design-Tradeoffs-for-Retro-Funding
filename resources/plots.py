import pandas as pd
import matplotlib.pyplot as plt

def plot_distribution(project_df):    
    fig, ax = plt.subplots(figsize=(15,5))
    project_df['token_amount'].plot(kind='bar', width=1, ax=ax)
    ax.set_xticks([])
    ax.set_xlabel("")
    ax.set_ylabel("Tokens")
    fig.show()
    
    
def plot_alignment(project_df):    
    fig, ax = plt.subplots(figsize=(15,5))
    project_df.plot(kind='scatter', x='rating', y='token_amount', ax=ax)
    ax.set_ylabel("Tokens")
    ax.set_xlabel("Impact")
    ax.set_xticks([])    
    fig.show()
    
    
def analyze_simulation(results):
    print(pd.Series(results).iloc[:-1].apply(lambda x: int(x) if isinstance(x, float) else x))
    data = results['data']
    project_df = pd.DataFrame(data).sort_values(by='token_amount', ascending=False)
    plot_distribution(project_df)    
    plot_alignment(project_df)