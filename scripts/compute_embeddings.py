import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from Latte import Latte


def compute():
    for project in ['biology', 'chemistry', 'computergraphics', 'ell', 'english', 'gamedev', 'graphicdesign', 'linguistics', 'photo', 'physics', 'softwareengineering']:
        print(f'Computing embeddings for {project}')
        df = pd.read_csv(f'data/{project}_sample.csv')
        latte = Latte(df)
        latte.embed(save_to_file=f'embeddings/{project}_minilm.pkl')
        #latte.embed('hf', model_name='all-mpnet-base-v2', save_to_file=f'embeddings/{project}_mpnet.pkl')
        #latte.embed('openai', save_to_file=f'embeddings/{project}_openai.pkl')
        #latte.embed('gemini', save_to_file=f'embeddings/{project}_gemini.pkl')

if __name__ == "__main__":
    # Run the sampling function
    compute() 