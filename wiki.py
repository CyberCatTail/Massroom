import wikipediaapi
import pandas as pd
import time

data = pd.read_csv('./wiki_title.csv')
data['detail'] = None

wiki_wiki = wikipediaapi.Wikipedia(user_agent='Mushroom (merlin@example.com)', language='en')

for index, row in data.iterrows():
    title = row['title']
    page = wiki_wiki.page(title)
    
    if page.exists():
        data.at[index, 'detail'] = page.text
    
    time.sleep(0.1)
    data.to_csv('./wiki.csv', index=False)
    print(f'finish row {index}/{data.shape[0]}')

print(data.head())