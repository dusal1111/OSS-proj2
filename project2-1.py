#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# 주어진 데이터를 DataFrame으로 읽어옴
df = pd.read_csv('2019_kbo_for_kaggle_v2.csv')

# 1. Print the top 10 players in H, avg, HR, and OBP for each year from 2015 to 2018.
for year in range(2015, 2019):
    print(f"\nTop 10 players in {year} for each category:")
    for category in ['H', 'avg', 'HR', 'OBP']:
        top_players = df[df['year'] == year].nlargest(10, category)[['batter_name']]
        print(f"\nTop 10 in {category} for {year}:")
        print(top_players['batter_name'].values)


# 2. Print the player with the highest war by position(CP) in 2018.
year = 2018
position_war_max = df[(df['year'] == year) & (df['cp'] != '지명타자')].groupby('cp')['war'].idxmax()
best_players_by_position = df.loc[position_war_max][['cp', 'batter_name', 'war']]

print(f"\nPlayer with the highest war by position(CP) in {year} (excluding DH):")
print(best_players_by_position[['cp', 'batter_name']].values)


# 3. Among R, H, HR, RBI, SB, war, avg, OBP, and SLG, which has the highest correlation with salary?
correlations = df[['R', 'H', 'HR', 'RBI', 'SB', 'war', 'avg', 'OBP', 'SLG', 'salary']].corr()['salary']
highest_correlation_category = correlations.abs().nlargest(2).index[1]

print(f"\n\nCategory with the highest correlation with salary:")
print(highest_correlation_category)


# In[ ]:




