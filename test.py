import pandas as pd


main_data = pd.read_csv('538 Ratings.csv')
# get data that 
df = pd.read_csv('538 Ratings.csv', usecols = ['TEAM', 'SEED', 'ROUND', 'POWER RATING', 'POWER RATING RANK'])
# getting the indexes of all data with 2023 in the year:
indexNames = main_data[main_data['YEAR'] == 2023].index
print(df.loc[indexNames])
data1 = []
for column in df.columns:
    pass

# I want to use machine learning to guess the winner of marchmadness every year, so I need to use the ratings from the previous 3 years as input data
# I also need to use the winner of marchmadness as the target data (which I can make externally)
# I will use the ratings from 2016, 2017, and 2018 to predict the winner of marchmadness in 2019