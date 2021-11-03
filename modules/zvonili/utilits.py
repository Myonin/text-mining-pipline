import sys
sys.path.append('..')
from tqdm import tqdm
import pandas as pd


def transform_data(df):
    shape_raw = df.shape

    # Extract names of companies and regions
    df['company'] = df['company_region'].str.extract('([А-Я]+ \".*\")')
    df['region'] = df['company_region'].str.extract('регионе (.*)\n')

    # Fast cleaning of comments
    df['comments'] = df['comments'].str.replace('\s+', ' ')
    df['comments'] = df['comments'].str.replace(' +', ' ')
    df['comments'] = df['comments'].str.replace('^ ', '')
    df['comments'] = df['comments'].str.replace(' $', '')

    # Convert to datetime
    df['date'] = pd.to_datetime(df['date'], format=' %d.%m.%Y %H:%M')

    # Convert to int
    df['views'] = df['views'].astype('int')

    # Convert to str
    df['tags'] = df['tags'].astype('str')

    df = df.drop('company_region', 1)

    # Split group of tags
    df_clean = pd.DataFrame(columns=df.columns)

    for i in tqdm(range(df.shape[0])):
        df_temp = pd.DataFrame({
            'number': df['number'].values[i],
            'views': df['views'].values[i],
            'tags': df['tags'].values[i].split(', '),
            'comments': df['comments'].values[i],
            'date': df['date'].values[i],
            'company': df['company'].values[i],
            'region': df['region'].values[i],
        })
        df_clean = pd.concat([df_clean, df_temp])

    shape_new = df_clean.shape
    print('Old: {} New: {}'.format(shape_raw, shape_new))

    return df_clean