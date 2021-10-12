from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt


def transform_data(df):
    shape_raw = df.shape

    # Split group of tags
    df_tags = pd.DataFrame(columns=df.columns)

    for i in tqdm(range(df.shape[0])):
        df_temp = pd.DataFrame({
            'id': df.id.values[i],
            'date': df.date.values[i],
            'tags': df.tags.values[i].split(' , '),
            'likes': df.likes.values[i],
            'text': df.text.values[i],
            'clean_text': df.clean_text.values[i],
        })
        df_tags = pd.concat([df_tags, df_temp])

    # Delete errors
    df_tags.tags = df_tags.tags.str.replace('[^а-я]', '')
    indx = df_tags.tags.value_counts()[df_tags.tags.value_counts() > 50].index
    df_tags = df_tags[df_tags.tags.isin(indx)]

    # Divide date in parts
    df_tags['days'] = df_tags.date.str.replace('^| .*$', '')
    df_tags['months'] = df_tags.date.str.replace('^\d+ | \d+.*$', '')
    df_tags['years'] = df_tags.date.str.replace('^\d+ [а-я]+ |, .*$', '')
    df_tags['time'] = df_tags.date.str.replace('^.*, |$', '')
    df_tags['hours'] = df_tags.time.apply(lambda x: x.split(':')[0], 1)
    df_tags['minutes'] = df_tags.time.apply(lambda x: x.split(':')[1], 1)

    # Replace RUS months
    df_tags.loc[df_tags.months == 'января', 'months'] = 1
    df_tags.loc[df_tags.months == 'февраля', 'months'] = 2
    df_tags.loc[df_tags.months == 'марта', 'months'] = 3
    df_tags.loc[df_tags.months == 'апреля', 'months'] = 4
    df_tags.loc[df_tags.months == 'мая', 'months'] = 5
    df_tags.loc[df_tags.months == 'июня', 'months'] = 6
    df_tags.loc[df_tags.months == 'июля', 'months'] = 7
    df_tags.loc[df_tags.months == 'августа', 'months'] = 8
    df_tags.loc[df_tags.months == 'сентября', 'months'] = 9
    df_tags.loc[df_tags.months == 'октября', 'months'] = 10
    df_tags.loc[df_tags.months == 'ноября', 'months'] = 11
    df_tags.loc[df_tags.months == 'декабря', 'months'] = 12

    # Union date in correct format
    df_tags['days'] = df_tags['days'].astype('str')
    df_tags['months'] = df_tags['months'].astype('str')
    df_tags['years'] = df_tags['years'].astype('str')
    df_tags.loc[:, 'date'] = df_tags['years'] + '/' + df_tags['months'] + '/' + df_tags['days'] + ' ' + df_tags['time']

    # Convert in datetime format
    df_tags.date = pd.to_datetime(df_tags.date)
    df_tags['day_week'] = df_tags.date.dt.day_name()

    shape_new = df_tags.shape
    print('Old: {} New: {}'.format(shape_raw, shape_new))

    df['likes'] = df['likes'].astype('int')

    return df_tags


def plot_bar(title, xlabel, x, y):
    plt.figure(figsize=(15,9))
    plt.title(title)
    plt.barh(x, y)
    plt.xlabel(xlabel)
    plt.show()