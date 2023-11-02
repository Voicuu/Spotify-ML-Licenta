from imports import *

def visualize_data(df):
    classified = df.copy()
    classified['pop_rating'] = ''

    for i, row in classified.iterrows():
        score = 'unpopular'
        if (row.popularity > 50) & (row.popularity < 75):
            score = 'medium'
        elif row.popularity >= 75:
            score = 'popular'
        classified.at[i, 'pop_rating'] = score

    fig, ax = plt.subplots(1,1, figsize=(8,5))
    _ = sns.countplot(x='pop_rating', data=classified)
    _ = plt.xlabel('Ratings', fontsize=14)
    _ = plt.title('Counts', fontsize=14)

    numeric_columns = df.columns[df.dtypes != 'object']
    numeric_df = pd.DataFrame(data=df, columns=numeric_columns, index=df.index)

    corr = np.abs(numeric_df.corr())
    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = sns.color_palette("Greens")
    sns.heatmap(corr, cmap=cmap, square=True)
    plt.title('Correlation between numerical features: abs values')
    plt.show()

    corr = numeric_df.corr()[['popularity']].sort_values(by='popularity', ascending=False)
    plt.figure(figsize=(8, 12))
    heatmap = sns.heatmap(corr, annot=True, cmap='Greens')
    heatmap.set_title('The most linear correlated features to POPULARITY', fontdict={'fontsize':18}, pad=16);

    fig, ax = plt.subplots(figsize=(15, 4))
    ax = df.groupby('year')['popularity'].mean().plot(color='green')
    ax.set_title('Mean Popularity over the years', c='green', weight='bold')
    ax.set_ylabel('Mean Popularity', weight='bold')
    ax.set_xlabel('Year', weight='bold')
    ax.set_xticks(range(1920, 2021, 5))
    plt.show()

    def regress_plot(x='', y='', data=None, xlab='', ylab='', titl=''):
        '''Plots a scatterplot with a regression line using given inputs'''
        data = data.groupby(x)[y].mean().to_frame().reset_index()
        fig, ax = plt.subplots(figsize=(10,6))
        _ = sns.regplot(x=x, y=y, data=data, scatter_kws={'color': 'g', "s": 10}, line_kws={'color':'black'} )
        _ = plt.xlabel(xlab, fontsize=12)
        _ = plt.ylabel(ylab, fontsize=12)
        _ = plt.title(titl, fontsize=14, c='green')
        _ = plt.ylim(-3, 103)
        plt.show()

    regress_plot(x='energy', y='popularity', data=df, xlab='Energy', ylab='Mean Popularity', titl='Mean Popularity vs. Energy')
    regress_plot(x='loudness', y='popularity', data=df, xlab='Loudness', ylab='Mean Popularity', titl='Mean Popularity vs. Loudness')
    regress_plot(x='danceability', y='popularity', data=df, xlab='Danceability', ylab='Mean Popularity', titl='Mean Popularity vs. Danceability')
    regress_plot(x='tempo', y='popularity', data=df, xlab='Tempo', ylab='Mean Popularity', titl='Mean Popularity vs. Tempo')
    regress_plot(x='acousticness', y='popularity', data=df, xlab='Acousticness', ylab='Mean Popularity', titl='Mean Popularity vs. Acousticness')
