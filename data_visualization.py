from imports import *

def visualize_data(df):
    classified = df.copy()
    classified['pop_rating'] = ''

    for i, row in classified.iterrows():
        score = 'nepopular'
        if (row.popularity > 50) & (row.popularity < 75):
            score = 'mediu'
        elif row.popularity >= 75:
            score = 'popular'
        classified.at[i, 'pop_rating'] = score

    fig, ax = plt.subplots(1,1, figsize=(8,5))
    sns.countplot(x='pop_rating', data=classified, palette="viridis")
    plt.xlabel('Evaluări de popularitate', fontsize=14)
    plt.ylabel('Numărul de înregistrări', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    numeric_columns = df.columns[df.dtypes != 'object']
    numeric_df = pd.DataFrame(data=df, columns=numeric_columns, index=df.index)

    corr = np.abs(numeric_df.corr())
    fig, ax = plt.subplots(figsize=(10, 10))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(corr, cmap=cmap, square=True, annot=True, fmt=".2f")
    plt.title('Corelația între variabilele numerice', fontsize=18, weight='bold')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    corr = numeric_df.corr()[['popularity']].sort_values(by='popularity', ascending=False)
    plt.figure(figsize=(8, 10))
    heatmap = sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    heatmap.set_title('Caracteristici corelate cu popularitatea', fontdict={'fontsize':20}, pad=16)

    fig, ax = plt.subplots(figsize=(15, 6))
    ax = df.groupby('year')['popularity'].mean().plot(kind='line', color='coral', marker='o')
    ax.set_title('Evoluția popularității medii de-a lungul anilor', fontsize=18, color='coral', weight='bold')
    ax.set_ylabel('Popularitatea medie', fontsize=14, weight='bold')
    ax.set_xlabel('Anul', fontsize=14, weight='bold')
    ax.set_xticks(range(1920, 2021, 10))
    ax.set_xticklabels(range(1920, 2021, 10), rotation=45)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

    def regress_plot(x='', y='', data=None, xlab='', ylab='', titl=''):
        '''Plots a scatterplot with a regression line using given inputs'''
        data = data.groupby(x)[y].mean().to_frame().reset_index()
        fig, ax = plt.subplots(figsize=(10,6))
        sns.regplot(x=x, y=y, data=data, scatter_kws={'color': 'blue', "s": 50}, line_kws={'color':'darkred'} )
        plt.xlabel(xlab, fontsize=14)
        plt.ylabel(ylab, fontsize=14)
        plt.title(titl, fontsize=16, color='darkred')
        plt.ylim(-3, 103)
        plt.grid(True, which='major', linestyle='--', linewidth=0.5)
        plt.show()

    regress_plot(x='energy', y='popularity', data=df, xlab='Energie', ylab='Popularitatea medie', titl='Popularitatea medie vs. Energie')
    regress_plot(x='loudness', y='popularity', data=df, xlab='Intensitate Sonoră', ylab='Popularitatea medie', titl='Popularitatea medie vs. Intensitate Sonoră')
    regress_plot(x='danceability', y='popularity', data=df, xlab='Dansabilitate', ylab='Popularitatea medie', titl='Popularitatea medie vs. Dansabilitate')
    regress_plot(x='tempo', y='popularity', data=df, xlab='Tempo', ylab='Popularitatea medie', titl='Popularitatea medie vs. Tempo')
    regress_plot(x='acousticness', y='popularity', data=df, xlab='Acustică', ylab='Popularitatea medie', titl='Popularitatea medie vs. Acustică')
