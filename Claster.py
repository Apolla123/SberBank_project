import json
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

plt.style.use('seaborn-v0_8-white')

with open('Data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

df = pd.DataFrame(data)
df['year'] = df['period'].str.split('_').str[0]
df_grouped = (
    df.groupby(['year', 'municipality_id'])['all_share'].mean().reset_index()
)

df_2023 = df_grouped[df_grouped['year'] == '2023'].copy()
df_2024 = df_grouped[df_grouped['year'] == '2024'].copy()

N_CLUSTERS = 3
CLUSTER_COLORS = {
    0: '#2EC4B6',
    1: '#FF9F1C',
    2: '#E71D36',
    3: '#3A86EF',
}


def plot_clean_clusters(df_year, year_title):
    if df_year.empty:
        print(f'Данные за {year_title} год отсутствуют в файле.')
        return

    X = df_year[['all_share']].values
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    df_year['cluster'] = kmeans.fit_predict(X)

    df_year = df_year.sort_values(by='all_share').reset_index(drop=True)

    cluster_order = (
        df_year.groupby('cluster')['all_share']
        .mean()
        .sort_values()
        .index.tolist()
    )
    cluster_map = {old: new for new, old in enumerate(cluster_order)}
    df_year['cluster_sorted'] = df_year['cluster'].map(cluster_map)

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=130)

    for c_id in range(N_CLUSTERS):
        subset = df_year[df_year['cluster_sorted'] == c_id]

        ax.scatter(
            subset.index,
            subset['all_share'],
            color=CLUSTER_COLORS[c_id],
            s=45, 
            alpha=0.85,
            edgecolor='white',
            linewidth=0.5,
            label=f'Кластер {c_id + 1} ({subset["all_share"].min():.3f} — {subset["all_share"].max():.3f})',
            zorder=3,
        )

    ax.set_title(
        f'Распределение и кластеризация МО по тратaм (all_share) — {year_title}',
        fontsize=13,
        fontweight='bold',
        pad=15,
        color='#2B2D42',
    )
    ax.set_xlabel('Муниципальные образования (ранжированы по росту трат)', fontsize=10, color='#555555', labelpad=8)
    ax.set_ylabel('Показатель трат (all_share)', fontsize=10, color='#555555', labelpad=8)

    
    ax.grid(True, axis='y', linestyle=':', alpha=0.6, color='#A0AAB2', zorder=0)


    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')

    ax.legend(
        title='Уровень трат МО',
        title_fontsize='10',
        fontsize=9,
        frameon=True,
        facecolor='#FFFFFF',
        edgecolor='#E0E0E0',
        loc='upper left',
    )

    plt.tight_layout()
    plt.show()


plot_clean_clusters(df_2023, '2023')
plot_clean_clusters(df_2024, '2024')