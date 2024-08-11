import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, r2_score, make_scorer, accuracy_score, f1_score, classification_report, \
    mean_squared_error, median_absolute_error, silhouette_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier, \
    GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier
import plotly.figure_factory as ff
import numpy as np
from sklearn import svm

df = pd.read_csv('diamonds.csv', sep=',', decimal='.')

print(df.shape)

print(df.dtypes)

print(df.head(10))

null = df.isnull().sum()
print(null)

print(df.head())

df = df.drop(df.columns[0], axis=1)
print(df.columns)

# GRAFOVI



grouped = df.groupby(['clarity', 'color']).agg({'carat': 'mean'}).reset_index()

plt.figure(figsize=(14, 8))
sns.barplot(x='clarity', y='carat', hue='color', data=grouped, palette='viridis')
plt.title('Povprečen karat po jasnosti(clarity) in barvi(color)')
plt.xlabel('Jasnosti(clarity)')
plt.ylabel('Povprečen karat(carat)')
plt.legend(title='Color', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


grouped = df.groupby(['cut', 'color']).agg({'price': 'mean'}).reset_index()

pivot_table = grouped.pivot(index='cut', columns='color', values='price')

plt.figure(figsize=(12, 8))
ax = sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap='Spectral', cbar_kws={'label': 'Povprečna cena(price)'})
plt.title('Povprečne cene diamantov po kakovosti reza(cut) in barvi(color)')
plt.show()

top_10_diamonds = df.nlargest(10, 'price')

fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('tight')
ax.axis('off')
table_data = top_10_diamonds[['price', 'color', 'clarity', 'carat']].rename(columns={'price': 'Cena', 'color': 'Barva', 'clarity': 'Jasnost', 'carat': 'Karat'})
table = ax.table(cellText=table_data.values, colLabels=table_data.columns, cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
plt.title('Tabela 10 najvrednejših diamantov')
plt.show()

# KLASIFIKACIJA

dfKlas = df.copy()

scaler = StandardScaler()
encoder = LabelEncoder()

numericKlas = dfKlas.select_dtypes(include="number").columns
categorical1Klas = dfKlas.drop(['cut'], axis=1).select_dtypes(include='object').columns

dfKlas[numericKlas] = scaler.fit_transform(dfKlas[numericKlas])
dfKlas[categorical1Klas] = encoder.fit_transform(categorical1Klas)

print(dfKlas.columns)
print(dfKlas.head(10))

a = dfKlas.drop('cut', axis=1)
b = dfKlas['cut']

a_train, a_test, b_train, b_test = train_test_split(a, b, test_size=0.3, random_state=42)

classifiers = [
    ('Logistična regresija', LogisticRegression(max_iter=1000)),
    ('Naključni gozdovi', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('Gradijentno ojačana drevesa', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('K-najbližjih sosedov', KNeighborsClassifier(n_neighbors=5))
]

for clf_name, clf in classifiers:
    clf.fit(a_train, b_train)

    y_pred = clf.predict(a_test)

    accuracy = accuracy_score(b_test, y_pred)
    f1 = f1_score(b_test, y_pred, average='weighted')

    print(f'{clf_name} Accuracy: {accuracy:.6f}')
    print(f'{clf_name} F1 Score: {f1:.6f}')
    print('-' * 50)

# REGRESIJA

dfReg = df.copy()

numericReg = dfReg.drop('price', axis=1).select_dtypes(include="number").columns
categoricalReg = dfReg.select_dtypes(include='object').columns

dfReg[numericReg] = scaler.fit_transform(dfReg[numericReg])
dfReg[categoricalReg] = encoder.fit_transform(categoricalReg)

x = dfReg.drop('price', axis=1)
y = dfReg['price']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

models = {
    'Linearna regresija': LinearRegression(fit_intercept=True, positive=False),
    'Naključni gozdovi': RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=2, random_state=42),
    'Gradijentno ojačana drevesa': GradientBoostingRegressor(n_estimators=100, max_depth=3, min_samples_split=2,
                                                             min_samples_leaf=1, random_state=42),
    'Regresor podpornih vektorjev': SVR(C=1.0, epsilon=0.2)
}

for name, model in models.items():
    print(f"Usposabljanje modela: {name}")
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'{name} - MAE: {mae:.2f}')
    print(f'{name} - R2: {r2:.2f}')
    print('-' * 50)

# GRUCENJE

dfClust = df.copy()

numericClust = dfClust.select_dtypes(include=['number']).columns.tolist()

scaler = StandardScaler()
df_scaled = scaler.fit_transform(dfClust[numericClust])


def k_elbow_method(data, max_k=10):
    distortions = []
    K = range(1, max_k + 1)
    for k in K:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(data)
        distortions.append(km.inertia_)

    plt.figure(figsize=(8, 4))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('Stevilo klasterov (k)')
    plt.ylabel('Izkrivljanje')
    plt.title('K-elbow metod')
    plt.show()


k_elbow_method(df_scaled, max_k=10)

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(df_scaled)

dfClust['Cluster'] = clusters


pca = PCA(n_components=2)
principalComponents = pca.fit_transform(df_scaled)
df_pca = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])
df_pca['Cluster'] = clusters

plt.figure(figsize=(10, 7))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', palette='viridis', data=df_pca, legend='full')
plt.title('Grucenje')
plt.show()
