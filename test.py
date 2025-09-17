# =============================================================================
# Korak 1: Ucitavanje biblioteka i podataka
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

# lista imena kolona
column_names = [
    "id",
    "diagnosis",    # ciljna promenljiva (B -> 0 ili M -> 1)
    "mean radius", "mean texture", "mean perimeter", "mean area",
    "mean smoothness", "mean compactness", "mean concavity", "mean concave points",
    "mean symmetry", "mean fractal dimension",
    "radius se", "texture se", "perimeter se", "area se",
    "smoothness se", "compactness se", "concavity se", "concave points se",
    "symmetry se", "fractal dimension se",
    "worst radius", "worst texture", "worst perimeter", "worst area",
    "worst smoothness", "worst compactness", "worst concavity", "worst concave points",
    "worst symmetry", "worst fractal dimension"
]

# učitavanje sa imenima kolona
df = pd.read_csv("data.csv", names=column_names)

# =============================================================================
# Korak 2: Ciscenje i pocetna analiza
# =============================================================================
# mapiranje B -> 0, M -> 1
df["diagnosis"] = df["diagnosis"].map({"B": 0, "M": 1}) 

# Provera nedostajucih vrednosti
print("\nNedostajuce vrednosti po kolonama:\n", df.isnull().sum())

# Uklanjanje atributa koji ocigledno ne uticu na izlaz
if 'id' in df.columns:
    df = df.drop(columns=['id'])

print("\nPrvih 5 redova:\n", df.head())
print("\nBroj uzoraka po klasama:\n", df["diagnosis"].value_counts())
df.info()

# =============================================================================
# Korak 3: Eksplorativna analiza skupa
# =============================================================================
correlation_matrix = df.corr()

# Vizualizacija korelacionog matriksa kao toplotne mape
plt.figure(figsize=(20, 18))
sbn.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Korelaciona matrica atributa', fontsize=20)
plt.show()

# Prikaz korelacija sa ciljnom promenljivom 'diagnosis'
print("\nKorelacija sa ciljnom promenljivom 'diagnosis':")
print(correlation_matrix['diagnosis'].sort_values(ascending=False))

# Kreiranje box plotova za sve atribute
plt.figure(figsize=(20, 15))
for i, column in enumerate(df.drop(columns=['diagnosis']).columns):
    plt.subplot(5, 6, i + 1)
    sbn.boxplot(x=df[column])
    plt.title(column, fontsize=10)
    plt.tight_layout()
plt.suptitle('Box Plotovi za sve atribute', fontsize=20, y=1.02)
plt.show()

# Prikaz statističkog pregleda
print("\nStatistički pregled atributa:")
print(df.describe())

# =============================================================================
# Korak 4: Podela i skaliranje podataka
# =============================================================================
# Odvajanje atributa (X) i ciljne promenljive (y)
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# Podela podataka na trening (80%) i test (20%) set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Oblik trening skupa: {X_train.shape}")
print(f"Oblik test skupa: {X_test.shape}")

# Skaliranje atributa (obavezno za LR, SVM, KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =============================================================================
# Korak 5: Treniranje svih modela
# =============================================================================
# Inicijalizacija rečnika za čuvanje predikcija i metrika
predictions = {}
performance_metrics = {}
models = {}

# Logisticka regresija
print("--- Treniranje modela: Logistička Regresija ---")
model_lr = LogisticRegression(solver='liblinear', random_state=42)
model_lr.fit(X_train_scaled, y_train)
y_pred_lr = model_lr.predict(X_test_scaled)
predictions['Logistička Regresija'] = y_pred_lr
models['Logistička Regresija'] = model_lr
print("Model Logističke Regresije je obučen.")

# Slučajna šuma
print("--- Treniranje modela: Slučajna Šuma ---")
model_rf = RandomForestClassifier(random_state=42)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)
predictions['Slučajna Šuma'] = y_pred_rf
models['Slučajna Šuma'] = model_rf
print("Model Slučajne Šume je obučen.")

# SVM
print("--- Treniranje modela: SVM ---")
model_svm = SVC(random_state=42)
model_svm.fit(X_train_scaled, y_train)
y_pred_svm = model_svm.predict(X_test_scaled)
predictions['SVM'] = y_pred_svm
models['SVM'] = model_svm
print("SVM model je obučen.")

# KNN
print("--- Treniranje modela: KNN ---")
model_knn = KNeighborsClassifier()
model_knn.fit(X_train_scaled, y_train)
y_pred_knn = model_knn.predict(X_test_scaled)
predictions['KNN'] = y_pred_knn
models['KNN'] = model_knn
print("KNN model je obučen.")


# =============================================================================
# Korak 6: Evaluacija i poređenje performansi
# =============================================================================
print("\n--- Rezultati i poređenje performansi ---")

for model_name, y_pred in predictions.items():
    print(f"\n--- Model: {model_name} ---")
    
    # Računanje metrika i čuvanje za tabelu
    performance_metrics[model_name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred)
    }

    # Matrica konfuzije
    cm = confusion_matrix(y_test, y_pred)
    
    # Prikaz matrice konfuzije
    plt.figure(figsize=(6, 5))
    sbn.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Benigno (0)', 'Maligno (1)'],
                yticklabels=['Benigno (0)', 'Maligno (1)'])
    plt.xlabel('Predviđena klasa')
    plt.ylabel('Stvarna klasa')
    plt.title(f'Matrica konfuzije za {model_name}')
    plt.show()
    
    # Prikaz izveštaja o klasifikaciji
    print("Izveštaj o klasifikaciji:")
    print(classification_report(y_test, y_pred))

# =============================================================================
# Korak 7: Vizuelna analiza i tumacenje rezultata
# =============================================================================
# Prikaz tabela performansi
metrics_df = pd.DataFrame(performance_metrics).T
print("--- Tabela performansi svih modela ---")
print(metrics_df.to_string(float_format="%.4f"))

# Vizualizacija metrika
metrics_df.plot(kind='bar', figsize=(12, 7), rot=0)
plt.title('Poređenje performansi klasifikacionih modela', fontsize=16)
plt.ylabel('Vrednost metrike', fontsize=12)
plt.xlabel('Model', fontsize=12)
plt.ylim(0.9, 1.0)
plt.legend(title='Metrike', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Analiza važnosti atributa (samo za Random Forest)
rf_model = models['Slučajna Šuma']
feature_importances = pd.DataFrame({
    'Atribut': X.columns,
    'Važnost': rf_model.feature_importances_
}).sort_values(by='Važnost', ascending=False)

print("\n--- 10 najvažnijih atributa prema modelu Slučajna Šuma ---")
print(feature_importances.head(10).to_string(index=False, float_format="%.4f"))

# Vizualizacija najvažnijih atributa
plt.figure(figsize=(12, 8))
sbn.barplot(x='Važnost', y='Atribut', data=feature_importances.head(10), palette='viridis')
plt.title('Važnost atributa (Slučajna Šuma)', fontsize=16)
plt.xlabel('Važnost', fontsize=12)
plt.ylabel('Atribut', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.show()

# Tumačenje rezultata
print("\n--- Tumačenje rezultata i preporuke ---")
print("Svi modeli su pokazali izuzetno visoke performanse, što ukazuje na to da su podaci visokog kvaliteta.")
print("Modeli **Slučajne Šume** i **SVM** su postigli gotovo savršene rezultate na test podacima, sa visokim preciznim odzivom za malignu klasu.")
print("Najvažniji atributi, kao što su 'worst concave points' i 'worst perimeter', ključni su za razlikovanje benignih i malignih tumora.")
print("\nPreporuka za poboljšanje:")
print("- Iako su performanse visoke, uvek je moguće dalje eksperimentisati sa hiperparametrima (npr. 'C' za SVM ili 'n_neighbors' za KNN) kako bi se postigli još bolji rezultati ili eliminisali preostali lažno negativni slučajevi.")