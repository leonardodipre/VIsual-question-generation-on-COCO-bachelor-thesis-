import pandas as pd
import matplotlib.pyplot as plt

# Carica il file CSV in un DataFrame di Pandas
df = pd.read_csv('D:\Leonardo\VQG_final\modelli\loss_valuesVQG_1_Final.csv', header=None, names=['Epoch', 'Loss'])

# Estrae i valori numerici dalle stringhe delle colonne 'Epoch' e 'Loss'
df['Epoch'] = df['Epoch'].str.extract('(\d+)')
df['Loss'] = df['Loss'].str.extract('(\d+\.\d+)').astype(float)

lista_epochs = df['Epoch']
lista_loss = df['Loss']

plt.scatter(lista_epochs.astype(str), lista_loss)

# Aggiunge un titolo e delle etichette agli assi
plt.title('Loss per Epoch')
plt.xlabel('Loss')
plt.ylabel('Epoch')

# Mostra il grafico
plt.show()