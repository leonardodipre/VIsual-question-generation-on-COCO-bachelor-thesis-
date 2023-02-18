import pandas as pd
import matplotlib.pyplot as plt
import numpy


# Carica il file CSV in un DataFrame di Pandas

df = pd.read_csv(r'D:\Leonardo\VQG_final\modelli\graph\beam_search\BLUE_beam_search.csv')


# Estrae i valori numerici dalle stringhe delle colonne 'Epoch' e 'Loss'
epochs = df['epochs']
BLEU1 = df['BLEU1']
BLEU2 = df['BLEU2']
BLEU3 = df['BLEU3']
BLEU4 = df['BLEU4']



plt.plot(epochs.astype(str), BLEU1, label="BLUE1")

plt.plot(epochs.astype(str), BLEU2, label="BLUE2")

plt.plot(epochs.astype(str), BLEU3, label="BLUE3")


plt.plot(epochs.astype(str), BLEU4, label="BLUE4")

plt.legend()

# Aggiunge un titolo e delle etichette agli assi
plt.title('BLEU')
plt.xlabel('model')


# Mostra il grafico
plt.savefig('BLEU_total_beam_search.png')
