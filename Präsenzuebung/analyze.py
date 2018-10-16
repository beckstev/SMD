import numpy as np
import matplotlib.pyplot as plt

gewicht, groesse = np.genfromtxt('Groesse_Gewicht.txt', unpack=True)

# --- Binning Numbers --- #
binning_list = [5, 10, 15, 20, 30, 50]

# --- Create random_numbers for excercise c) and directly log them --- #
random_numbers = np.log(np.random.uniform(1, 100, int(1e2)))

# --- Create figures --- #
fig_grosse, axx_groesse = plt.subplots(2, 3)
fig_gewicht, axx_gewicht = plt.subplots(2, 3)
fig_random, axx_random = plt.subplots(2, 3)


for i in range(0, 6, 1):
    # --- Fill figures --- #
    if i < 3:

        axx_groesse[0, i].hist(groesse, bins=binning_list[i])
        axx_groesse[0, i].set_title('Anzahl Bins: ' + str(binning_list[i]))
        axx_groesse[0, i].set_xlabel('Groesse in m')
        axx_groesse[0, i].set_ylabel('Anzahl Menschen')

        axx_gewicht[0, i].hist(gewicht, bins=binning_list[i])
        axx_gewicht[0, i].set_title('Anzahl Bins: ' + str(binning_list[i]))
        axx_gewicht[0, i].set_xlabel('Groesse')
        axx_gewicht[0, i].set_ylabel('Anzahl Menschen')

        axx_random[0, i].hist(random_numbers, bins=binning_list[i])
        axx_random[0, i].set_title('Anzahl Bins: ' + str(binning_list[i]))
        axx_random[0, i].set_xlabel('Anzahl')
        axx_random[0, i].set_ylabel(r'$\log(Zahl)$')

    else:
        axx_groesse[1, i-3].hist(groesse, bins=binning_list[i])
        axx_groesse[1, i-3].set_title('Anzahl Bins: ' + str(binning_list[i]))
        axx_groesse[0, i-3].set_xlabel('Groesse')
        axx_groesse[0, i-3].set_ylabel('Anzahl Menschen')

        axx_gewicht[1, i-3].hist(gewicht, bins=binning_list[i])
        axx_gewicht[1, i-3].set_title('Anzahl Bins: ' + str(binning_list[i]))
        axx_gewicht[1, i-3].set_xlabel('Groesse')
        axx_gewicht[1, i-3].set_ylabel('Anzahl Menschen')

        axx_random[1, i-3].hist(random_numbers, bins=binning_list[i])
        axx_random[1, i-3].set_title('Anzahl Bins: ' + str(binning_list[i]))
        axx_random[1, i-3].set_xlabel('Anzahl')
        axx_random[1, i-3].set_ylabel(r'$\log(Zahl)$')

# -- Save figures -- #
fig_grosse.tight_layout()
fig_grosse.savefig('./plots/hists_grosse.pdf')

fig_gewicht.tight_layout()
fig_gewicht.savefig('./plots/hists_gewicht.pdf')

fig_random.tight_layout()
fig_random.savefig('./plots/hists_random.pdf')

# ---  Besprechung --- #
# Nicht zu wenig, da sonst Information verloren gehen. Außerdem führt eine zu
# genaue Binnings, zu leeren Bins und man verliert die Struktur. Außerdem
# sollte noch beachtet werden ob Daten gerundet sind oder nicht
# Minimale Binbreite muss mit Nachkommerstellen passen. Bei 2 Nachkommerstellen
# (70.05) maximal so groß wie die Auflösung also 0.01. Außerdem sollte bei dem
# Beispiel von 70.05 die Bin Mitte genau bei 70.05 liegen.
#
# Future Improvments: USE A FUNCTION!!!!!!

for i, j in zip([1,2,3,4], [5,6,7,8]):
    print(i,j)
