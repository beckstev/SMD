import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

np.random.seed(42)

# möglicher Algorithmus für das Neumannsche Rückweisungsverfahren
def Rückweisungsverfahren(function, Anzahl, x, y):

    verteilung_x = []
    verteilung_y = []

    for i in range(Anzahl):
        if y[i] <= function(x[i]):
            verteilung_x.append(x[i])
            verteilung_y.append(y[i])
    return verteilung_x, verteilung_y

# Normierungskonstante der Planck-Verteilung
N = np.pi**4 / 15

def planck_verteilung(x):
    return N * x**3 / (np.exp(x) - 1)

################## Aufgabenteil a.) ##################

# Anzahl der Zufallszahlen
Anzahl_zufallszahlen = int(1e5)

# obere Grenze der x-Werte
x_cut = 20

# untere Grenze der x-Werte
x_min = 0

# x Intervall
x_intervall = np.linspace(x_min, x_cut, Anzahl_zufallszahlen)

#obere Grenze der y-Werte
def ableitung_planck(x):
    return (3 * x**2 * (np.exp(x) - 1) - x**3 * np.exp(x)) / (np.exp(x) - 1)**2

y_cut = planck_verteilung(brentq(ableitung_planck, x_min + 0.5, x_cut))

#untere Grenze der y-Werte
y_min = planck_verteilung(x_cut)

# Zufallszahlen der y-Achse der Planck Verteilung 
random_y = np.random.uniform(y_min, y_cut, 5 * Anzahl_zufallszahlen)

# Zufallszahlen der x-Achse der gegebenen Verteilungsfunktion
random_x = np.sort(np.random.uniform(x_min, x_cut, 5 * Anzahl_zufallszahlen))

verteilung_x, verteilung_y = Rückweisungsverfahren(planck_verteilung, 5 * Anzahl_zufallszahlen, random_x, random_y)

# Plotten der durch das Neumannsche Rückweisungsverfahren gefilterten Zufallszahlen
plt.plot(verteilung_x, verteilung_y, 'r.', label = 'Planckverteilte Zufallszahlen')
# Plotten der Daten der Planck Verteilung zum Vergleichen
plt.plot(x_intervall, planck_verteilung(x_intervall), 'b-', label = 'Planckverteilung')
plt.legend(loc = 'best')
plt.tight_layout()
#plt.savefig('Pics/planck_verteilung.pdf')

print(len(verteilung_x))

################## Aufgabenteil b.) ##################

# Bestimmen von x_s

def majorante_xs(x):
    return y_cut - 200 * N * x**(-0.1) * np.exp(-x**(0.9))   

x_s = brentq(majorante_xs, 0.1, x_cut)

# Inversionsmethode

N_trafo = 9 / 2000 / (np.exp(-x_s**(0.9)) - np.exp(-x_cut**(0.9)))

def transformation_rückweisung(function, Anzahl, x_s, x_min, x_max, y_min, y_max):
    # Erzeugen von mehr Zufallszahlen als gefordert, um nachher zahlen "wegzuwerfen"
    buffer_x = np.append(np.random.uniform(x_min, x_s, 3 * int(0.8 * Anzahl)), np.random.uniform(x_s, x_max, 3 * int(0.2 * Anzahl)))
    buffer_y = np.random.uniform(y_min, y_max, 3 * Anzahl)
    buffer_x[buffer_x > x_s] = transformation(np.random.uniform(0, 1, len(buffer_x[buffer_x > x_s])))
    verteilung_x, verteilung_y = (Rückweisungsverfahren(function, 3 * Anzahl, buffer_x, buffer_y))
    # Zufällig auswhlen welche Elemente nicht übergeben werden sollen
    random_index = np.random.choice(range(0, len(verteilung_x), 1), Anzahl, replace=False)
    return np.array(verteilung_x)[random_index], np.array(verteilung_y)[random_index]

def transformation(u):
    return (-np.log(-9 * u / (2000 * N_trafo) + np.exp(-x_s**(0.9))))**(10 / 9)

def majorante_planck(x):
    y = []
    for i in range(len(x)):
        if x[i] <= x_s:
            y.append(y_cut)
        else:
            y.append(200 * N * x[i]**(-0.1) * np.exp(-x[i]**(0.9)))         
    return y 

# Erzeugen der Zufallszahlen gemäß der Majorantenfunktion
planck_majorante_x, planck_majorante_y = transformation_rückweisung(planck_verteilung, Anzahl_zufallszahlen, x_s, x_min, x_cut, y_min, y_cut)

print(max(planck_majorante_x))

plt.clf()
plt.plot(planck_majorante_x, planck_majorante_y, 'r.', label = 'Planckverteilte Zufallszahlen')
plt.plot(x_intervall, planck_verteilung(x_intervall), 'b-', label = 'Planckverteilung')
plt.plot(x_intervall, majorante_planck(x_intervall), 'k--', label = 'Majorante')
plt.legend(loc = 'best')
plt.tight_layout()
#plt.savefig('Pics/Zufallszahlen_aus_Majorante_planck.pdf')
