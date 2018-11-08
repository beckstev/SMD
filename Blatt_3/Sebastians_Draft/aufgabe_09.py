import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def planck_verteilung(x):
    return 15 / np.pi**4 * x**3 / (np.exp(x) - 1)

def Übergangswahrsch(function, x_i, x_j):
    return min(1, function(x_j) / function(x_i))

def Vorschlags_PDF(x, s):
    return abs(np.random.uniform(x - s, x + s, 200)[np.random.random_integers(0, 199, 1)])

def Metropolis(function, x_0, s, anzahl):
    x = [x_0]
    random = np.random.uniform(0, 1, anzahl)
    for i in range(anzahl - 1):
        x_j = Vorschlags_PDF(x[i], s)
        if random[i] <= Übergangswahrsch(function, x[i], x_j):
            x.append(x_j)
        else:
            x.append(x[i])           
    return x   

# Startwert
x_0 = 30

# Schrittweite
step_size = 2

anzahl = int(1e5)

zufallszahlen_planck = Metropolis(planck_verteilung, x_0, step_size, anzahl)

plt.clf()
plt.hist(zufallszahlen_planck, bins = 100, density=True, histtype='step', label='Zufallszahlen aus Metropolis')
plt.plot(np.linspace(0.01, 30, 1000), planck_verteilung(np.linspace(0.01, 30, 1000)), label='Planck-Verteilung')
plt.ylabel('y')
plt.xlabel('x')
plt.legend(loc='best')
plt.savefig('Pics/a9_b_c.pdf')

plt.clf()
plt.plot(np.linspace(1, anzahl, anzahl), zufallszahlen_planck, 'g,', label='Iteratrionsschritt gegen Zufallszahl')
plt.xscale('log')
plt.ylabel('y')
plt.xlabel('x')
plt.legend(loc='best')
plt.savefig('Pics/a9_d_Trace_plot.pdf')

print(planck_verteilung(np.mean(zufallszahlen_planck)), max(planck_verteilung(np.linspace(0.01, 30, 1000))))

# Die erzeugten Zufallszahlen konvergieren in den ersten hundert Iterationen sehr schnell gegen kleine x-Werte und fluktuieren dann
# anschließend in der Umgebung um das Maximum umher