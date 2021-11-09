import cv2
import numpy as np
from numpy import linalg as la


koord =  [] #lista odabranih tacaka

def homogena(tacka):
    return [tacka[0], tacka[1], 1]

def sortiranje(tacke):
    sortirane = sorted(tacke)
    
    if sortirane[1][1] < sortirane[0][1]:
        sortirane[0], sortirane[1] = sortirane[1], sortirane[0]
    if sortirane[2][1] < sortirane[3][1]:
        sortirane[2], sortirane[3] = sortirane[3], sortirane[2]
    return sortirane


def matrica_korespodencije(o, s):
    m = np.matrix([[0, 0, 0, -s[2]*o[0], -s[2]*o[1], -s[2]*o[2], s[1]*o[0], s[1]*o[1], s[1]*o[2]],
     [s[2]*o[0], s[2]*o[1], s[2]*o[2], 0, 0, 0, -s[0]*o[0], -s[0]*o[1], -s[0]*o[2]]])

    return m

def dlt_algoritam(n, originali, slike):
    
    matrica = matrica_korespodencije(originali[0], slike[0])
    
    for i in range(1, n):
        m = matrica_korespodencije(originali[i], slike[i])
        matrica = np.concatenate((matrica, m), axis = 0)
    
    U, D, Vt = la.svd(matrica, full_matrices=True)
    
    P = Vt[-1]
    P = P.reshape(3,3)

    return P

def click_event(event, x, y, flags, params):

    if event == cv2.EVENT_LBUTTONDOWN:
        koord.append((x,y))
        print((x, y))
        font = cv2.FONT_HERSHEY_SIMPLEX
        color =  (102, 255, 178)
        cv2.putText(img, str(x) + ', ' + str(y), (x, y), font, 0.5, color, 2)
        cv2.imshow('slika', img)

# za slike odabranih tacaka uzimamo temena pravougaonika
# za duzine stranica pravougaonika uzmemo npr. aritmeticku sredinu naspramnih stranica originala
def nadji_slike(originali):
    slike = []

    duzina = ((originali[2][0] - originali[1][0])+(originali[3][0] - originali[0][0]))/2.0
    sirina = ((originali[1][1] - originali[0][1])+(originali[2][1] - originali[3][1]))/2.0 

    slike.append((originali[0][0], originali[0][1], originali[0][2]))
    slike.append((originali[0][0], originali[0][1] + sirina, originali[0][2]))
    slike.append((originali[0][0] + duzina, originali[0][1] + sirina, originali[0][2]))     
    slike.append((originali[0][0] + duzina, originali[0][1], originali[0][2])) 

    return slike

# ucitavanje slike
print('Unesite naziv slike: ')
img_name = input()
img = cv2.imread(img_name, 1)
img = cv2.resize(img,(900,600), interpolation = cv2.INTER_AREA)
cv2.imshow('slika', img)
cv2.setMouseCallback('slika', click_event)

while True:
    if cv2.waitKey(1) & len(koord) == 4:
        break

originali = []
for i in range(4):
    originali.append(homogena(koord[i]))

originali = sortiranje(originali)
slike = nadji_slike(originali)

matrica = dlt_algoritam(4, originali, slike)
matrica = np.round(matrica,decimals=10)

img = cv2.imread(img_name, 1)
img = cv2.resize(img,(900,600))

# otklanjanje distorzije
M = np.float32(matrica)
dst = cv2.warpPerspective(img,M,(900,600))

cv2.imshow('rezultat',dst)
cv2.waitKey(0)





