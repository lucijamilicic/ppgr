import cv2
import numpy as np
from numpy import linalg as la

np.set_printoptions(suppress = True)


# funkcija za ucitavanje piksel koordinata odabranih tacaka
def ucitaj_koordinate():

    koord = []
    def click_event(event, x, y, flags, params):

        if event == cv2.EVENT_LBUTTONDOWN:
            koord.append((x,y))
            # print((x, y))
            font = cv2.FONT_HERSHEY_SIMPLEX
            color =  (102, 255, 178)
            cv2.putText(img, str(x) + ', ' + str(y), (x, y), font, 0.5, color, 2)
            cv2.imshow('slika', img)

    img = cv2.imread("scena.jpeg", 1)
    img = cv2.resize(img,(900,600), interpolation = cv2.INTER_AREA)
    cv2.imshow('slika', img)
    cv2.setMouseCallback('slika', click_event)

    while True:
        if cv2.waitKey(1) & len(koord) == 6:
            break
    
    cv2.waitKey(0)

    return koord


def homogene(koord):
    
    lista = []
    for i in range(6):
        M = np.array([koord[i][0], koord[i][1], 1])
        lista.append(M)

    return np.array(lista)


def cameraEquations(o, p):
    m = np.matrix([[0, 0, 0, 0, -p[2]*o[0], -p[2]*o[1], -p[2]*o[2],  -p[2]*o[3], p[1]*o[0], p[1]*o[1], p[1]*o[2],p[1]*o[3]],
                   [p[2]*o[0], p[2]*o[1], p[2]*o[2], p[2]*o[3], 0, 0, 0, 0, -p[0]*o[0], -p[0]*o[1], -p[0]*o[2], -p[0]*o[3]]])
    return m

def cameraDLP(originali, projekcije):
    
    matrica = cameraEquations(originali[0], projekcije[0])
    
    for i in range(1, 6):
        m = cameraEquations(originali[i], projekcije[i])
        matrica = np.concatenate((matrica, m), axis=0)
            
    U, D, Vt = la.svd(matrica)
    
    #matrica P ce biti poslednja kolona matrice V sto je ustvari poslednja vrsta matrica Vt
    #skaliramo tako da element na poziciji (0,0) bude 1
    P = Vt[-1] / Vt[-1, 0]
    P = P.round(4)
    P = P.reshape(3, 4)
   
    return P

# originali - koordinate izmerene na sceni (mm):
M1 = np.array([32, 118, 160, 1])
M2 = np.array([60, 241, 0, 1])
M3 = np.array([111, 186, 96, 1])
M4 = np.array([146, 153, 0, 1])
M5 = np.array([96, 0, 15, 1])
M6 = np.array([190, 0, 15, 1])
originali = np.array([M1, M2, M3, M4, M5, M6])

# projekcije - ucitavaju se sa slike:
projekcije = homogene(ucitaj_koordinate())

P = cameraDLP(originali, projekcije)
print("Dobijena matrica:")
print(P)
