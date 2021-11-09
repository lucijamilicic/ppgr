import numpy as np

class Tacka:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
def homogena(t):
    return [t.x, t.y, 1]

def afina(T):
    return Tacka(round(T[0] / T[2]), round(T[1] / T[2]))

def nevidljiva_tacka(A, B, C, D, A1, B1, C1):
    A = homogena(A)
    B = homogena(B)
    C = homogena(C)
    D = homogena(D)
    A1 = homogena(A1)
    B1 = homogena(B1)
    C1 = homogena(C1)

    # paralelne ivice AA1, BB1, CC1 i DD1 seku se u P
    AA1 = np.cross(A, A1)
    BB1 = np.cross(B, B1)
    CC1 = np.cross(C, C1)
    
    P1 = np.cross(AA1, BB1)
    P2 = np.cross(AA1, CC1)
    P3 = np.cross(BB1, CC1)
    
    P = (P1 + P2 + P3) / 3
    
    # paralelne ivice BC, AD, B1C1 i A1D1 seku se u Q
    BC = np.cross(B, C)
    AD = np.cross(A, D)
    B1C1 = np.cross(B1, C1)
    
    Q1 = np.cross(BC, AD)
    Q2 = np.cross(BC, B1C1)
    Q3 = np.cross(B1C1, AD)
    
    Q = (Q1 + Q2 + Q3) / 3

    #Trazeno teme D1:
    PD = np.cross(P,D)
    A1Q = np.cross(A1,Q)
    D1 = np.cross(PD, A1Q)

    return afina(D1)
     
#koordinate sa moje slike
C1 = Tacka(549, 375) #1
B1 = Tacka(317, 671) #2
A1 = Tacka(60, 505) #3
C = Tacka(573, 237) #5
B = Tacka(306, 513) #6
A = Tacka(8, 362) #7
D = Tacka(330, 187) #8

t = nevidljiva_tacka(A, B, C, D, A1, B1, C1)
print("Nevidljivo teme sa prilozene slike: (", t.x, ",", t.y , ")")

#zadati test primer
C1 = Tacka(595, 301) #1
B1 = Tacka(292, 517) #2
A1 = Tacka(157, 379) #3
C = Tacka(665, 116) #5
B = Tacka(304, 295) #6
A = Tacka(135, 163) #7
D = Tacka(509, 43) #8

nevidljiva_tacka(A, B, C, D, A1, B1, C1)
t = nevidljiva_tacka(A, B, C, D, A1, B1, C1)
print("Nevidljivo teme na zadatom test primeru: (", t.x, ",", t.y , ")")
