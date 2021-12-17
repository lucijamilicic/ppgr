from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib import animation
from numpy import linalg as la
import numpy as np
import math


def euler_2_a(fi, teta, psi):
    
    Rx_fi = np.array([[1, 0, 0],
                    [0, math.cos(fi), -math.sin(fi)],
                    [0, math.sin(fi), math.cos(fi)]])
    
    Ry_teta = np.array([[math.cos(teta), 0, math.sin(teta)],
                      [0, 1, 0],
                      [-math.sin(teta), 0, math.cos(teta)]])
    
    Rz_psi = np.array([[math.cos(psi), -math.sin(psi), 0],
                     [math.sin(psi), math.cos(psi), 0],
                     [0, 0, 1]])
    
    A = np.matmul(Ry_teta, Rx_fi)
    A = np.matmul(Rz_psi, A)
    A = np.round(A, 6)
    
    return A


def axis_angle(A):
    
    if round(la.det(A), 4) != 1:
        print("Determinanta matrice A mora biti 1")
        print(round(la.det(A), 5))
        return

    if np.any(np.round(A.dot(A.T), 5) != np.eye(3)):
        print("Matrica A mora biti ortogonalna")
        return
    
    # odredjivanje sopstvenog vektora
    lambdas, vector = la.eig(A)
    for i in range(len(lambdas)):
        if np.round(lambdas[i], 6) == 1.0:
            p = np.real(vector[:, i])

    p1, p2, p3 = p[0], p[1], p[2]
    
    if p1 == 0 and p2 == 0 and p3 == 0:
        print("Ne sme biti nula vektor")
        return
        
    # u je izabran prozvoljni jedinicni vektor koji je normalan na p 
    u = np.cross(p, np.array([1, 1, 1]))
    u = u/math.sqrt(u[0]**2+u[1]**2+u[2]**2)

    up = A.dot(u)

    fi = np.round(math.acos(np.dot(u, up)), 6)
    if np.round(np.dot(np.cross(u, up), p), 6) < 0:
        p = (-1)*p

    
    return [p, fi]


def axis_angle_2_q(p, fi):
    
    norma = la.norm(p)
    
    if(norma == 0):
        print('Ne sme biti nula vektor')
        return
        
    p /= norma
        
    [x, y, z] = math.sin(fi/2.0) * p
    w = round(math.cos(fi/2.0), 6)

    return [x, y, z, w]

#Linearna interpolacija
def lerp(q1, q2, tm, t):
    return (1-(t/tm))*q1 + (t/tm)*q2

#Sferna linearna interpolacija
def slerp(q1, q2, tm, t):
    cos0 = np.dot(q1, q2)
    if cos0 < 0:
        q1 = -1 * q1
        cos0 = -cos0
    if cos0 > 0.95:
        return lerp(q1, q2, tm, t)
    angle = math.acos(cos0)
    q = (math.sin(angle*(1-t/tm)/math.sin(angle)))*q1 + (math.sin(angle*(t/tm)/math.sin(angle)))*q2
    return q

def q_conjugate(q):
    x, y, z, w = q
    return (-x, -y, -z, w)

def q_mult(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return x, y, z, w

# rotacija kvaternionom:  q*v*q_conj
def transform(v, q):
    v1 = (v[0], v[1], v[2], 0.0)
    return q_mult(q_mult(q, v1), q_conjugate(q))[:-1]
    
    
    # ---------------------------- 

# broj frejmova
tm = 100

# koordinate centra pocetnog i krajnjeg polozaja
pos1 = np.array([0, 4, 1])
pos2 = np.array([9, 1, 7])

# Ojlerovi uglovi pocetnog i krajnjeg polozaja
e_angle1 = np.array([math.radians(-30), math.radians(45), math.radians(60)])
e_angle2 = np.array([math.radians(20), math.radians(-30), math.radians(90)])

# Na osnovu Ojlerovih uglova dobija se matrica A 
# zatim se ona predstavi pomocu ose i ugla, na osnovu kojih se dobije jedinicni kvaternion
A = euler_2_a(e_angle1[0], e_angle1[1], e_angle1[2])
N = axis_angle(A)
p = N[0] 
fi = N[1] 
q1 = axis_angle_2_q(p,fi)

A = euler_2_a(e_angle2[0], e_angle2[1], e_angle2[2])
N = axis_angle(A)
p = N[0] 
fi = N[1]
q2 = axis_angle_2_q(p,fi)


# iscrtavanje 3d grafika
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1], projection='3d')
ax.set_xlim((0, 10))
ax.set_ylim((0, 10))
ax.set_zlim((0, 10))

ax.view_init(10, 0)

colors = ['#b3b3ff', '#ff9999', '#00ffcc']

lines = np.array(sum([ax.plot([], [], [], c=c) for c in colors], []))

# zadavanje pocetnog polozaja
startpoints = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
endpoints = np.array([[3, 0, 0], [0, 3, 0], [0, 0, 3]])

for i in range(3):

    # rotacija
	start = transform(startpoints[i], q1)
	end = transform(endpoints[i], q1)

    # translacija
	start += pos1
	end += pos1

	ax.plot([start[0],end[0]], [start[1],end[1]],zs=[start[2],end[2]], color=colors[i])

# zadavanje krajnjeg polozaja
startpoints = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
endpoints = np.array([[3, 0, 0], [0, 3, 0], [0, 0, 3]])

for i in range(3):

    # rotacija
	start = transform(startpoints[i], q2)
	end = transform(endpoints[i], q2)

    # translacija
	start += pos2
	end += pos2

	ax.plot([start[0],end[0]], [start[1],end[1]],zs=[start[2],end[2]], color=colors[i])


# funkcija init za animaciju
def init():
	for line in lines:
		line.set_data(np.array([]), np.array([]))
		line.set_3d_properties(np.array([]))

	return lines

# za svaki frejm se poziva slerp algoritam koji vraca kvaternion
# taj kvaternion se koristi za rotaciju
def animate(i):

	q = slerp(np.array(q1), np.array(q2), tm,i)
	k = i *(pos2-pos1)/tm

	for line, start, end in zip(lines, startpoints, endpoints):

        # rotacija za q
		start = transform(np.array(start), np.array(q))
		end = transform(np.array(end), np.array(q))

        # translacija za k
		start += pos1 + k
		end += pos1 + k

		line.set_data(np.array([start[0], end[0]]), np.array([start[1], end[1]]))
		line.set_3d_properties(np.array([start[2], end[2]]))

	fig.canvas.draw()
	return lines

anim = animation.FuncAnimation(fig, frames=tm, func=animate, init_func=init, interval=5, blit=True)
anim.save('animation.gif')