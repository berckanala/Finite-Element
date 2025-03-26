import numpy as np
import matplotlib.pyplot as plt
from elementclass import Element
from nodeclass import Node
from structuresolver import StructureSolver
#-----------------------------------------------------------------------------------------------------------
def plotElement(self, ax=None, color='b', text=False):
    """
    Dibuja el elemento como una línea entre sus nodos.
    """
    if ax is None:
        fig, ax = plt.subplots()

    xi, yi = self.node_i.coordenadas
    xj, yj = self.node_j.coordenadas

    ax.plot([xi, xj], [yi, yj], color + '-', linewidth=2)

    if text:
        # Ubica el texto en el centro del elemento
        xc, yc = (xi + xj) / 2, (yi + yj) / 2
        ax.text(xc, yc, f'E{self.node_i.name}-{self.node_j.name}', fontsize=10, color=color)

    return ax

#---------------------------------------------DATOS INICIALES-----------------------------------------------
globalParameters = {'nDoF': 3}  
fc = 27  # MPa 
E = 4700 * (fc ** 0.5)  # Módulo de elasticidad
A_v = [1.6, 1.6, 1.16, 0.68]
A_ch = [0.36, 0.33, 0.3, 0.24]
A_cc = [5, 4.55, 3.70, 2.83, 2.57]
A_ce = [3.70, 3.70, 2.83, 2.57, 2.33]
Alt_cs = [11.13, 19.05, 26.97, 34.89]

Espaciado = 9.14
a = 3.96 / 2
b = 3.96
Altura = [0, 3.66, 5.49, a, a, b, a, a, b, a, a, b, a, a, b]
ql = 26

# Función para calcular la carga
def calcular_carga(i, alt, ql):
    if i % 2 == 0 and i > 3:
        A = A_v[3] * A_ch[3]
        qd = A * Espaciado * 2.5
        if alt == 40.83:
            ql = 23.1
        q = ql + qd
    elif i == 1:
        A = A_v[0] * A_ch[0]
        qd = A * Espaciado * 2.5
        q = ql + qd
    elif i == 2:
        A = A_v[1] * A_ch[1]
        qd = A * Espaciado * 2.5
        q = ql + qd
    elif i % 2 == 1 and i > 3:
        A = A_v[2] * A_ch[2]
        qd = A * Espaciado * 2.5
        q = ql + qd
    else:
        q = 0
    return q

# Crear nodos
nodos = []
alt = 0

for i, h in enumerate(Altura):
    alt += h
    for j in range(6):
        k = i * 6 + j
        restriccion = ['r', 'r', 'r'] if i == 0 else ['f', 'f', 'f']
        
        q = calcular_carga(i, alt, ql)
        
        if j == 0:
            load = [0, -q * Espaciado / 2, q * Espaciado ** 2 / 12] 
        elif j == 5:
            load = [0, -q * Espaciado / 2, -q * Espaciado ** 2 / 12]
        else:
            load = [0, -q * Espaciado, 0]
        
        nodo = Node(name=k + 1, coordenadas=[j * Espaciado, alt], nodalLoad=load, restrain=restriccion)
        nodos.append(nodo)

# Crear elementos (columnas y vigas)
elementos = []
altidx = 0
cont1 = 0

for i in range(len(Altura) - 1):
    altidx += Altura[i]
    
    for j in range(6):
        k = i * 6 + j
        if any(np.isclose(altidx, val) for val in Alt_cs):
            if j == 0:
                cont1 += 1
        
        # Crear elementos de columna
        if j == 0 or j == 5:
            A1 = 0.14 * A_ce[cont1]
            I1 = 0.14 * (A_ce[cont1]) ** 3 / 12
            velemento = Element(nodos[k], nodos[k + 6], E, I1, A1, dx=0, dy=1)
            elementos.append(velemento)
        else:
            A2 = 0.14 * A_cc[cont1]
            I2 = 0.14 * (A_cc[cont1]) ** 3 / 12
            velemento = Element(nodos[k], nodos[k + 6], E, I2, A2, dx=0, dy=1)
            elementos.append(velemento)
        
        # Crear vigas solo si NO hay cambio de sección
        if j < 5 and i != 0 and not any(np.isclose(altidx, val) for val in Alt_cs):
            if i % 2 == 0 and i > 3:
                A = A_v[3] * A_ch[3]
                I = A_ch[3] * (A_v[3]) ** 3 / 12
            elif i == 1:
                A = A_v[0] * A_ch[0]
                I = A_ch[0] * (A_v[0]) ** 3 / 12
            elif i == 2:
                A = A_v[1] * A_ch[1]
                I = A_ch[1] * (A_v[1]) ** 3 / 12
            elif i % 2 == 1 and i > 3:
                A = A_v[2] * A_ch[2]
                I = A_ch[2] * (A_v[2]) ** 3 / 12
            
            elemento = Element(nodos[k], nodos[k + 1], E, I, A, dx=1, dy=0)
            elementos.append(elemento)

# Conectar el último piso horizontalmente (techo)
for j in range(5):
    k = (len(Altura) - 1) * 6 + j
    A = 0.24 * 0.68
    I = 0.24 * (0.68) ** 3 / 12
    elemento = Element(nodos[k], nodos[k + 1], E, I, A, dx=1, dy=0)
    elementos.append(elemento)





nDoF_total = len(nodos) * globalParameters['nDoF']
K_global = np.zeros((nDoF_total, nDoF_total))

# Agregar cada elemento en la matriz global
for elemento in elementos:
    dofs = np.hstack([elemento.node_i.idx, elemento.node_j.idx])
    for i in range(6):
        for j in range(6):
            K_global[dofs[i], dofs[j]] += elemento.k_global[i, j]


# Imprimir la matriz global
#print("\n🔹 Matriz de Rigidez Global de la Estructura:")
#print(K_global)

solver = StructureSolver(nodos, K_global, globalParameters['nDoF'])

# Obtener resultados
desplazamientos = solver.get_displacements()
reacciones = solver.get_reactions()

# Mostrar resultados
solver.resumen()
for i, nodo in enumerate(nodos):
    # Obtén los desplazamientos en x y y del nodo
    delta_x = desplazamientos[i * 3]  # Suponiendo que desplazamientos[i*3] es el desplazamiento en X
    delta_y = desplazamientos[i * 3 + 1]  # Suponiendo que desplazamientos[i*3+1] es el desplazamiento en Y
    # Actualizamos las coordenadas deformadas
    nodo.coordenadas_deformadas = nodo.coordenadas + np.array([delta_x, delta_y])

# Graficamos la estructura original y deformada
fig, ax = plt.subplots(figsize=(8, 6))

# Graficar los elementos originales
for elem in elementos:
    ax = elem.plotElement(ax=ax, color='b', text=True)

# Graficar los nodos originales
for nodo in nodos:
    ax.plot(nodo.coordenadas[0], nodo.coordenadas[1], 'bo')  # Nodos originales en azul

# Graficar la estructura deformada
for nodo in nodos:
    ax.plot(nodo.coordenadas_deformadas[0], nodo.coordenadas_deformadas[1], 'ro')  # Nodos deformados en rojo

# Dibujar las líneas de conexión entre los nodos deformados
for elem in elementos:
    xi, yi = elem.node_i.coordenadas_deformadas if hasattr(elem.node_i, 'coordenadas_deformadas') else elem.node_i.coordenadas
    xj, yj = elem.node_j.coordenadas_deformadas if hasattr(elem.node_j, 'coordenadas_deformadas') else elem.node_j.coordenadas
    ax.plot([xi, xj], [yi, yj], 'r--', linewidth=1)  # Líneas de conexión entre nodos deformados

ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_title('Deformación de la Estructura')
ax.grid(True)
ax.axis('equal')
plt.show()


