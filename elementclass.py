import numpy as np
import matplotlib.pyplot as plt

class Element:
    def __init__(self, node_i, node_j, E, I, A, dx=0, dy=0, printSummary=True):
        # Desplazamientos rígidos para ambos nodos
        self.dx1 = dx
        self.dy1 = dy
        self.dx2 = -dx  # Desplazamiento negativo para el nodo j
        self.dy2 = dy

        # Nodos del elemento
        self.node_i = node_i
        self.node_j = node_j
        self.nodal_loads = np.hstack([node_i.nodalLoad, node_j.nodalLoad])
        # Propiedades del material y la sección transversal
        self.E = E
        self.I = I
        self.A = A

        # Cálculo de la longitud y ángulo de la viga
        self.length = np.linalg.norm(self.node_j.coordenadas - self.node_i.coordenadas)
        delta_x = self.node_j.coordenadas[0] - self.node_i.coordenadas[0]
        delta_y = self.node_j.coordenadas[1] - self.node_i.coordenadas[1]
        self.cos_theta = delta_x / self.length
        self.sin_theta = delta_y / self.length

        # Calculamos la matriz de rotación (R) en el constructor
        self.R = self._get_rotation_matrix()

        # Matrices de rigidez y transformaciones
        self.k_local = self._stiffness_matrix_local()
        self.k_trans = self._transform_to_global()
        self.k_global = self._rigidoffset()

        #if printSummary:
        #    self.print_summary()
    def calculate_forces(self):
        """
        Calcula las fuerzas de corte, axial y momento en los nodos del elemento
        usando las cargas nodales de los nodos i y j.
        """

        # Carga nodal en cada nodo (nodalLoad de los nodos i y j)
        load_i = self.node_i.nodalLoad
        load_j = self.node_j.nodalLoad

        # Determinar las coordenadas de los nodos
        xi, yi = self.node_i.coordenadas
        xf, yf = self.node_j.coordenadas

        # Comprobamos si el elemento es horizontal o vertical
        if xi == xf:
            # Elemento vertical
            V1 = load_i[0]  # Fuerza cortante en el nodo i
            V2 = -load_j[0]  # Fuerza cortante en el nodo j
            A1 = load_i[1]  # Fuerza axial en el nodo i (no suele aplicarse en elementos verticales, por lo general es 0)
            A2 = load_j[1]  # Fuerza axial en el nodo j
            M1 = load_i[2]  # Momento en el nodo i
            M2 = load_j[2]  # Momento en el nodo j
            
        else:
            # Elemento horizontal
            A1 = load_i[0]  # Fuerza axial en el nodo i
            A2 = load_j[0]  # Fuerza axial en el nodo j
            V1 = load_i[1]  # Fuerza cortante en el nodo i
            V2 = -load_j[1]  # Fuerza cortante en el nodo j
            M1 = load_i[2]  # Momento en el nodo i
            M2 = load_j[2]  # Momento en el nodo j

        # Crear una malla de puntos a lo largo de la viga
        x = np.linspace(0, self.length, 100)

        # Interpolación de la fuerza cortante (V)
        V = V1 + (V2 - V1) * x / self.length # Interpolación lineal de la fuerza cortante

        # Calcular el momento integrando la fuerza cortante
        M = -np.cumsum(V) * (x[1] - x[0])  # Aproximación de la integral del esfuerzo cortante
        M -= M[0]  # Ajustar para que M(x=0) = 0

        # Sumamos M1 y M2 al momento en los extremos
        M += M1  # Añadir el momento en el primer nodo

        # Escalar los diagramas para visualización
        escala = 0.01  # Este factor ajusta la escala del diagrama
        V_scaled = V * escala
        M_scaled = M * escala
        A = A1  # Asumiendo fuerza axial constante
        A_scaled = np.full_like(x, A * 0.1)

        return V_scaled, M_scaled, A_scaled



    def _get_rotation_matrix(self):
        # Calcular la matriz de rotación R para el ángulo de la viga
        c = self.cos_theta
        s = self.sin_theta
        R = np.array([
            [ c, s],
            [-s, c]
        ])
        return R

    def _stiffness_matrix_local(self):
        L = self.length
        E = self.E
        I = self.I
        A = self.A

        # Matriz de rigidez local para el elemento (viga-columna)
        k = np.array([
            [E*A/L,     0,            0,      -E*A/L,    0,            0],
            [0,       12*E*I/L**3,   6*E*I/L**2,  0, -12*E*I/L**3,  6*E*I/L**2],
            [0,        6*E*I/L**2,   4*E*I/L,    0, -6*E*I/L**2,   2*E*I/L],
            [-E*A/L,   0,            0,       E*A/L,    0,            0],
            [0,      -12*E*I/L**3,  -6*E*I/L**2,  0, 12*E*I/L**3,  -6*E*I/L**2],
            [0,        6*E*I/L**2,   2*E*I/L,    0, -6*E*I/L**2,   4*E*I/L]
        ])
        
        return k

    def _transform_to_global(self):
        # Realiza la transformación local a global
        c = self.cos_theta
        s = self.sin_theta

        T = np.array([
            [ c, s,  0,  0,  0,  0],
            [-s, c,  0,  0,  0,  0],
            [ 0, 0,  1,  0,  0,  0],
            [ 0, 0,  0,  c,  s,  0],
            [ 0, 0,  0, -s,  c,  0],
            [ 0, 0,  0,  0,  0,  1]
        ])

        k_trans = T.T @ self.k_local @ T
        self.T = T  # Guardamos T para uso posterior
        return k_trans

    def _rigidoffset(self):
        # Desplazamientos rígidos locales para cada nodo
        dx1, dy1 = self.dx1, self.dy1
        dx2, dy2 = self.dx2, self.dy2

        # Transformación de desplazamientos rígidos locales a globales usando la matriz de rotación R
        offset_i_local = np.array([dx1, dy1])  # Nodo i
        offset_j_local = np.array([dx2, dy2])  # Nodo j

        # Aplicamos la rotación con la matriz de rotación R
        offset_i_global = self.R @ offset_i_local  # Nodo i
        offset_j_global = self.R @ offset_j_local  # Nodo j

        # Construcción de la matriz de transformación para los desplazamientos rígidos
        T_rigid = np.array([
            [1, 0, -offset_i_global[1], 0, 0,   0],
            [0, 1,  offset_i_global[0], 0, 0,   0],
            [0, 0,    1, 0, 0,   0],
            [0, 0,    0, 1, 0, -offset_j_global[1]],
            [0, 0,    0, 0, 1,  offset_j_global[0]],
            [0, 0,    0, 0, 0,   1]
        ])

        return T_rigid.T @ self.k_trans @ T_rigid
    def extractForces(self, desplazamientos):
        """
        Calcula las fuerzas internas (cortantes y momentos) a partir de los desplazamientos de los nodos
        """
        dofs = np.hstack([self.node_i.idx, self.node_j.idx])
        desplazamientos_local = desplazamientos[dofs]  # Desplazamientos en los nodos locales
        
        # Las fuerzas internas son K_local * desplazamientos_local
        self.f_local = np.dot(self.k_local, desplazamientos_local)
        return self.f_local
    def plotElement(self, ax=None, color='b', text=False):
        if ax is None:
            fig, ax = plt.subplots()

        xi, yi = self.node_i.coordenadas
        xj, yj = self.node_j.coordenadas
        ax.plot([xi, xj], [yi, yj], color + '-', linewidth=2)

        if text:
            xc, yc = (xi + xj) / 2, (yi + yj) / 2
            ax.text(xc, yc, f'E{self.node_i.name}-{self.node_j.name}', fontsize=10, color=color)

        return ax

    def print_summary(self):
        print(f"\nNodo inicial: {self.node_i.coordenadas}")
        print(f"Nodo final: {self.node_j.coordenadas}")
        print(f"Longitud: {self.length}")
        print(f"Matriz de rigidez local (k_local): \n{self.k_local}")
        print(f"Matriz de rigidez global transformada (k_trans): \n{self.k_trans}")
        print(f"Matriz de rigidez global con desplazamientos rígidos (k_global): \n{self.k_global}")
