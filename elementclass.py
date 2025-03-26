import numpy as np
import matplotlib.pyplot as plt

globalParameters = {'nDoF': 3} 
class Element:
    def __init__(self, node_i, node_j, E, I, A, dx=0, dy=0, printSummary=True):
        """
        Elemento viga-columna entre dos nodos en 2D (6 grados de libertad).
        :param node_i: Nodo inicial
        :param node_j: Nodo final
        :param E: Módulo de elasticidad
        :param I: Momento de inercia
        :param A: Área de la sección transversal
        """
        self.dx1=dx
        self.dy1=dy
        self.dx2=-dx
        self.dy2=dy
        self.node_i = node_i
        self.node_j = node_j
        self.E = E
        self.I = I
        self.A = A

        # Longitud del elemento
        self.length = np.linalg.norm(self.node_j.coordenadas - self.node_i.coordenadas)

        # Cálculo de coseno y seno de la dirección del elemento
        delta_x = self.node_j.coordenadas[0] - self.node_i.coordenadas[0]
        delta_y = self.node_j.coordenadas[1] - self.node_i.coordenadas[1]
        self.cos_theta = delta_x / self.length
        self.sin_theta = delta_y / self.length

        # Calcular matriz de rigidez local
        self.k_local = self._stiffness_matrix_local()

        # Transformar a coordenadas globales
        self.k_trans = self._transform_to_global()

        self.k_global = self._rigidoffset()

        #if printSummary:
        #    self.printSummary()

    def _stiffness_matrix_local(self):
        """
        Matriz de rigidez en coordenadas locales (sistema del elemento).
        """
        L = self.length
        E = self.E
        I = self.I
        A = self.A

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
        """
        Convierte la matriz de rigidez local a coordenadas globales.
        """
        c = self.cos_theta
        s = self.sin_theta

        # Matriz de transformación
        T = np.array([
            [ c, s,  0,  0,  0,  0],
            [-s, c,  0,  0,  0,  0],
            [ 0, 0,  1,  0,  0,  0],
            [ 0, 0,  0,  c,  s,  0],
            [ 0, 0,  0, -s,  c,  0],
            [ 0, 0,  0,  0,  0,  1]
        ])

        # Transformar la matriz de rigidez local
        k_trans = T.T @ self.k_local @ T
        return k_trans
    
    def _rigidoffset(self):
        dx1, dy1 = self.dx1, self.dy1
        dx2, dy2 = self.dx2, self.dy2
        T=np.array([
            [1,0,-dy1,0,0,0],
            [0,1, dx1,0,0,0],
            [0,0,1,0,0,0],
            [0,0,0,1,0,-dy2],
            [0,0,0,0,1,dx2],
            [0,0,0,0,0,1]
        ])

        k_global = T.T @ self.k_trans @ T
        return k_global

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
            # Ubicar texto al centro del elemento
            xc, yc = (xi + xj) / 2, (yi + yj) / 2
            #ax.text(xc, yc, f'E{self.node_i.name}-{self.node_j.name}', fontsize=10, color=color)

        return ax

    def printSummary(self):
        print(f'--------------------------------------------')
        print(f"Elemento entre Nodos {self.node_i.name} y {self.node_j.name}")
        print(f"Longitud: {self.length:.2f}")
        print(f"Dirección: cos(θ)={self.cos_theta:.2f}, sin(θ)={self.sin_theta:.2f}")
        print(f"Matriz de Rigidez Global:\n{self.k_global}")
        print(f"Área: {self.A:.2f}, Inercia: {self.I:.2f}, Módulo de Elasticidad: {self.E:.2f}")
        print(f'--------------------------------------------\n')