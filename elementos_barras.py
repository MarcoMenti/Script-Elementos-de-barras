import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import math
from PIL import Image
import os
from IPython.display import HTML
from io import BytesIO
import pandas as pd
import scipy.linalg as sc
from MRPy import MRPy

def vibration_modes(K, M):

# Uses scipy to solve the standard eigenvalue problem
    w2, Phi = sc.eig(K, M)

# Ensure ascending order of eigenvalues
    iw  = w2.argsort()
    w2  = w2[iw]
    Phi = Phi[:,iw]

# Eigenvalues to vibration frequencies
    wk  = np.sqrt(np.real(w2)) 
    fk  = wk/2/np.pi

    return fk, wk, Phi

def matriz_rigidez(nos, elementos, modulos_elasticidade, areas):
    matrizes_rigidez_elementos = []
    
    for i, (no_inicial, no_final) in enumerate(elementos):
        x1, y1 = nos[no_inicial]
        x2, y2 = nos[no_final]
        L = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        c = (x2 - x1) / L
        s = (y2 - y1) / L
        E = modulos_elasticidade[i]
        A = areas[i]
        k = (E * A) / L

        matriz_rigidez_elemento = k * np.array([[c ** 2, c * s, -c ** 2, -c * s],
                                                 [c * s, s ** 2, -c * s, -s ** 2],
                                                 [-c ** 2, -c * s, c ** 2, c * s],
                                                 [-c * s, -s ** 2, c * s, s ** 2]])
        matrizes_rigidez_elementos.append(matriz_rigidez_elemento)
    
    return matrizes_rigidez_elementos

def matriz_rigidez_global(nos, matrizes_rigidez_elementos, elementos):
    n_nos = len(nos)
    matriz_global = np.zeros((2 * n_nos, 2 * n_nos))
    
    for i, (no_inicial, no_final) in enumerate(elementos):
        matriz_rigidez_elemento = matrizes_rigidez_elementos[i]
        indices_global = np.array([2 * no_inicial - 2, 2 * no_inicial - 1, 2 * no_final - 2, 2 * no_final - 1])

        for i_local, i_global in enumerate(indices_global):
            for j_local, j_global in enumerate(indices_global):
                matriz_global[i_global, j_global] += matriz_rigidez_elemento[i_local, j_local]
                
    return matriz_global

def semi_largura_banda(nos, elementos):   # em termos de partições nodais
    # Criar matriz de conectividade a partir dos elementos
    n_nos = len(nos)
    matriz = [[0] * n_nos for _ in range(n_nos)]
    for el in elementos:
        i, j = el
        matriz[i-1][j-1] = matriz[j-1][i-1] = 1
    
    # Calcular semi largura de banda
    max_deslocamento = 0
    for i in range(n_nos):
        for j in range(i+1, n_nos):
            if matriz[i][j]:
                deslocamento = abs(j - i)
                if deslocamento > max_deslocamento:
                    max_deslocamento = deslocamento + 1
    
    return max_deslocamento

# FUNÇÕES PARA AVALIAÇÃO

def exibir_matriz(matriz, tipo='global', elemento=None):
    if tipo == 'global':
        figsize = (10, 10)
    else:
        figsize = (4, 4)
    plt.figure(figsize=figsize)

    ax = sns.heatmap(matriz, annot=True, fmt='.2e', cmap='coolwarm', cbar=False)
    
    n_nos = matriz.shape[0] // 2
    tick_labels = [f'X{i}' if j % 2 == 0 else f'Y{i}' for j, i in enumerate(np.repeat(range(1, n_nos + 1), 2))]
    
    if tipo == 'global':
        ax.set_xticklabels(tick_labels)
        ax.set_yticklabels(tick_labels)
        plt.title(f'Matriz de Rigidez {tipo.capitalize()}')
    else:
        ax.set_xticklabels([f'{j}' for j in range(1, 5)])
        ax.set_yticklabels([f'{j}' for j in range(1, 5)])
        plt.title(f'Matriz de Rigidez {tipo.capitalize()} - Elemento {elemento}')
    if tipo == 'global':
        plt.show()
    else:
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image = Image.open(buffer)
        return image
    
def exibir_matriz_elemento(matrizes_rigidez_elementos, indice_elemento):
    elemento = indice_elemento + 1
    matriz = matrizes_rigidez_elementos[indice_elemento]
    exibir_matriz(matriz, tipo='elemento', elemento=elemento)
    
def vetor_forcas_nodal(nos, forcas):
    vetor_forcas = np.zeros(2 * len(nos))
    for no, forca in forcas.items():
        vetor_forcas[2 * (no - 1)] = forca[0]
        vetor_forcas[2 * (no - 1) + 1] = forca[1]
    return vetor_forcas

def resolver_sistema(matriz_global, vetor_forcas_nodal, restricoes):
    indices_livres = []
    vetor_cc = np.zeros(matriz_global.shape[0])

    for no, restricao in restricoes.items():
        if 'x' in restricao:
            indices_livres.append(2 * (no - 1))
        if 'y' in restricao:
            indices_livres.append(2 * (no - 1) + 1)

    matriz_global_reduzida = np.delete(matriz_global, indices_livres, axis=0)
    matriz_global_reduzida = np.delete(matriz_global_reduzida, indices_livres, axis=1)
    vetor_forcas_nodal_reduzido = np.delete(vetor_forcas_nodal, indices_livres)
    vetor_cc_reduzido = np.linalg.solve(matriz_global_reduzida, vetor_forcas_nodal_reduzido)

    j = 0
    for i in range(vetor_cc.shape[0]):
        if i in indices_livres:
            vetor_cc[i] = 0
        else:
            vetor_cc[i] = vetor_cc_reduzido[j]
            j += 1

    n_nos = len(vetor_cc) // 2
    for no in range(1, n_nos + 1):
        deslocamento_x = vetor_cc[2 * (no - 1)]
        deslocamento_y = vetor_cc[2 * (no - 1) + 1]
        print(f"Nó {no}: Deslocamento em x = {1000*deslocamento_x:.6f}mm, Deslocamento em y = {1000*deslocamento_y:.6f}mm")

    return vetor_cc

def plot_estrutura(nos, elementos, restricoes=None, forcas=None, escala_forca=1):
    plt.figure(figsize=(10, 5))

    for no, (x, y) in nos.items():
        plt.scatter(x, y, c='b')
        plt.text(x + 0.2 , y + 0.1, f'{no}', fontsize=12, color='blue')
        
        if restricoes and no in restricoes:
            restricao = restricoes[no]
            if restricao == 'xy':
                plt.plot([x - 1, x + 1], [y, y], c='g', lw=2)
                plt.plot([x, x], [y - 1, y + 1], c='g', lw=2)
            elif restricao == 'x':
                plt.plot([x - 1, x + 1], [y, y], c='g', lw=2)
            elif restricao == 'y':
                plt.plot([x, x], [y - 1, y + 1], c='g', lw=2)
            elif restricao == 'xyz':
                plt.plot([x - 1, x + 1], [y, y], c='g', lw=2)
                plt.plot([x, x], [y - 1, y + 1], c='g', lw=2)
                plt.scatter(x, y, c='r', marker='o', s=100, facecolors='none', edgecolors='r')

        if forcas and no in forcas:
            fx, fy = forcas[no]
            if fx != 0:
                if fx > 0:
                    plt.arrow(x-1.5, y, escala_forca, 0, color='m', width=0.1, head_width=0.5, head_length=0.5, length_includes_head=True)
                    plt.text(x - 1.5, y + 0.5, f'{fx}', fontsize=0, color='m')
                else:
                    plt.arrow(x, y, escala_forca, 0, color='m', width=0.1, head_width=0.5, head_length=0.5, length_includes_head=True)
                    plt.text(x - 1.5, y + 0.5, f'{-fx}', fontsize=0, color='m')
            if fy != 0:
                if fy > 0:
                    plt.arrow(x, y, 0, escala_forca, color='m', width=0.1, head_width=0.5, head_length=0.5, length_includes_head=True)
                    plt.text(x + 0.5, y + 0.5, f'{fy}', fontsize=0, color='m')
                else:
                    plt.arrow(x, y, 0, -escala_forca, color='m', width=0.1, head_width=0.5, head_length=0.5, length_includes_head=True)
                    plt.text(x + 0.5, y - 0.5, f'{-fy}', fontsize=0, color='m')

    for i, (no_inicial, no_final) in enumerate(elementos, 1):
        x1, y1 = nos[no_inicial]
        x2, y2 = nos[no_final]
        plt.plot([x1, x2], [y1, y2], c='k', lw=2)
        xm, ym = (x1 + x2) / 2, (y1 + y2) / 2
        plt.text(xm + 0.2, ym + 0.1, f'{i}', fontsize=0, color='red')

    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
