import os
import _osx_support
import pickle
"""
 boven = coord[i][0]
 links = coord[i][1]
 rechts = coord[i][2]
 onder = coord[i][3] 
"""


matrix_anneloes = pickle.load(open("matrix_robin_olivia.pkl","rb"))

def hoekpunten_vinden(matrix_anneloes):

    number_of_elements = matrix_anneloes.max()

    coord = [[0, ] * 4 for k in range(0, number_of_elements)]

    for i in range(1,number_of_elements+1):

        for rij in range(len(matrix_anneloes)):
            for kolom in range(len(matrix_anneloes[0])):
                if matrix_anneloes[rij][kolom] == i:

                    if coord[i-1][0] == 0 or (coord[i-1][0])[0] > rij:
                        coord[i-1][0] = (rij,kolom)

                    if coord[i-1][1] == 0 or (coord[i-1][1])[1] < kolom:
                        coord[i-1][1] = (rij,kolom)

                    if coord[i-1][2] == 0 or (coord[i-1][2])[1] > kolom:
                        coord[i-1][2] = (rij, kolom)

                    if coord[i-1][3] == 0 or (coord[i-1][3])[0] < rij:
                        coord[i-1][3] = (rij, kolom)
    return coord

