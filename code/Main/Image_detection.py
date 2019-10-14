def make_matrix_voor_annelies(iar,h,w):

    iar_bool_to_bin = [0 if i == False else 1 for i in iar]

    matrix = []
    index2 = 0
    for i in range(h):
        matrix.append([])
        for j in range(w):
            k = iar_bool_to_bin[index2]
            matrix[i].append(k)
            index2+=1

    return matrix,len(matrix[0]),len(matrix)




