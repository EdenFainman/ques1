


def calc_av(A, b):
    """
    multiply inverse matrix and vector b
    :param A:matrix
    :param b: b vector
    :return: inverse
    """
    inverseMat = inverse(A)
    av = mul_matrix_wVector(inverseMat, b)
    return av


def identity_matrix(size):
    """
    calculates the identity matrix
    :param size: size of matrix
    :return: the identity matrix
    """
    I = list(range(size))
    for i in range(size):
        I[i] = list(range(size))
        for j in range(size):
            if i == j:
                I[i][j] = 1
            else:
                I[i][j] = 0
    return I


def mul_matrix(m1, m2):
    """

    :param m1: first matrix
    :param m2: second matrix
    :return: the multiply of the matrices
    """
    len_m1 = len(m1)
    cols_m1 = len(m1[0])
    rows_m2 = len(m2)
    if cols_m1 != rows_m2:  # Checks if it is valid to multiply between matrix
        print("Cannot multiply between matrix (incorrect size)")
        return
    new_mat = list(range(len_m1))
    val = 0
    for i in range(len_m1):
        new_mat[i] = list(range(rows_m2))
        for j in range(len(m2[0])):
            for k in range(cols_m1):
                val += m1[i][k] * m2[k][j]
            new_mat[i][j] = val
            val = 0
    return new_mat


def inverse(mat):
    """
    calculates the inverse matrix

    :param mat: matrix
    :return: the inverse matrix
    """
    size = len(mat)
    invert_mat = identity_matrix(size)
    for col in range(size):
        elem_mat = identity_matrix(size)
        max_row = max_val_index(mat, col)  # Returns the index of the row with the maximum value in the column
        invert_mat = mul_matrix(eMatForSwap(size, col, max_row), invert_mat)  # Elementary matrix for swap rows
        mat = mul_matrix(eMatForSwap(size, col, max_row), mat)  # swap between rows in case the pivot is 0
        pivot = mat[col][col]
        for row in range(size):
            if row != col and mat[row][col] != 0:
                elem_mat[row][col] = (-1) * (mat[row][col] / pivot)
        mat = mul_matrix(elem_mat, mat)
        invert_mat = mul_matrix(elem_mat, invert_mat)
    # check diagonal numbers
    for i in range(size):
        pivot = mat[i][i]
        if pivot != 1:
            for col in range(size):
                invert_mat[i][col] /= float(pivot)
            mat[i][i] = 1
    return invert_mat


def mul_matrix_wVector(m, v):
    """
    multiply matrix and vector

    :param m: matrix
    :param v: vector
    :return: the multiplication of matrix and the vector
    """
    len_m = len(m)
    cols_m = len(m[0])
    rows_v = len(v)
    if cols_m != rows_v:  # Checks if it is valid to multiply between matrix
        print("Cannot multiply between matrix (incorrect size)")
        return
    new_mat = list(range(len_m))
    val = 0
    for i in range(len_m):
        for k in range(len(m[0])):
            val += m[i][k] * v[k]
        new_mat[i] = val
        val = 0
    return new_mat


def max_val_index(mat, col):
    """
    find the index with the max value
    :param mat: matrix
    :param col: column
    :return: the index
    """
    max = abs(mat[col][col])
    index = col
    for row in range(col, len(mat)):
        if abs(mat[row][col]) > max:
            max = abs(mat[row][col])
            index = row
    return index


def eMatForSwap(size, index1, index2):
    """
    swaps columns of the matrix
    :param size: matrix size
    :param index1: first index
    :param index2: second index
    :return: the new matrix
    """
    mat = identity_matrix(size)
    # swap rows
    tmp = mat[index1]
    mat[index1] = mat[index2]
    mat[index2] = tmp
    return mat


def polynomial_interpolation(plist, x):
    """
    calculates f(x) by the polynomial interpolation

    :param plist: matrix
    :param x: x value
    :return: the y parameter matches to the given x value
    """
    n = len(plist)
    stringnew=""
    arr=[]
    A = [[0 for i in range(n)] for j in range(n)]
    for i in range(n):
        for j in range(n):
            A[i][j] = plist[i][0] ** j
    b = [0 for i in range(n)]
    for i in range(n):
        b[i] = plist[i][1]
    print(str(b))
    xpow=0
    for i in range(len(b)):
        if(i==0):
            xpow=xpow
        else:
            xpow=xpow+1
        print(("p"+str(i)+"(x)="+str(b[i])+"x^"+str(xpow)))
        stringnew=stringnew+(str(b[i])+"x^"+str(xpow))
        if(i==len(b)-1):
            print(stringnew)
        else:
            stringnew = stringnew + "+"

        #print("p["+str(n-1)+"]="+arr[i]+"+")
    a = calc_av(A, b)
    p = 0

    for i in range(n):
        p = p + a[i] * x ** i
    return p


def neville_interpolation(plist, x):
    """
    calculates f(x) by the neville interpolation

    :param plist: matrix
    :param x: x value
    :return: the y parameter matches to the given x value
    """
    for i in range(len(plist)):
        if x == plist[i][0]:
            return plist[i][1]
    tot = 0
    size = len(plist)
    temp_table = [[point[0], point[1]] for point in plist]
    for j in range(1, size):
        for i in range(size - 1, j - 1, -1):
            temp_table[i][1] = ((x - temp_table[i - j][0]) * temp_table[i][1] - (x - temp_table[i][0]) *
                                temp_table[i - 1][1]) / (temp_table[i][0] - temp_table[i - j][0])
            print("["+str(i)+"]"+"["+str(j)+"]"+"="+str(temp_table[i][1]))

    tot = temp_table[i][1]
    return tot


x = 1.47
x = float(x)
list_inter = [[1.2, 1.5095], [1.3, 1.6984], [1.4, 1.9043], [1.5, 2.1293], [1.6, 2.3756]]  # table for linear interpolation
fx = neville_interpolation(list_inter, x)
print("The value of the point", x, "by neville interpolation method is: %.4f" % fx)

fx = polynomial_interpolation(list_inter, x)
print("The value of the point", x, "by polynomial interpolation method is: %.4f" % fx)

