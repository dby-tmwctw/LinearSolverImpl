import numpy as np
from helper import *
import numpy.linalg as LA
from  matplotlib import *
import matplotlib.pyplot as plt

"""
For problem of the form 0.5 * ||Ax-b||^2

"""

cached_vals = {}

def func(C, D, B, x):
    #A = C.dot(D)
    return 0.5 * LA.norm(C.dot(D).dot(x)-B, ord=2) ** 2
def admm_step():
    """
    x, w, v are updated
    """
    pass

def w_step(w_k, x_k, v_k, C, D, B, rho):
    """
    Primal, w = D * x
    """

    if "w_step" not in cached_vals:
        I = np.identity(C.shape[1])
        cached_vals["w_step"] = LA.inv(C.transpose().dot(C) + rho * I)

    return cached_vals["w_step"].dot(v_k + rho * D.dot(x_k) + C.transpose().dot(B))

def x_step(w_k, x_k, v_k, C, D, rho):

    """
    x is what we're solving for
    """

    if "DTD" not in cached_vals:
        cached_vals["DTD"] = D.transpose().dot(D)

    temp = (-1 * v_k.transpose().dot(D) + rho * cached_vals["DTD"])
    return LA.inv(temp).dot(rho * D.transpose().dot(w_k))

def y_step(w_k, x_k, v_k, C, D, rho):

    """
    Dual, y regulates Dx = w
    """

    return v_k + rho * (w_k - D.dot(x_k))

def admm_mat(C, D, B, x_solve, iters):
    """
    ADMM matrix version.
    """
    #penalty factor
    #print("rho: {rho}".format(rho))
    rho = 1.0e-50
    print("rho: {rho}".format(rho=rho))

    w_k = np.zeros(x_solve.shape)
    x_k = np.zeros(x_solve.shape)
    v_k = np.zeros(x_solve.shape)

    #

    i = []
    val = []
    i.append(0)
    val.append(func(C, D, B, x_k))
    # print("Curr Val @ iter zero: {t}".format(t = func(C, D, B, x_k)))
    # print("starting")
    for iter in range(iters):

        #argmin w
        w_k = w_step(w_k,x_k, v_k, C, D, B, rho)

        #argmin x
        x_k = x_step(w_k, x_k, v_k, C, D, rho)

        #argmin v
        v_k = y_step(w_k, x_k, v_k, C, D, rho)


        if ((iter+1) % 1 == 0):
            temp = func(C, D, B, x_k)
            #print("Curr Val @ iter {i}: {t}".format(i = iter, t = temp))

            val.append(temp)
            i.append((iter+1))
        # if func(C, D, B, x_k) < 10:
        #     print("Better x_k {}".format(x_k))
        if (iter+1) == iters:
            print("Final Value: {}".format(func(C, D, B, x_k)))
    print(i)
    print(val)
    plt.plot(i, val, linewidth=2.0)
    plt.xlabel("iterations")
    plt.ylabel('val')
    # plt.show()




def mission_control():
    iter = 5
    shape = (1000, 1000)
    print("~~~~~~~~ Params below")
    print("iters: {}".format(iter))
    print("Size: {}".format(shape[1]))

    C = np.random.random(shape)
    D = np.random.random(shape)
    #A = C.dot(D)

    x_solve = np.zeros((shape[0], 1))

    x = np.random.random((shape[0], 1))
    # print("Correct x_solve: ", x)
    B = C.dot(D).dot(x)


    x = admm_mat(C, D, B, x_solve, iter)



mission_control()
