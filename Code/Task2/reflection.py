import cv2
import numpy as np
from scipy import sparse
from scipy.fftpack import dct, idct
import pywt
np.seterr(divide='ignore', invalid='ignore')

def normalize(arr):
    OldMax = np.amax(arr)
    OldMin = np.amin(arr)
    NewMax=1
    NewMin=0
    norm_arr = []
    OldRange = (OldMax - OldMin)
    NewRange = (NewMax - NewMin)
    # NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
    for i in arr:
        buff1=[]
        for j in i:
            buff=[]
            for k in j:
                temp = (((k- OldMin)* NewRange)/ OldRange)+ NewMin
                buff.append(temp)
            # print(buff)
            buff1.append(buff)
        norm_arr.append(buff1)

    return np.array(norm_arr)
def main():
    input= cv2.imread("toy_example.jpg")
    min_width=min(input.shape[:1])
    # cv2.imshow("in1",input)
    input=cv2.resize(input,(min_width,min_width))
    # cv2.imshow("in2",input)
    # cv2.waitKey(0)

    output= reflectSuppress(input, 0.033, 1e-8)
    # output =normalize(output)
    print("Output",output.shape,output)
    # norm_img = np.zeros((800, 800))

    output = cv2.normalize(output, 0, 255, cv2.NORM_MINMAX)

    cv2.imshow("Output", output)
    cv2.imshow("Outpddut", input)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # return 0
def reflectSuppress(Im, h, epsilon):
    Y= np.around(cv2.normalize(Im.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX),4)
    # print(Y)

    [m, n, r]= np.shape(Y)
    T= np.zeros((m, n, r))
    Y_Laplacian_2= np.zeros((m, n, r))
    cv2.imshow("Outwput", Im)
    cv2.waitKey(0)
    for dim in range(r):
        GRAD= grad(Y[:,:, dim])
        GRAD_x= GRAD[:,:, 0]
        GRAD_y= GRAD[:,:, 1]
        GRAD_norm= np.sqrt((GRAD_x ** 2) + (GRAD_y ** 2))
        GRAD_norm_thresh= pywt.threshold(GRAD_norm, h, 'hard')
        print('hard',GRAD_norm_thresh)
        for i in range(len(GRAD_norm_thresh)):
            for j in range(len(GRAD_norm_thresh[i])):
                if GRAD_norm_thresh[i][j]== 0:
                    GRAD_x[i][j]= 0
                    GRAD_y[i][j]= 0
        GRAD_thresh= np.zeros((m, n, 3))
        GRAD_thresh[:,:, 0]= GRAD_x
        GRAD_thresh[:,:, 1]= GRAD_y
        Y_Laplacian_2[:,:, dim]= div(grad(div(GRAD_thresh)))
    rhs = Y_Laplacian_2 + epsilon * Y
    for dim in range (r):
        T[:,:, dim]= PoissonDCT_variant(rhs[:,:, dim], 1, 0, epsilon)
    print(T.shape)
    return T
def PoissonDCT_variant(rhs, mu, Lambda, epsilon):
        [M,N]=rhs.shape
        print(rhs.shape)
        k= np.arange(0, M)
        k= k.reshape(1, M)
        l= np.arange(1, N+1)
        l= l.reshape(1, N)
        k=k.conjugate()
        k=np.transpose(k)
        eN=np.ones((1,N))
        eM=np.ones((M,1))
        k= np.nan_to_num(np.cos(np.pi/M*(k-1)))
        print(k)
        l= np.nan_to_num(np.cos(np.pi/(N*(l-1))))

        print(k.shape,l.shape)
        k= np.kron(k, eN)
        l= np.kron(l, eM)
        print(k.shape,l.shape)
        kappa=2*(k+l-2)
        const= mu * (kappa ** 2) - Lambda * kappa + epsilon
        u=dct2(rhs)
        # u=u/const
        # for i in range(u.shape[0]):
        #     for j in range(u.shape[1]):
        #         u[i][j]=u[i][j]/const[i][j]
        u= idct2(u)
        print(const,"const")
        print(u,"u")
        return u
# 2D DST

def dct2(y):
    return dct(dct(y.T).T)


######################################################################
# 2D inverse DCT

def idct2(b):
    return idct(idct(b.T).T)

import numpy as np

def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]
def grad(A):
        [m,n]=np.shape(A)
        B= np.zeros((m, n, 2))
        Ar= np.zeros((m, n))
        Ar[:, 0: n -1]=A[:, 1: n]
        Ar[:,n-1]= A[:,n-1]
        Au= np.zeros((m, n))
        Au[0: m -1 ,:]=A[1: m,:]
        Au[m-1,:]=A[m-1,:]
        B[:,:, 0]=Ar -A
        B[:,:, 1]=Au -A
        return B
def div(A):
        [m, n, r]= np.shape(A)
        B= np.zeros((m, n))
        T= A[:, :, 0]
        T1= np.zeros((m, n))
        T1[:, 1: n]=T[:, 0: n -1]
        B= B + T -T1
        T= A[:, :, 1]
        T1= np.zeros((m, n))
        T1[1: m,:]=T[0: m -1,:]
        B= B + T -T1
        return B
def im2double(im):
    info= np.iinfo(im.dtype)
    return im.astype(np.float) /info.max

if __name__== "__main__":
    main()