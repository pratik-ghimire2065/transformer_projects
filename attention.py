import numpy as np
def create_qkv(x,w_q,w_k,w_v):
    q=np.dot(x,w_q)
    k=np.dot(x,w_k)
    v=np.dot(x,w_v)
    return q , k, v
if __name__ =="__main__":
    x=np.array([
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 2.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 2.0]
    ])
    w_q=np.array([
       [1.0,0.0],
       [0.0,1.0],
       [1.0,0.0],
       [0.0,1.0] 
    ])
    #for now we are creating same as for the finding reasoning so we create same for temporary
    w_k=w_q
    w_v=w_q
    q,k,v=create_qkv(x,w_q,w_k,w_v)
    print("Q is ",q)
    print("K is ",k) 
    print("V is ",v)
    