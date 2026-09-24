import numpy as np
def create_qkv(x,w_q,w_k,w_v):
    q=np.dot(x,w_q)
    k=np.dot(x,w_k)
    v=np.dot(x,w_v)
    return q , k, v
def attention_score(q,k):
    score=np.dot(q,k.T)
    return score
def scaled_attention_score(score,d_k):
    scaled_attention=score/np.sqrt(d_k)
    return scaled_attention
def softmax(x):
    x=x-np.max(x,axis=1,keepdims=True)
    x_exp=np.exp(x)
    return x_exp/np.sum(x_exp,axis=1,keepdims=True)
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
    score=attention_score(q,k)
    d_k=k.shape[-1]
    softmaxs=softmax(score)
    print("Softmax is ",softmaxs)
    
    scaled_score=scaled_attention_score(score,d_k)
    print("Scaled attention Score is ",scaled_score)
    print("Score is ",score)
    print("Q is ",q)
    print("K is ",k) 
    print("V is ",v)