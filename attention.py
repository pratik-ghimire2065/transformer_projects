import numpy as np
def self_attention(x,w_q,w_k,w_v):
    #creating q k v
    q=np.dot(x,w_q)
    k=np.dot(x,w_k)
    v=np.dot(x,w_v)
    #attention score 
    score=np.dot(q,k.T)
    #scaling 
    q_k=k.shape[-1]
    scaled_score=score/np.sqrt(q_k)
    #applying softmax to the manual transformer 
    x=scaled_score - np.max(scaled_score,axis=1,keepdims=True)
    x_exp=np.exp(x)
    softmax=x_exp/np.sum(x_exp,axis=1,keepdims=True)
    #applying attention weight 
    output=np.dot(softmax,v)
    print("Output of the attention is \n",output)
    return output
w_q = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [0.0, 1.0]
])

w_k = np.array([
    [0.5, 1.0],
    [1.0, 0.5],
    [0.5, 0.5],
    [1.0, 1.0]
])

w_v = np.array([
    [1.0, 0.5],
    [0.5, 1.0],
    [1.0, 1.0],
    [0.5, 0.5]
])
x=np.array([ [1.0, 0.0, 1.0, 0.0], [0.0, 2.0, 0.0, 1.0], [1.0, 1.0, 0.0, 2.0] ])
self_attention(x,w_q,w_k,w_v)