import tensorflow as tf
import numpy as np
import sys

class conv3:
  def __init__(self,numfilter):
    self.numfilter=numfilter
    self.filmat=np.random.randn(3,3,numfilter)/9 #to decrease 
  def forward(self,input):
    l,b=input.shape
    self.chache_input=input
    #paddedinput=zeros(input.shape[0]+2,input.shape[1]+2)
    #paddedinput[1:input.shape[0]+1,1:input.shape[1]+1]+=input
    out=np.zeros((l-2,b-2,8))
    for i in range(l-2):
      for j in range(b-2):
        for f in range(self.numfilter):
          #dl_dfilter[:,:,f]+=dl_dout[i,j,f]*self.chache_input[i:i+3,j:j+3]
          out[i,j,f]=np.sum(input[i:i+3,j:j+3]*self.filmat[:,:,f],axis=(0,1))
    return out
  def backward(self,dl_dout,learning):
    l,b,h=self.filmat.shape
    dl_dfilter=np.zeros((l,b,h))
    for i in range(l-2):
      for j in range(b-2):
        for f in range(h):
          dl_dfilter[:,:,f]+=dl_dout[i,j,f]*self.chache_input[i:i+3,j:j+3]

    
    self.filmat-=learning*dl_dfilter
    return None

class maxpool2:
  def forward(self,input):
    l,b,h=input.shape
    self.chache_input=input
    out=np.zeros((l//2,b//2,h))
    for i in range(l//2):
      for j in range(b//2):
        out[i,j,:]=np.amax(input[2*i:2*i+2,2*j:2*j+2,:],axis=(0,1))
    return out
  def backward(self,gradient):
    l,b,h=self.chache_input.shape
    dl_dinput=np.zeros((l,b,h))
    for i in range(l//2):
      for j in range(b//2):
        for k in range(h):
          amaxi=np.amax(self.chache_input[2*i:2*i+2,2*j:2*j+2,k],axis=(0,1))
          if self.chache_input[2*i,2*j,k]==amaxi :
            dl_dinput[2*i,2*j,k]=gradient[i,j,k]
          elif self.chache_input[2*i+1,2*j,k]==amaxi :
            dl_dinput[2*i+1,2*j,k]=gradient[i,j,k]
          elif self.chache_input[2*i,2*j+1,k]==amaxi :
            dl_dinput[2*i,2*j+1,k]=gradient[i,j,k]
          elif self.chache_input[2*i+1,2*j+1,k]==amaxi :
            dl_dinput[2*i+1,2*j+1,k]=gradient[i,j,k] 
    return dl_dinput

class Softmax:
  def __init__(self,input_length,nodes):
    self.weights=np.random.randn(input_length,nodes)/input_length     #1
    self.biases = np.zeros(nodes)
  def forward(self,input):
    self.chache_shape=input.shape
    input=input.flatten()
    self.chache_input=input
    input_len, nodes = self.weights.shape
    total=np.dot(input,self.weights)+self.biases
    total=total.astype(np.float128)
    # self.chache_total=tot
    #total=tota/np.amax(tota)
    ex=np.exp(total)
    self.chache_prob=ex/np.sum(ex,axis=0)     #2
    return self.chache_prob
  def backward(self , dl_dout,learning,label): #gradient is dl_dout

    dout_dt=-(self.chache_prob)*self.chache_prob[label]
    dout_dt[label]+=self.chache_prob[label]

    dl_dt= dl_dout * dout_dt   #dl/dt=dl/dout * dout/dt

    #totals=input*weight+bias
    dt_db=1
    dt_dinput=self.weights
    dt_dw=self.chache_input

    #dl/dinput=dl/dt*dt/dinput
    dl_dinput= dt_dinput @ dl_dt
    #dl/dw=dl/dt*dt/dw
    dl_dw=dt_dw[np.newaxis].T @ dl_dt[np.newaxis]
    #dl/db=dl/dt*dt/db (dt/db=1)
    dl_db=dl_dt

    #print(learning)
    self.weights-=learning * dl_dw
    self.biases-=learning * dl_db
    
    return dl_dinput.reshape(self.chache_shape)

# Data initialisation
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print([i.shape for i in (x_train, y_train, x_test, y_test)])

#taking only 2000 examples

x_train=x_train[0:1000]
y_train=y_train[0:1000]
x_test, y_test=x_test[0:1000], y_test[:1000]
numfilter=8
conv = conv3(8)                  #8 layer filters
pool = maxpool2()                  # 26x26x8 -> 13x13x8 ,pool size=2
softmax  = Softmax(13 * 13 * 8, 10) # 13x13x8 -> 10  ,10nodes

def forward(image, label):
  out = conv.forward((image / 255) - 0.5)
  out = pool.forward(out)
  out = softmax.forward(out)
  # Calculate cross-entropy loss and accuracy. np.log() is the natural log.
  loss = -np.log(out[label])
  acc = 1 if np.argmax(out) == label else 0
  return out, loss, acc

def train(image,label,learning):
  lr=learning
  out, loss, acc=forward(image,label) #array with probability , 10*1
  gradient=np.zeros(10)
  if out[label]!=0:
    gradient[label]=-1/out[label]    #3
  gradient=softmax.backward( gradient,learning,label )
  gradient=pool.backward(gradient)
  gradient=conv.backward(gradient,lr)
  
  return loss,acc

print('MNIST CNN initialized!')

loss = 0
num_correct = 0
for j in range(1,4):
  print("training round %d"%(j))
  permut=np.random.permutation(len(x_train))
  x_train=x_train[permut]
  y_train=y_train[permut]
  for i, (im, label) in enumerate(zip(x_train, y_train)):
    # Do a forward pass.
    

    # Print stats every 100 steps.
    if i % 100 == 99:
      print(
        '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
        (i + 1, loss / 100, num_correct)
      )
      loss = 0
      num_correct = 0
    
    l, acc = train(im, label,0.005)
    loss += l
    num_correct += acc
loss = 0
num_correct = 0
for i, (im, label) in enumerate(zip(x_test, y_test)):
  # Do a forward pass.
  _, l, acc = forward(im, label)
  loss += l
  num_correct += acc
#num_tests = len(x_test)
print('Test Loss:', loss / 1000)
print('Test Accuracy:', num_correct / 1000)

