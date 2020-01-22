# -*- coding: utf-8 -*-
import numpy as np
##################################################################
#Affine層為X跟W內積加B
class Affine:
    def __init__(self, W, b):
        self.W =W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx
##################################################################
#活化函數
#RELU函數
class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx
#小於0為0,大於0為本身
#sigmoid函數
def sigmoid(x):
    return 1 / (1 + np.exp(-x))  
class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx 
#階梯函數
def step_function(x):
    return np.array(x > 0, dtype=np.int)
#小於0為0,大於0為本身

#神經網路都選用非線性活化函數，因為線性函數只會活化一次，這樣多層網路就沒效果了
#又以RELU最常被使用

##################################################################
#輸出層
#恆等函數
def identity_function(x):
    return x
#歸一化指數函數(softmax)
#第一個為只有一維可以用，第二個為批次處理可以使用
def softmax(a):
    c=np.max(a)#防止溢位p63
    exp_a=np.exp(a-c)
    sum_exp_a=np.sum(exp_a)
    y=exp_a/sum_exp_a
    
    return y

def high_softmax(a):
    #先看輸出為單一，還是批次處理
    if a.ndim==1:
        c=np.max(a)#防止溢位p63
        exp_a=np.exp(a-c)
        sum_exp_a=np.sum(exp_a)
        final=exp_a/sum_exp_a
    else:
        final=np.zeros(a.shape[1])
        for i in range(len(a)):
            c=np.max(a[i])
            exp_a=np.exp(a[i]-c)
            sum_exp_a=np.sum(exp_a)
            y=exp_a/sum_exp_a
            final=np.vstack([final,y])#橫向為一組，然後每組以直的方向併
    final=np.delete(final,0,axis=0)#把起始第一列刪除
    return final
####################################################################
#loss損失函數
def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)
def some_mean_squared_error(y, t):
    if t.size == y.size:
        tt=t
    else:#如果數量不一樣則代表t為數字，所以要轉換成onehot
        from sklearn.preprocessing import OneHotEncoder
        tt=onehot_encoder.fit_transform(t.reshape(len(t), 1))
    return np.mean(np.sum((y-tt)**2,axis=1)*0.5)


#把數字標籤轉成one-hot
#from sklearn.preprocessing import OneHotEncoder
#tt=onehot_encoder.fit_transform(t.reshape(len(t), 1))


#y為預測機率，t為真實y，此程式不管t為one-hot or數字皆可以用
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    
    if t.size == y.size:#如果數量一樣則代表t為onehot
        t = t.argmax(axis=1)#接換成數字標籤
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
#y[np.arange(batch_size), t]這邊是因為只需取出t答案的y位置t答案的機率

#############################################################################
#損失函數結合輸出層
#softwithloss
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None # softmaxの出力
        self.real_y = None # 教師データ

    def forward(self, x, real_y):
        self.real_y = real_y
        self.y = high_softmax(x)
        self.loss = cross_entropy_error(self.y, self.real_y)
        
        return self.loss

    def backward(self, dout=1):
        batch_size = self.real_y.shape[0]
        if self.real_y.size == self.y.size: 
            dx = (self.y - self.real_y) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.real_y] -= 1
            dx = dx / batch_size
        
        return dx
#############################################################################
#數值微分求梯度(爆破法)
def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    x=x.flatten()#將所有參數拉平，迭代
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val#恢復預設
        #一定要恢復不然會影響後面下降法的x
    return grad
#############################################################################
#梯度下降法
def gradient_descent(f, init_x, lr, step_num):
    x = init_x
    x_history = []

    for i in range(step_num):
        x_history.append( x.copy() )#把每次更新的x加到紀錄後面，append為把東西加到列表後面

        grad = numerical_gradient(f, x)
        x -= lr * grad#V新=V舊-ALPHA*梯度，lr學習率

    return x, np.array(x_history)
#############################################################################
#參數更新方法
class SGD:
    #我假設學習率為0.01
    def __init__(self, lr=0.01):
        self.lr = lr
    #.keys可以抓出param裡面的表單名稱，w、b
    def update(self, param, grad):
        for key in param.keys():
            param[key] -= self.lr * grad[key]
#數學更新手法
#w新=w舊-lr*w新

class Momentum:

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
        
    def update(self, param, grad):
        #先將參數v預設好，以零為初始值
        if self.v is None:
            self.v = {}
            for key, val in param.items(): 
                #params.items()會將名稱和數值分開迭代                               
                self.v[key] = np.zeros_like(val)
                
        for key in param.keys():
            self.v[key] = self.momentum*self.v[key] - self.lr*grad[key] 
            param[key] += self.v[key]
#數學更新手法
#v=momentum*v-lr*grad
#w新=w舊+v
class AdaGrad:

    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None
        
    def update(self, param, grad):
        if self.h is None:
            self.h = {}
            for key, val in param.items():
                self.h[key] = np.zeros_like(val)
            
        for key in param.keys():
            self.h[key] += grad[key] * grad[key]
            param[key] -= self.lr * grad[key] / (np.sqrt(self.h[key]) + 1e-7)
#數學更新手法
#h=h+grad*grad
#w=w-lr*[1/sqrt(h)]*grad





