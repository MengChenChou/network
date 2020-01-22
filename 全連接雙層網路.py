import numpy as np
from collections import OrderedDict
import matplotlib.pylab as plt
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



















class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std):
        #所有參數初始
        self.param={}
        self.param['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.param['b1'] = np.zeros(hidden_size)
        self.param['W2'] = weight_init_std * np.random.randn(hidden_size, output_size) 
        self.param['b2'] = np.zeros(output_size)
        #總共兩層
        self.layer = OrderedDict()
        self.layer['Affine1'] = Affine(self.param['W1'], self.param['b1'])
        self.layer['Relu1'] = Relu()
        self.layer['Affine2'] = Affine(self.param['W2'], self.param['b2'])
        self.lastLayer = SoftmaxWithLoss()
        
    def predict(self, x):
        #將每層丟入x跟上面參數計算
        for layer in self.layer.values():
            x = layer.forward(x)
        
        return x
    def loss(self, x, real_y):
        #我使用的是LOSS結合輸出層   softmaxwithloss
        y = self.predict(x)
        return self.lastLayer.forward(y, real_y)
    
    def accuracy(self, x, real_y):
        #想看看每次1epach的準確率
        y = self.predict(x)
        #利用argmax轉換比對，ex:y=(0,1,0,0,0,0,0,0,0,0)=>argmax=>1
        y = np.argmax(y, axis=1)
        if real_y.ndim != 1 : real_y = np.argmax(real_y, axis=1)
        accuracy = np.sum(y == real_y) / float(x.shape[0])
        return accuracy
        
    def numerical_gradient(self, x,real_y):
        loss_function = lambda W: self.loss(x,real_y) #lambda宣告w為一個涵式，這樣才可以
        #用之前寫的
        
        grad = {}
        grad['W1'] = numerical_gradient(loss_function, self.param['W1'])
        grad['b1'] = numerical_gradient(loss_function, self.param['b1'])
        grad['W2'] = numerical_gradient(loss_function, self.param['W2'])
        grad['b2'] = numerical_gradient(loss_function, self.param['b2'])
        
        return grad
        
    def gradient(self, x, real_y):
        # forward
        self.loss(x, real_y)
        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        
        layer = list(self.layer.values())
        layer.reverse()
        for layer_iter in layer:
            dout = layer_iter.backward(dout)

        # 設定
        grad = {}
        grad['W1'], grad['b1'] = self.layer['Affine1'].dW, self.layer['Affine1'].db
        grad['W2'], grad['b2'] = self.layer['Affine2'].dW, self.layer['Affine2'].db

        return grad





###########################################################################
#照片資料 MNIST
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# 讀入 MNIST
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
x_train = mnist.train.images
y_train = mnist.train.labels
x_test = mnist.test.images
y_test = mnist.test.labels

# 檢視結構
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
print("---")
t_train=y_train
t_test=y_test
###########################################################################
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10,weight_init_std=0.01)
#共迭代10000次，每次做100個樣本小批次，以這個資料是每550次會看完一次資料
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    #寫了兩種梯度求法
    #grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)
    
    # 更新用隨機梯度下降法SGD
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.param[key] -= learning_rate * grad[key]
    
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)
##############################################################################
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
# 1:実験の設定==========
optimizers = {}
optimizers['SGD'] = SGD()
optimizers['Momentum'] = Momentum()
optimizers['AdaGrad'] = AdaGrad()

###########################################################################
#想比較一下在雙層全連接網路下不同更新手法是否有差
#共迭代10000次，每次做100個樣本小批次，以這個資料是每550次會看完一次資料
network = {}
train_loss = {}
t_train=y_train
t_test=y_test
#將每一種更新方法設定雙層網路
for key in optimizers.keys():
    network[key] =TwoLayerNet(input_size=784, hidden_size=50, output_size=10,weight_init_std=0.01)
    train_loss[key] = []    

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 更新用隨機梯度下降法SGD
    for key in optimizers.keys():
        grad = network[key].gradient(x_batch, t_batch)
        optimizers[key].update(network[key].param, grad)
    
        loss = network[key].loss(x_batch, t_batch)
        train_loss[key].append(loss)
    
    
    if i % 100 == 0:
        print( "===========" + "iteration:" + str(i) + "===========")
        for key in optimizers.keys():
            loss = network[key].loss(x_batch, t_batch)
            print(key + ":" + str(loss))
###########################################################################
#畫平滑曲線視覺化
def smooth_curve(x):
    window_len = 11
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    w = np.kaiser(window_len, 2)
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[5:len(y)-5]
markers = {"SGD": "o", "Momentum": "x", "AdaGrad": "s"}
x = np.arange(iters_num)
for key in optimizers.keys():
    plt.plot(x, smooth_curve(train_loss[key]), marker=markers[key], markevery=100, label=key)
plt.xlabel("iterations")
plt.ylabel("loss")
plt.ylim(0, 1)
plt.legend()
plt.show()
############################################################################
#從上面可以發現更新手法也會影響訓練成效，因為決定方向很重要














#############################################################################
#接著，想要看看初始化參數的差別，分別有"HE","XAVIER"
class TwoLayerNet:

    def __init__(self, input_size, hidden_size_list, output_size, 
                 activation='relu', weight_init_std='he'):
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.param = {}
         # 參數初始化
        self.__init_weight(weight_init_std)
        
        
        #層級的堆疊
        activation_layer = {'sigmoid': Sigmoid, 'relu': Relu}#活化函數常用的兩種
        self.layer = OrderedDict()
        for idx in range(1, self.hidden_layer_num+1):
            #我將隱藏層皆用affine連接
            self.layer['Affine' + str(idx)] = Affine(self.param['W' + str(idx)],
                                                      self.param['b' + str(idx)])
            #將每層的活化函數給Sigmoid或者Relu函數
            self.layer['Activation_function' + str(idx)] = activation_layer[activation]()
        #最後一個隱藏層拉出來寫，因為上面range是1~hidden_layer_num，且最後不用活化
        idx = self.hidden_layer_num + 1
        self.layer['Affine' + str(idx)] = Affine(self.param['W' + str(idx)], self.param['b' + str(idx)])

        self.last_layer = SoftmaxWithLoss()
        
        
        
        
        
    def __init_weight(self, weight_init_std):
        #先取每個層的節點數，然後初始每個參數
        all_node_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for idx in range(1, len(all_node_list)):
            sd = weight_init_std
            if str(weight_init_std).lower() in ('relu', 'he'):#.lower() 把所有字轉小寫
                sd = np.sqrt(2.0 / all_node_list[idx - 1])  #RELU通常使用'HE'
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                sd = np.sqrt(1.0 / all_node_list[idx - 1])  # sigmoid通常使用'XAVIER'
            self.param['W' + str(idx)] = sd * np.random.randn(all_node_list[idx-1], all_node_list[idx])
            self.param['b' + str(idx)] = np.zeros(all_node_list[idx])
    
    
    
    def predict(self, x):
        #將每層丟入x跟上面參數計算
        for layer in self.layer.values():
            x = layer.forward(x)
        
        return x
    def loss(self, x, real_y):
        #我使用的是LOSS結合輸出層   softmaxwithloss
        y = self.predict(x)
        return self.last_layer.forward(y, real_y)
    
    def accuracy(self, x, real_y):
        #想看看每次1epach的準確率
        y = self.predict(x)
        #利用argmax轉換比對，ex:y=(0,1,0,0,0,0,0,0,0,0)=>argmax=>1
        y = np.argmax(y, axis=1)
        if real_y.ndim != 1 : real_y = np.argmax(real_y, axis=1)
        accuracy = np.sum(y == real_y) / float(x.shape[0])
        return accuracy
        
    def gradient(self, x, real_y):
        # forward
        self.loss(x, real_y)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layer.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 設定
        grad = {}
        for idx in range(1, self.hidden_layer_num+2):
            grad['W' + str(idx)] = self.layer['Affine' + str(idx)].dW 
            grad['b' + str(idx)] = self.layer['Affine' + str(idx)].db


        return grad
#######################################################################################
train_size = x_train.shape[0]
batch_size = 100
max_iterations = 6000


# 1:參數設定==========
weight_init_types = {'std=0.01': 0.01, 'Xavier': 'sigmoid', 'He': 'relu'}
optimizer = SGD(lr=0.01)

networks = {}
train_loss = {}
for key, weight_type in weight_init_types.items():#這邊更改不同的初始值##########
    #networks[key]為兩種初始值的model
    networks[key] =TwoLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100],
                                  output_size=10, weight_init_std=weight_type)
    train_loss[key] = []

# 2:訓練迭代==========
for i in range(max_iterations):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    for key in weight_init_types.keys():
        grad = networks[key].gradient(x_batch, t_batch)
        optimizer.update(networks[key].param, grad)
    
        loss = networks[key].accuracy(x_batch, t_batch)#accuracy
        train_loss[key].append(loss)
    
    if i % 100 == 0:
        print("===========" + "iteration:" + str(i) + "===========")
        for key in weight_init_types.keys():
            loss = networks[key].accuracy(x_batch, t_batch)#
            print(key + ":" + str(loss))


# 3.做成表和圖==========
markers = {'std=0.01': 'o', 'Xavier': 's', 'He': 'D'}
x = np.arange(max_iterations)
for key in weight_init_types.keys():
    plt.plot(x, smooth_curve(train_loss[key]), marker=markers[key], markevery=100, label=key)
plt.xlabel("iterations")
plt.ylabel("loss")
plt.ylim(0, 100)#2.5
plt.legend()
plt.show()
############################################################################################

#接著，我要在affine後面接batch norm
#加深網路層數    
    


#######################################################################################
class BatchNormalization:
    #http://arxiv.org/abs/1502.03167 我有參考這篇論文
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        #gamma,beta為參數
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None # 用來判定輸進來的資料 

        # 測試時使用的平均值與方差
        self.running_mean = running_mean
        self.running_var = running_var  
        
        # backward時に使用する中間データ
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        #train_flg是決定是否mu跟var給定，而不是用批次裡的資訊
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)#將資料拉成N張照片，代表一張照片有一列

        out = self.__forward(x, train_flg)
        
        return out.reshape(*self.input_shape)
            
    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)
                        
        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std
            
            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * var            
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))
            
        out = self.gamma * xn + self.beta 
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size
        
        self.dgamma = dgamma
        self.dbeta = dbeta
        
        return dx
#############################################################################
#dropout
class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            #np.random.rand(*x.shape)可以產生與x.shape的隨機機率矩陣
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask


##################################################################################
class Full_Layer_Net:

    def __init__(self, input_size, hidden_size_list, output_size, 
                 activation='relu', weight_init_std='he',use_batchnorm=False,
                 weight_decay_lambda=0, use_dropout = False, dropout_ration = 0.5):
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.param = {}
         # 參數初始化
        self.use_dropout = use_dropout
        self.weight_decay_lambda = weight_decay_lambda
        self.__init_weight(weight_init_std)
        self.use_batchnorm = use_batchnorm
        
        #層級的堆疊
        activation_layer = {'sigmoid': Sigmoid, 'relu': Relu}#活化函數常用的兩種
        self.layer = OrderedDict()
        for idx in range(1, self.hidden_layer_num+1):
            #我將隱藏層皆用affine連接
            self.layer['Affine' + str(idx)] = Affine(self.param['W' + str(idx)],
                                                      self.param['b' + str(idx)])
            
            #batchnorm層
            if self.use_batchnorm:
                self.param['gamma' + str(idx)] = np.ones(hidden_size_list[idx-1])
                self.param['beta' + str(idx)] = np.zeros(hidden_size_list[idx-1])
                self.layer['BatchNorm' + str(idx)] = BatchNormalization(self.param['gamma' + str(idx)], self.param['beta' + str(idx)])
            
            if self.use_dropout:
                self.layer['Dropout' + str(idx)] = Dropout(dropout_ration)
            
            #將每層的活化函數給Sigmoid或者Relu函數
            self.layer['Activation_function' + str(idx)] = activation_layer[activation]()
            
            
        
        #最後一個隱藏層拉出來寫，因為上面range是1~hidden_layer_num，且最後不用活化
        idx = self.hidden_layer_num + 1
        self.layer['Affine' + str(idx)] = Affine(self.param['W' + str(idx)], self.param['b' + str(idx)])

        self.last_layer = SoftmaxWithLoss()
        
        
        
        
        
    def __init_weight(self, weight_init_std):
        #先取每個層的節點數，然後初始每個參數
        all_node_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for idx in range(1, len(all_node_list)):
            sd = weight_init_std
            if str(weight_init_std).lower() in ('relu', 'he'):#.lower() 把所有字轉小寫
                sd = np.sqrt(2.0 / all_node_list[idx - 1])  #RELU通常使用'HE'
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                sd = np.sqrt(1.0 / all_node_list[idx - 1])  # sigmoid通常使用'XAVIER'
            self.param['W' + str(idx)] = sd * np.random.randn(all_node_list[idx-1], all_node_list[idx])
            self.param['b' + str(idx)] = np.zeros(all_node_list[idx])
    
    
    
    def predict(self, x, train_flg=False):
        #將每層丟入x跟上面參數計算
        for key, layer in self.layer.items():
            if "Dropout" in key or "BatchNorm" in key:
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)

        return x
    
    
    def loss(self, x, real_y, train_flg=False):
        #我使用的是LOSS結合輸出層   softmaxwithloss
        y = self.predict(x, train_flg)#
        #weight_decay我這邊使用l2norm:0.5*lambda*w^2
        weight_decay = 0
        for idx in range(1, self.hidden_layer_num + 2):
            W = self.param['W' + str(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W**2)

        return self.last_layer.forward(y, real_y) + weight_decay
    
    def accuracy(self, x, real_y):
        #想看看每次1epach的準確率
        y = self.predict(x)
        #利用argmax轉換比對，ex:y=(0,1,0,0,0,0,0,0,0,0)=>argmax=>1
        y = np.argmax(y, axis=1)
        if real_y.ndim != 1 : real_y = np.argmax(real_y, axis=1)
        accuracy = np.sum(y == real_y) / float(x.shape[0])
        return accuracy
        
    def gradient(self, x, real_y):
        # forward
        self.loss(x, real_y, train_flg=True)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layer.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 設定
        grad = {}
        for idx in range(1, self.hidden_layer_num+2):
            grad['W' + str(idx)] = self.layer['Affine' + str(idx)].dW 
            grad['b' + str(idx)] = self.layer['Affine' + str(idx)].db

            if self.use_batchnorm and idx != self.hidden_layer_num+1:
                grad['gamma' + str(idx)] = self.layer['BatchNorm' + str(idx)].dgamma
                grad['beta' + str(idx)] = self.layer['BatchNorm' + str(idx)].dbeta

        return grad
####################################################################################
#比較做batchnorm的差別
max_epochs = 20
train_size = x_train.shape[0]
batch_size =500
learning_rate = 0.01
def train_function(weight_init_std):
    bn_network = Full_Layer_Net(input_size=784, hidden_size_list=[100, 100, 100, 100, 100], output_size=10, 
                                    weight_init_std=weight_init_std, use_batchnorm=True)
    network = Full_Layer_Net(input_size=784, hidden_size_list=[100, 100, 100, 100, 100], output_size=10,
                                weight_init_std=weight_init_std)
    optimizer = SGD(lr=learning_rate)
    
    train_acc_list = []
    bn_train_acc_list = []
    
    iter_per_epoch =max(train_size / batch_size, 1)
    epoch_cnt = 0
    
    for ii in range(1000000000):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
    
        #for _network in (bn_network, network):
        #    grad = _network.gradient(x_batch, t_batch)
        #    optimizer.update(_network.param, grad)
        grad1 = bn_network.gradient(x_batch, t_batch)
        optimizer.update(bn_network.param, grad1)
        grad2 = network.gradient(x_batch, t_batch)
        optimizer.update(network.param, grad2)
        
        if ii % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            bn_train_acc = bn_network.accuracy(x_train, t_train)
            train_acc_list.append(train_acc)
            bn_train_acc_list.append(bn_train_acc)
    
            print("epoch:" + str(epoch_cnt) + " | " + str(train_acc) + " - " + str(bn_train_acc))
    
            epoch_cnt += 1
            if epoch_cnt >= max_epochs:
                break   
    return train_acc_list, bn_train_acc_list   
    
  
########################################################################################
#下面是比較不同預設初始值及是否有用batch normalization
weight_scale_list=np.concatenate([np.array(['sigmoid', 'relu']),np.logspace(0, -4, num=14)])
x = np.arange(max_epochs)
train_list=[]
bn_train_list=[]
for i, w in enumerate(weight_scale_list):
        print( "============== " + str(i+1) + "/16" + " ==============")
        if i>=2:
            w=float(w)
        train_acc_list, bn_train_acc_list = train_function(w)
        train_list.append(train_acc_list)
        bn_train_list.append(bn_train_acc_list)
        
        plt.subplot(4,4,i+1)
        plt.title("W:" + str(w))
        if i == 15:
            plt.plot(x, bn_train_acc_list, label='Batch Normalization', markevery=1)
            plt.plot(x, train_acc_list, linestyle = "--", label='Normal(without BatchNorm)', markevery=1)
        else:
            plt.plot(x, bn_train_acc_list, markevery=2)
            plt.plot(x, train_acc_list, linestyle="--", markevery=2)

        plt.ylim(0, 1.0)
        if i % 4:
            plt.yticks([])
        else:
            plt.ylabel("accuracy")
        if i < 12:
            plt.xticks([])
        else:
            plt.xlabel("epochs")
        plt.legend(loc='lower right')
        
    
    
plt.show()    


    
########################################################################################
#下面是比較不同預設初始值及是否有用dropout

max_epochs = 20
train_size = x_train.shape[0]
batch_size = 500
learning_rate = 0.01
def train_function(weight_init_std):
    drop_network = Full_Layer_Net(input_size=784, hidden_size_list=[100, 100, 100, 100, 100], output_size=10, 
                                    weight_init_std=weight_init_std, use_batchnorm=True,
                                    use_dropout = True, dropout_ration = 0.3)
    network = Full_Layer_Net(input_size=784, hidden_size_list=[100, 100, 100, 100, 100], output_size=10,
                                weight_init_std=weight_init_std, use_batchnorm=True)
    optimizer = SGD(lr=learning_rate)
    
    train_acc_list = []
    drop_train_acc_list = []
    test_acc_list = []
    drop_test_acc_list = []
    
    
    iter_per_epoch =max(train_size / batch_size, 1)
    epoch_cnt = 0
    
    for ii in range(1000000000):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
    
        #for _network in (drop_network, network):
        #    grad = _network.gradient(x_batch, t_batch)
        #    optimizer.update(_network.param, grad)
        grad1 = drop_network.gradient(x_batch, t_batch)
        optimizer.update(drop_network.param, grad1)
        grad2 = network.gradient(x_batch, t_batch)
        optimizer.update(network.param, grad2)
        
        if ii % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            drop_train_acc = drop_network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            drop_test_acc = drop_network.accuracy(x_test, t_test)
            
            train_acc_list.append(train_acc)
            drop_train_acc_list.append(drop_train_acc)
            test_acc_list.append(test_acc)
            drop_test_acc_list.append(drop_test_acc)
            
    
            print("epoch:" + str(epoch_cnt) + " | " + str(train_acc) + " - " + str(drop_train_acc)+"-"+str(test_acc)+"-"+str(drop_test_acc))
    
            epoch_cnt += 1
            if epoch_cnt >= max_epochs:
                break   
    return train_acc_list, drop_train_acc_list ,test_acc_list,drop_test_acc_list  







weight_scale_list=np.concatenate([np.array(['sigmoid', 'relu']),np.logspace(0, -4, num=14)])
x = np.arange(max_epochs)
train_list=[]
drop_train_list=[]
test_list=[]
drop_test_list=[]
for i, w in enumerate(weight_scale_list):
        print( "============== " + str(i+1) + "/16" + " ==============")
        if i>=2:
            w=float(w)
        train_acc_list, drop_train_acc_list, test_acc_list, drop_test_acc_list = train_function(w)
        train_list.append(train_acc_list)
        drop_train_list.append(drop_train_acc_list)
        test_list.append(test_acc_list)
        drop_test_list.append(drop_test_acc_list)
        

#這邊是畫train
for i, w in enumerate(weight_scale_list):  
    
    if i>=2:
        w=float(w)
    plt.subplot(4,4,i+1)
    plt.title("W:" + str(w))
    if i == 15:
        plt.plot(x, drop_train_list[i], label='Batch Normalization', markevery=1)
        plt.plot(x, train_list[i], linestyle = "--", label='Normal(without BatchNorm)', markevery=1)
    else:
        plt.plot(x, drop_train_list[i], markevery=2)
        plt.plot(x, train_list[i], linestyle="--", markevery=2)

    plt.ylim(0, 1.0)
    if i % 4:
        plt.yticks([])
    else:
        plt.ylabel("accuracy")
    if i < 12:
        plt.xticks([])
    else:
        plt.xlabel("epochs")
    plt.legend(loc='lower right')
        
    
    
plt.show()
    
#這邊是畫test
for i, w in enumerate(weight_scale_list):  
    
    if i>=2:
        w=float(w)
    plt.subplot(4,4,i+1)
    plt.title("W:" + str(w))
    if i == 15:
        plt.plot(x, drop_test_list[i], label='bn,drop', markevery=1)
        plt.plot(x, test_list[i], linestyle = "--", label='bn,no drop', markevery=1)
    else:
        plt.plot(x, drop_test_list[i], markevery=2)
        plt.plot(x, test_list[i], linestyle="--", markevery=2)

    plt.ylim(0, 1.0)
    if i % 4:
        plt.yticks([])
    else:
        plt.ylabel("accuracy")
    if i < 12:
        plt.xticks([])
    else:
        plt.xlabel("epochs")
    plt.legend(loc='lower right')
        
    
    
plt.show()


##########################################################################
#decay、lr的參數調整


def shuffle_dataset(x, t):#將資料隨機排列
    permutation = np.random.permutation(x.shape[0])
    x = x[permutation,:] if x.ndim == 2 else x[permutation,:,:,:]
    t = t[permutation]

    return x, t
x_train = x_train[:500]
t_train = t_train[:500]

# 分割資料
#data-1.train 1.x_train
#                           ===抽mini_batch_size===>x_batch,t_batch
#             2.t_train
#     2.test  1.x_val
#             2.t_val
validation_rate = 0.20
validation_num = int(x_train.shape[0] * validation_rate)
x_train, t_train = shuffle_dataset(x_train, t_train)
x_val = x_train[:validation_num]
t_val = t_train[:validation_num]
x_train = x_train[validation_num:]
t_train = t_train[validation_num:]

class Trainer:
    """ニューラルネットの訓練を行うクラス
    """
    def __init__(self, network, x_train, t_train, x_test, t_test,
                 epochs=20, mini_batch_size=100,
                 optimizer='SGD', optimizer_param={'lr':0.01}, 
                 evaluate_sample_num_per_epoch=None, verbose=True):
        self.network = network
        self.verbose = verbose#決定要不要印出過程
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.epochs = epochs
        self.batch_size = mini_batch_size
        self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch


        # optimizer
        optimizer_class_dict = {'sgd':SGD, 'momentum':Momentum,'adagrad':AdaGrad}
        self.optimizer = optimizer_class_dict[optimizer.lower()](**optimizer_param)
        
        self.train_size = x_train.shape[0]
        self.iter_per_epoch = max(self.train_size / mini_batch_size, 1)
        self.max_iter = int(epochs * self.iter_per_epoch)
        self.current_iter = 0
        self.current_epoch = 0
        
        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

    def train_step(self):
        batch_mask = np.random.choice(self.train_size, self.batch_size)#從0~train_size抽出batch_size個
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]
        
        grads = self.network.gradient(x_batch, t_batch)
        self.optimizer.update(self.network.params, grads)
        
        loss = self.network.loss(x_batch, t_batch)
        self.train_loss_list.append(loss)
        if self.verbose: print("train loss:" + str(loss))
        
        if self.current_iter % self.iter_per_epoch == 0:
            self.current_epoch += 1
            
            x_train_sample, t_train_sample = self.x_train, self.t_train
            x_test_sample, t_test_sample = self.x_test, self.t_test
            if not self.evaluate_sample_num_per_epoch is None:
                t = self.evaluate_sample_num_per_epoch
                x_train_sample, t_train_sample = self.x_train[:t], self.t_train[:t]
                x_test_sample, t_test_sample = self.x_test[:t], self.t_test[:t]
                
            train_acc = self.network.accuracy(x_train_sample, t_train_sample)
            test_acc = self.network.accuracy(x_test_sample, t_test_sample)
            self.train_acc_list.append(train_acc)
            self.test_acc_list.append(test_acc)

            if self.verbose: print("=== epoch:" + str(self.current_epoch) + ", train acc:" + str(train_acc) + ", test acc:" + str(test_acc) + " ===")
        self.current_iter += 1

    def train(self):
        for i in range(self.max_iter):
            self.train_step()

        test_acc = self.network.accuracy(self.x_test, self.t_test)

        if self.verbose:
            print("=============== Final Test Accuracy ===============")
            print("test acc:" + str(test_acc))


def __train(lr, weight_decay, epocs=50):
    network = MultiLayerNetExtend(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100],
                            output_size=10, weight_decay_lambda=weight_decay)
    trainer = Trainer(network, x_train, t_train, x_val, t_val,
                      epochs=epocs, mini_batch_size=100,
                      optimizer='sgd', optimizer_param={'lr': lr}, verbose=False)
    trainer.train()

    return trainer.test_acc_list, trainer.train_acc_list




# ハイパーパラメータのランダム探索======================================
optimization_trial = 100
results_val = {}
results_train = {}
for _ in range(optimization_trial):
    # 探索したハイパーパラメータの範囲を指定===============
    weight_decay = 10 ** np.random.uniform(-8, -4)
    lr = 10 ** np.random.uniform(-6, -2)
    # ================================================

    val_acc_list, train_acc_list = __train(lr, weight_decay)
    print("val acc:" + str(val_acc_list[-1]) + " | lr:" + str(lr) + ", weight decay:" + str(weight_decay))
    key = "lr:" + str(lr) + ", weight decay:" + str(weight_decay)
    results_val[key] = val_acc_list
    results_train[key] = train_acc_list


# グラフの描画========================================================
print("=========== Hyper-Parameter Optimization Result ===========")
graph_draw_num = 20
col_num = 5
row_num = int(np.ceil(graph_draw_num / col_num))
i = 0

for key, val_acc_list in sorted(results_val.items(), key=lambda x:x[1][-1], reverse=True):
    print("Best-" + str(i+1) + "(val acc:" + str(val_acc_list[-1]) + ") | " + key)

    plt.subplot(row_num, col_num, i+1)
    plt.title("Best-" + str(i+1))
    plt.ylim(0.0, 1.0)
    if i % 5: plt.yticks([])
    plt.xticks([])
    x = np.arange(len(val_acc_list))
    plt.plot(x, val_acc_list)
    plt.plot(x, results_train[key], "--")
    i += 1

    if i >= graph_draw_num:
        break

plt.show()























class MultiLayerNetExtend:
    """拡張版の全結合による多層ニューラルネットワーク
    
    Weiht Decay、Dropout、Batch Normalizationの機能を持つ
    Parameters
    ----------
    input_size : 入力サイズ（MNISTの場合は784）
    hidden_size_list : 隠れ層のニューロンの数のリスト（e.g. [100, 100, 100]）
    output_size : 出力サイズ（MNISTの場合は10）
    activation : 'relu' or 'sigmoid'
    weight_init_std : 重みの標準偏差を指定（e.g. 0.01）
        'relu'または'he'を指定した場合は「Heの初期値」を設定
        'sigmoid'または'xavier'を指定した場合は「Xavierの初期値」を設定
    weight_decay_lambda : Weight Decay（L2ノルム）の強さ
    use_dropout: Dropoutを使用するかどうか
    dropout_ration : Dropoutの割り合い
    use_batchNorm: Batch Normalizationを使用するかどうか
    """
    def __init__(self, input_size, hidden_size_list, output_size,
                 activation='relu', weight_init_std='relu', weight_decay_lambda=0, 
                 use_dropout = False, dropout_ration = 0.5, use_batchnorm=False):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.use_dropout = use_dropout
        self.weight_decay_lambda = weight_decay_lambda
        self.use_batchnorm = use_batchnorm
        self.params = {}

        # 重みの初期化
        self.__init_weight(weight_init_std)

        # レイヤの生成
        activation_layer = {'sigmoid': Sigmoid, 'relu': Relu}
        self.layers = OrderedDict()
        for idx in range(1, self.hidden_layer_num+1):
            self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)],
                                                      self.params['b' + str(idx)])
            if self.use_batchnorm:
                self.params['gamma' + str(idx)] = np.ones(hidden_size_list[idx-1])
                self.params['beta' + str(idx)] = np.zeros(hidden_size_list[idx-1])
                self.layers['BatchNorm' + str(idx)] = BatchNormalization(self.params['gamma' + str(idx)], self.params['beta' + str(idx)])
                
            self.layers['Activation_function' + str(idx)] = activation_layer[activation]()
            
            if self.use_dropout:
                self.layers['Dropout' + str(idx)] = Dropout(dropout_ration)

        idx = self.hidden_layer_num + 1
        self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)], self.params['b' + str(idx)])

        self.last_layer = SoftmaxWithLoss()

    def __init_weight(self, weight_init_std):
        """重みの初期値設定
        Parameters
        ----------
        weight_init_std : 重みの標準偏差を指定（e.g. 0.01）
            'relu'または'he'を指定した場合は「Heの初期値」を設定
            'sigmoid'または'xavier'を指定した場合は「Xavierの初期値」を設定
        """
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for idx in range(1, len(all_size_list)):
            scale = weight_init_std
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale = np.sqrt(2.0 / all_size_list[idx - 1])  # ReLUを使う場合に推奨される初期値
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                scale = np.sqrt(1.0 / all_size_list[idx - 1])  # sigmoidを使う場合に推奨される初期値
            self.params['W' + str(idx)] = scale * np.random.randn(all_size_list[idx-1], all_size_list[idx])
            self.params['b' + str(idx)] = np.zeros(all_size_list[idx])

    def predict(self, x, train_flg=False):
        for key, layer in self.layers.items():
            if "Dropout" in key or "BatchNorm" in key:
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)

        return x

    def loss(self, x, t, train_flg=False):
        """損失関数を求める
        引数のxは入力データ、tは教師ラベル
        """
        y = self.predict(x, train_flg)

        weight_decay = 0
        for idx in range(1, self.hidden_layer_num + 2):
            W = self.params['W' + str(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W**2)

        return self.last_layer.forward(y, t) + weight_decay

    def accuracy(self, X, T):
        Y = self.predict(X, train_flg=False)
        Y = np.argmax(Y, axis=1)
        if T.ndim != 1 : T = np.argmax(T, axis=1)

        accuracy = np.sum(Y == T) / float(X.shape[0])
        return accuracy

    def numerical_gradient(self, X, T):
        """勾配を求める（数値微分）
        Parameters
        ----------
        X : 入力データ
        T : 教師ラベル
        Returns
        -------
        各層の勾配を持ったディクショナリ変数
            grads['W1']、grads['W2']、...は各層の重み
            grads['b1']、grads['b2']、...は各層のバイアス
        """
        loss_W = lambda W: self.loss(X, T, train_flg=True)

        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            grads['W' + str(idx)] = numerical_gradient(loss_W, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_W, self.params['b' + str(idx)])
            
            if self.use_batchnorm and idx != self.hidden_layer_num+1:
                grads['gamma' + str(idx)] = numerical_gradient(loss_W, self.params['gamma' + str(idx)])
                grads['beta' + str(idx)] = numerical_gradient(loss_W, self.params['beta' + str(idx)])

        return grads
        
    def gradient(self, x, t):
        # forward
        self.loss(x, t, train_flg=True)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 設定
        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            grads['W' + str(idx)] = self.layers['Affine' + str(idx)].dW + self.weight_decay_lambda * self.params['W' + str(idx)]
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db

            if self.use_batchnorm and idx != self.hidden_layer_num+1:
                grads['gamma' + str(idx)] = self.layers['BatchNorm' + str(idx)].dgamma
                grads['beta' + str(idx)] = self.layers['BatchNorm' + str(idx)].dbeta

        return grads