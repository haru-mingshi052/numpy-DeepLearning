## sigmoidとは
活性化関数の１種。  
２値分類のタスクの時に、出力層の活性化関数として使われる。  

## 式の確認
![](https://latex.codecogs.com/gif.latex?f(x)&space;=&space;\frac{1}{1&plus;\exp(-x)})

## 順伝播の計算グラフ
![](./image/sigmoid/forward.png)
<br />
```python
def forward(self, x):
    out = 1 / (1 + np.exp(-x))
    
    return out
```

## 逆伝播の計算グラフ
![](./image/sigmoid/backward.png)
<br />
```python
def backward(self, dout):
    dx = dout * (1.0 - self.out) * self.out
    
    return dx
```

## ノード4の逆伝播
![](./image/sigmoid/node4.png)  
<br />
![](https://latex.codecogs.com/gif.latex?\div)ノードの計算グラフ  
<br />
![](./image/utility/div_.png)  
<br />
![](https://latex.codecogs.com/gif.latex?\div)ノードは入力値の逆数を出力とするノード。  
逆伝播は計算すれば出力を２乗してーをつければOK。

## ノード3の逆伝播
![](./image/sigmoid/node3.png)

## ノード2の逆伝播
![](image/sigmoid/node2.png)  
<br />
![](https://latex.codecogs.com/gif.latex?\exp)ノードの計算グラフ  
<br />
![](./image/utility/exp.png)

## ノード1の逆伝播
![](image/sigmoid/node1.png)

## 式の変換
![](https://latex.codecogs.com/gif.latex?\frac{\partial&space;L}{\partial&space;x}&space;=&space;\frac{\partial&space;L}{\partial&space;y}&space;y^2&space;\exp(-x))  

ここで  

![](https://latex.codecogs.com/gif.latex?y&space;=&space;\frac{1}{1&space;&plus;&space;\exp(-x)})  

より  

![](https://latex.codecogs.com/gif.latex?\frac{\partial&space;L}{\partial&space;x}&space;=&space;\frac{\partial&space;L}{\partial&space;y}&space;\cdot&space;\frac{1}{1&space;&plus;&space;exp(-x)}&space;\left\\{&space;\frac{1&space;&plus;&space;exp(-x)}{1&space;&plus;&space;exp(-x)}&space;-&space;\frac{1}{1&space;&plus;&space;exp(-x)}&space;\right\\})  

と表せる。 

これを変形させると  

![](https://latex.codecogs.com/gif.latex?\frac{\partial&space;L}{\partial&space;x}&space;=&space;\frac{\partial&space;L}{\partial&space;y}&space;\cdot&space;\frac{1&space;&plus;&space;exp(-x)&space;-&space;1}{1&space;&plus;&space;exp(-x)}&space;\cdot&space;\frac{1}{1&space;&plus;&space;exp(-x)})  

となるので  

![](https://latex.codecogs.com/gif.latex?\frac{1}{1&space;&plus;&space;\exp(-x)})  

を抜き出すと  

![](https://latex.codecogs.com/gif.latex?\frac{\partial&space;L}{\partial&space;x}&space;=&space;\frac{\partial&space;L}{\partial&space;y}&space;\cdot&space;\frac{1}{1&space;&plus;&space;exp(-x)}&space;\left\\{&space;\frac{1&space;&plus;&space;exp(-x)}{1&space;&plus;&space;exp(-x)}&space;-&space;\frac{1}{1&space;&plus;&space;exp(-x)}&space;\right\\})  

になる。  

ここで  

![](https://latex.codecogs.com/gif.latex?\frac{1}{1&space;&plus;&space;\exp(-x)}&space;=&space;y)  

なので  

![](https://latex.codecogs.com/gif.latex?\frac{\partial&space;L}{\partial&space;x}&space;=&space;\frac{\partial&space;L}{\partial&space;y}&space;y&space;(&space;1&space;-&space;y&space;))  

となる。  

## おまけ（sigmoid関数の微分）  
![](https://latex.codecogs.com/gif.latex?f(x)&space;=&space;\frac{1}{1&space;&plus;&space;\exp(-x)})

![](https://latex.codecogs.com/gif.latex?f(x)&space;=&space;u^{-1},&space;u&space;=&space;1&space;&plus;&space;\exp(v),&space;v&space;=&space;-x)  

とおくと  

![](https://latex.codecogs.com/gif.latex?\frac{&space;\mathrm{d}f(x)&space;}{&space;\mathrm{d}x&space;}&space;=&space;\frac{&space;\mathrm{d}f(x)&space;}{&space;\mathrm{d}u&space;}&space;\cdot&space;\frac{&space;\mathrm{d}u&space;}{&space;\mathrm{d}v&space;}&space;\cdot&space;\frac{&space;\mathrm{d}v&space;}{&space;\mathrm{d}x&space;})  

と表せるので  

![](https://latex.codecogs.com/gif.latex?\frac{&space;\mathrm{d}f(x)&space;}{&space;\mathrm{d}u&space;}&space;=&space;-u^{-2})

![](https://latex.codecogs.com/gif.latex?\frac{&space;\mathrm{d}u&space;}{&space;\mathrm{d}v&space;}&space;=&space;\exp(v))  

![](https://latex.codecogs.com/gif.latex?\frac{&space;\mathrm{d}v&space;}{&space;\mathrm{d}x&space;}&space;=&space;-1)  

より  

![](https://latex.codecogs.com/gif.latex?\frac{&space;\mathrm{d}f(x)&space;}{&space;\mathrm{d}x&space;}&space;=&space;u^{-2}&space;\exp(v))

ここで、元の値を代入すると  

![](https://latex.codecogs.com/gif.latex?\frac{\mathrm{d}f(x)&space;}{\mathrm{d}x}&space;=&space;\left&space;\\{&space;\frac{1}{1&space;&plus;&space;exp(-x)}&space;\right&space;\\}^2&space;\cdot&space;exp(-x))  

となり、逆伝播時に求めた値と一致する。