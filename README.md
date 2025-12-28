# 🧠 Recurrent Neural Network (RNN) — Built From Scratch (NumPy Only)

This project implements a **vanilla Recurrent Neural Network (RNN)** completely from scratch — using only **NumPy**.

The goal is to understand how RNNs actually work:

- how sequences are processed
- how memory flows from step to step
- how gradients train the model (Backpropagation Through Time)
- how text can be generated afterward

We train the model on a tiny string:

```
hello
```

The network learns to predict the next character.

---

## 📚 What is an RNN?

Normal neural networks assume inputs are **independent**.

But language, time-series, and audio have **order**:

```
h → e → l → l → o
```

So the meaning depends on what came before.

An RNN keeps a **hidden state (memory)** that is updated at each step.

### RNN formula

```
h_t = tanh(Wxh * x_t + Whh * h_{t-1} + bh)
y_t = softmax(Why * h_t + by)
```

Where:

| Symbol | Meaning |
|---|---|
x_t | current input (one-hot vector)
h_t | hidden state (memory)
Wxh | input → hidden weights
Whh | hidden → hidden (recurrent) weights
bh | hidden bias
Why | hidden → output weights
by | output bias

---

## 🏗️ Project structure

```
rnn_from_scratch.py   # main code
README.md             # documentation
```

---

## 🔎 Step-by-step explanation

### 1️⃣ Data preparation

Convert characters → indices → one-hot vectors.

Example (`vocab_size = 4`):

```
'l' -> [0, 0, 1, 0]
```

---

### 2️⃣ Initialize weights

Weights start as **small random numbers**:

```python
Wxh = np.random.randn(hidden_size, vocab_size) * 0.01
```

Why small?

- prevents tanh from saturating
- keeps gradients stable early on

---

### 3️⃣ Forward pass

For every character:

1) mix input + previous memory  
2) apply tanh  
3) compute probabilities

```python
hs[t] = np.tanh(Wxh @ xs[t] + Whh @ hs[t-1] + bh)
ys[t] = Why @ hs[t] + by
ps[t] = softmax(ys[t])
```

---

### 4️⃣ Loss (how wrong the model is)

We use **cross-entropy**:

```
loss = -log(probability of correct character)
```

Big penalty if model is confident but wrong.

---

### 5️⃣ Backpropagation Through Time (BPTT)

We compute gradients:

- dWxh — input → hidden
- dWhh — recurrent
- dWhy — hidden → output
- dbh, dby — biases
- dh_next — gradient flowing backward through time

We also clip gradients to avoid exploding values.

---

### 6️⃣ Weight updates

Gradient Descent:

```
weight = weight - learning_rate * gradient
```

This slowly improves predictions.

---

### 7️⃣ Text generation

Start with a seed character and repeatedly:

1) compute hidden state  
2) sample next character  
3) feed it back in  

Eventually:

```
hello
```

(or close!)

---

## 🧪 What you learn here

- what hidden state actually is
- why RNNs can remember sequences
- how loss + gradients train the network
- what frameworks (PyTorch/TensorFlow) do behind the scenes

---

## 🚀 Possible extensions

- add LSTM (forget + input gates)
- add GRU
- train on words instead of characters
- convert to PyTorch
- add batching

If you want to build any of those, open an issue or try it yourself 🙂

---

## 🙌 Credits

Educational RNN example rewritten and documented for clarity.

