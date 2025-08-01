import numpy as np

def AdamWOptimizer():
  def __init__(self, params, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.01 ):
    self.params = params
    self.learning_rate = learning_rate
    self.beta1 = beta1
    self.beta2 = beta2
    self.epsilon = epsilon
    self.weight_decay = weight_decay

    self.t = 0

    self.m = [np.zeros_like(param) for param in self.params]
    self.v = [np.zeros_like(param) for param in self.params]

  def step(self, grads):
    self.t += 1
    for i in range(len(self.params)):

      # first moment calculation
      self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]

      # second moment calculation
      self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grads[i] ** 2)

      # bias correction
      m_hat = self.m[i] / (1 - self.beta1 ** self.t)
      v_hat = self.v[i] / (1 - self.beta2 ** self.t)

      # parameter update
      update = self.learning_rate * m_hat / (v_hat ** (1/2) + self.epsilon)
      self.params[i] -= update

      # Weight decay
      if self.weight_decay > 0:
        self.params[i] -= self.learning_rate * self.weight_decay * self.params[i]