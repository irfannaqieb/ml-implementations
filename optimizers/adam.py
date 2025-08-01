import numpy as np

class AdamOptimizer():
  def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    self.params = params
    self.lr = lr
    self.beta1 = beta1
    self.beta2 = beta2
    self.epsilon = epsilon

    # time step
    self.t = 0

    # first and second moment estimates initialization
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

      # update parameters
      self.params[i] -= self.lr * m_hat / (v_hat ** 0.5 + self.epsilon)
