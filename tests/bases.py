import torch
import matplotlib.pyplot as plt

from functional_data import FourierBasis, RandomFourierBasis

torch.set_default_dtype(torch.float64)

thetas = torch.linspace(0, 1, 100).numpy()


test_rffs = RandomFourierBasis(100, 233, 300, [0, 1])

Phi = test_rffs.compute_Phi(thetas)

plt.plot(Phi[0])
plt.plot(Phi[1])
plt.plot(Phi[2])
plt.show()

test_fourier = FourierBasis(0, 10, [0, 1])
Phi = test_fourier.compute_Phi(thetas)
