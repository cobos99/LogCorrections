import numpy as np
import matplotlib.pyplot as plt

pi = np.pi

def delta1(L, N):
    result = 0
    # niter = 0
    for j in range(1, N+1):
        for k in range(j+1, N+1):
            # niter = niter + 1
            # print(f"j = {j}, k = {k}")
            result = result + np.log(np.abs(np.exp(4j * pi * k / L) - np.exp(4j * pi * j / L)))
    # print(f"N = {N}, niter = {niter}, expected = {N*(N-1)/2}")
    return result

def delta2(L, N):
    result = 0
    for j in range(1, N):
        result = result + np.log(np.abs(1 - np.exp(4j * pi * j / L)))
    return (N/2) * result


def delta3(L, N):
    result = 0
    for j in range(1, N+1):
        for k in range(1, N+1):
            if j == k:
                continue
            result = result + np.log(np.abs(np.exp(4j * pi * k / L) - np.exp(4j * pi * j / L)))
    return result / 2

# delta1(20, 10)

domain_N = np.arange(50, 100)
delta1_even_L = np.array([ delta1(2*N,   N) for N in domain_N ])
delta2_even_L = np.array([ delta2(2*N,   N) for N in domain_N ])
delta3_even_L = np.array([ delta3(2*N,   N) for N in domain_N ])
delta1_odd_L =  np.array([ delta1(2*N+1, N) for N in domain_N ])
delta2_odd_L =  np.array([ delta2(2*N+1, N) for N in domain_N ])
delta3_odd_L =  np.array([ delta3(2*N+1, N) for N in domain_N ])

plt.subplots(1, 2)

# Even L
plt.subplot(1, 2, 1)
plt.plot(domain_N, delta1_even_L, label=r"$\Delta_1$")
plt.plot(domain_N, delta2_even_L, label=r"$\Delta_2$")
plt.plot(domain_N, delta3_even_L, label=r"$\Delta_3$")
plt.legend()
# odd L
plt.subplot(1, 2, 2)
plt.plot(domain_N, delta1_odd_L, label=r"$\Delta_1$")
plt.plot(domain_N, delta2_odd_L, label=r"$\Delta_2$")
plt.plot(domain_N, delta3_odd_L, label=r"$\Delta_3$")
plt.legend()


plt.show()
