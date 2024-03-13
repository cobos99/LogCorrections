# ENTROPY PLOTS DESCRIPTION

## Filename legend

### Quantities

- SR: Rènyi entropy of infinite finite order
- SS: Shannon entropy

### Fitting parameters

- I always fit to the same function:  $S_{\mathrm{R}/\mathrm{Sh}} = a L + b \log L + c$

### Hamiltonian

- global_neg_False: $H_+ = \sum_n \left( X_n X_{n+1} + Y_n Y_{n+1} + \Delta Z_n Z_{n+1} \right)$

- global_neg_True: $H_- = -\sum_n \left( X_n X_{n+1} + Y_n Y_{n+1} + \Delta Z_n Z_{n+1} \right)$

    (With periodic boundary conditions)

### Chain lengths

- N indicates the chain length, a step of 2 is always assumed, so that the odd/even chains can be identified from the filename.

## Exact value of the fitting parameters at notable points

### Rènyi entropy

- $H_+$ | $L \; \mathrm{odd} \in [11, 23]$

    | $\Delta$ 	| $a$   	| $b$    	| $c$    	|
    |----------	|-------	|--------	|--------	|
    | $-1$     	| 0.635 	| -0.419 	| -0.293 	|
    | $-1/2$   	| 0.450 	| 0.145  	| -0.339 	|
    | $0$      	| 0.346 	| 0.255  	| -0.205 	|
    | $1/2$    	| 0.261 	| 0.362  	| -0.073 	|
    | $1$      	| 0.18  	| 0.555  	| -0.053 	|

- $H_-$ | $L \; \mathrm{odd} \in [11, 23]$

    | $\Delta$ 	| $a$   	| $b$    	| $c$    	|
    |----------	|-------	|--------	|--------	|
    | $-1$     	| 0.18  	| 0.579  	| -0.056 	|
    | $-1/2$   	| 0.260 	| 0.370  	| -0.056 	|
    | $0$      	| 0.346 	| 0.255  	| -0.206 	|
    | $1/2$    	| 0.450 	| 0.172  	| -0.440 	|
    | $1$      	| 0.691 	| -0.415 	| -0.465 	|

- $H_{+/-}$ | $L \; \mathrm{even} \in [10, 22]$

    | $\Delta$ 	| $a$   	| $b$    	| $c$    	|
    |----------	|-------	|--------	|--------	|
    | $-1$     	| 0.180 	| 0.017  	| -0.369 	|
    | $-1/2$   	| 0.262 	| -0.012 	| 0.202  	|
    | $0$      	| 0.346 	| 0      	| 0      	|
    | $1/2$    	| 0.450 	| 0.012  	| -0.240 	|
    | $1$      	| 0.692 	| -0.464 	| -0.320 	|

    **Comments**:

    - The parameters $a, b$ for the odd cases are close for opposite signs of $\Delta$. In particular $a$ is almost the same. $b$ is close enough for discrepancies to be due to numerical errors. $c$ shows more discrepancies. This is maybe caused by the non-null quasimomenta of the ground state of $H_-$.

    - $a$ Indicates that the even model is actually more similar to the odd $H_-$ case.

    - The quasimomenta $k$ of the ground states is

        - $H_+: k_1 = \left\lfloor (N - 1)/4 \right\rfloor + (\left\lfloor (N - 1)/4 \right\rfloor \mod 2)$
        
            $k_2 = N - k_1$

        - $H_+: k_1 = 0$

### Shannon entropy

- $H_+$ | $L \; \mathrm{odd} \in [11, 23]$

    | $\Delta$ 	| $a$   	| $b$    	| $c$    	|
    |----------	|-------	|--------	|--------	|
    | $-1$     	| 0.69  	| -0.419 	| -0.469 	|
    | $-1/2$   	| 0.587 	| 0.095  	| -0.841 	|
    | $0$      	| 0.534 	| 0.151  	| -0.765 	|
    | $1/2$    	| 0.486 	| 0.145  	| -0.628 	|
    | $1$      	| 0.429 	| 0.131 	| -0.406 	|

- $H_-$ | $L \; \mathrm{odd} \in [11, 23]$

    | $\Delta$ 	| $a$   	| $b$    	| $c$    	|
    |----------	|-------	|--------	|--------	|
    | $-1$     	| 0.429 	| 0.139  	| -0.368 	|
    | $-1/2$   	| 0.486 	| 0.136  	| -0.583 	|
    | $0$      	| 0.534 	| 0.151  	| -0.765 	|
    | $1/2$    	| 0.587 	| 0.102  	| -0.874 	|
    | $1$      	| 0.691 	| -0.416 	| -0.465 	|

- $H_{+/-}$ | $L \; \mathrm{even} \in [10, 22]$

    | $\Delta$ 	| $a$   	| $b$    	| $c$    	|
    |----------	|-------	|--------	|--------	|
    | $-1$     	| 0.434 	| 0.053  	| -0.279 	|
    | $-1/2$   	| 0.492 	| 0.042  	| -0.402 	|
    | $0$      	| 0.541 	| 0.022  	| -0.499 	|
    | $1/2$    	| 0.594 	| -0.021 	| -0.608 	|
    | $1$      	| 0.692 	| -0.466 	| -0.320 	|

    **Comments**:

    - There are non-negligible logaritmic corrections in the odd case also in the Shannon entropy.

    - Same comments on the comparison between different $+/-$ models apply for this case.

## CONCLUSIONS

- The entropy in both $+/-$ Hamiltonian behaves very similarly. We need TNs to go to larger sizes and remove discrepancies. If we want to really get into this the real next step is to addres the computation of the entropies from the TN side. If we do this, I would root for fully getting into the logarithmic corrections in odd critical model business and study many different models to see whether this is actually a general thing. So what I should do before getting into TNs from my side is explore other models with exact diagonalization so that the effort related with TNs is worth it.

- If we want to keep ourselves inside the classical-quantum correspondeces with the vertex models we need to find what is the meaning of these logaritmic corrections in in the classical model.

- Personal opinion: I think the logaritmic corrections in odd critical models thing, if true, is more interesting
