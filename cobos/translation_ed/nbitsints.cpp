#include<iostream>
using namespace std;

unsigned int next_lex_permutation(unsigned int v) {
    unsigned int t = v | (v - 1);
    return (t + 1) | (((~t & -~t) - 1) >> (__builtin_ctz(v) + 1));
}

void fill_with_lex_permutations(unsigned long to_fill[], unsigned int nbits, unsigned int nones) {
    unsigned long first_int = 0;
    for (int n = 0; n < nones; n++) {
        first_int += 2^n;
    }
    unsigned long last_int = 0;
    for (int n = 0; n < nones; n++) {
        last_int += 2^(nbits - n);
    }
    unsigned long current_int = first_int;
    int i = 0;
    while (current_int < last_int)
    {
        to_fill[i] = current_int;
        current_int = next_lex_permutation(current_int);
        i++;
    }
}

int main(void){
    unsigned int nones = 3;
    unsigned int nbits = 10;
    unsigned long to_fill[(int) 2^10];

    fill_with_lex_permutations(to_fill, nbits, nones);

    for (int i = 0; i < ((int) 2^10); i++){
        cout << to_fill[i] << endl;
    }

    return 0;
}