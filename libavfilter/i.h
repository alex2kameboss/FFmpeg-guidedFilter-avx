#ifndef I_H

#define I_H

#define V 8

 // log2(V)
#define POW_V  3

//number of registers to use in a loop
#define NR 16   

// log2(V*NR)
#define POW_T  7  


//block size for transpose 32x32 floats = 4096 bytes, x 2 = 8KB goes
//in L1 data cache of 16kB
//#define BL 32
#define BL 64

// log2(BL)
#define POW_BL 6

//BL divided by V
#define BL_V 8

typedef __m256 float_packed;

#define MUL(a,b)     _mm256_mul_ps(a,b)
#define DIV(a,b)     _mm256_div_ps(a,b)
#define ADD(a,b)     _mm256_add_ps(a,b)
#define SUB(a,b)     _mm256_sub_ps(a,b)
#define LOAD(a)      _mm256_load_ps(&a)
#define STORE(a,b)   _mm256_store_ps(&a,b)
#define BROADCAST(a) _mm256_broadcast_ss(&a)



extern const float zero;

#endif