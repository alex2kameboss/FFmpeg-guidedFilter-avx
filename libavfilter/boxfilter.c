#include <immintrin.h>
#include <stdalign.h>
#include "i.h"
#include "boxfilter.h"

alignas(32) const float zero = 0;

int boxfilter(const float *x_in, float *x_out, size_t r, size_t n, size_t ld, float *work)
{

  /*
  - x_in  = pointer to 2D image n x n as 1D array with leading dimension ld

  - x_out = pointer to output 2D image n x n as 1D array with leading dimension ld
          = result of filter applied in 1 dimension to x_in

  - r     = radius of filter, made of 2*r+1 ones

  - n     = size of image, should be >= 2 * r + 1

  - ld    = leading dimension for x_in and x_out, should be divisible by V!

  - work  = pointer to a work array ld x ld+2 of leading dimension ld divisible by 8 and greater than n

   - return values:
          0     ok
         -1     n < 2 * r + 1, image smaller than filter.
         -2     ld is not divisible by V, vector register length (for floats!)

  */

  float *t;
  float *ai;
  float *bi;

  if (n < 2 * r + 1)
  {
    return -1;
  }
  if ((ld >> POW_V) << POW_V != ld)
  {
    return -2; // ld not divisible by V
  }

  t = work;

  ai = work + ld * ld;

  bi = work + (ld + 1) * ld;

  size_t i;

  for (i = 0; i < r + 1; i++)
  {
    ai[i] = 1. / (r + 1 + i) / (2 * r + 1);
    bi[i] = 1. / (r + 1 + i) * (2 * r + 1);
  }

  for (i = r + 1; i < n - (r + 1); i++)
  {
    ai[i] = 1. / (2 * r + 1) / (2 * r + 1);
  }

  for (i = n - (r + 1); i < n; i++)
  {
    ai[i] = ai[n - 1 - i];
  }

  boxfilter1D(x_in, t, r, n, ld);

  // cblas_somatcopy(CblasRowMajor, CblasTrans, n, n, 1.0f, t, ld, x_out, ld);

  transpose(t, x_out, n, ld);

  boxfilter1D_norm(x_out, t, r, n, ld, ai, bi);

  // cblas_somatcopy(CblasRowMajor, CblasTrans, n, n, 1.0f, t, ld, x_out, ld);

  transpose(t, x_out, n, ld);

  return 0;
}

void boxfilter1D_norm(const float *x_in, float *x_out, size_t r, size_t n, size_t ld, const float *a_norm, const float *b_norm)
{

  /*
    - x_in  = pointer to 2D image n x n as 1D array

    - x_out = pointer to output 2D image n x n as 1D array
            = result of filter applied in 1 dimension to x_in
        AND final normalization by multiplication with matrix 1./N

    - r     = radius of filter, made of 2*r+1 ones

    - n     = size of image, should be >= 2 * r + 1

    - ld    = leading dimension for x_in and x_out, should be divisible by V!

    - a_norm = pointer to a vector of length ld

    - b_norm = pointer to a vector of length r+1

    Note: a_norm @ b_norm = 1./N, where @ is the tensor product and N
    is the normalization matrix.

    b_norm(k) = 1 for k > r && k < n - r

    These values for b are not saved, as multiplication with 1 can be
    omitted!

    See further comments in boxfilter1D.c

  */

  float_packed v[NR];
  float_packed s[NR];
  float_packed va, vb;

  const float *a0 = x_in;
  const float *a_diff = x_in;
  float *b0 = x_out;

  const float *a;
  float *b;

  const float *ai = a_norm;

  size_t i, j, k, rest, ni;

  // Loop using NR registers with V values, with POW_T = log2(V*NR)

  ni = n >> POW_T; // ni = n/V/NR;

  //  k = ni;

  // printf("%ld\n", ni);

  for (k = 0; k < ni; k++)
  {
    a = a0;
    b = b0;
    a_diff = a0;

    s[0] = BROADCAST(zero);

    for (i = 1; i < NR; i++)
    {
      s[i] = s[0];
    }

    for (j = 0; j < r; j++)
    {
      for (i = 0; i < NR; i++)
      {
        v[i] = LOAD(a[i * V]);
        s[i] = ADD(s[i], v[i]);
      }
      a += ld;
    }

    for (j = 0; j < r + 1; j++)
    {
      vb = BROADCAST(b_norm[j]);
      for (i = 0; i < NR; i++)
      {
        v[i] = LOAD(a[i * V]);
        s[i] = ADD(s[i], v[i]);
        va = LOAD(ai[i * V]);
        v[i] = MUL(s[i], va);
        v[i] = MUL(v[i], vb);
        STORE(b[i * V], v[i]);
      }
      a += ld;
      b += ld;
    }

    for (j = 0; j < n - 2 * r - 1; j++)
    {
      for (i = 0; i < NR; i++)
      {
        v[i] = LOAD(a[i * V]);
        s[i] = ADD(s[i], v[i]);
        v[i] = LOAD(a_diff[i * V]);
        s[i] = SUB(s[i], v[i]);
        va = LOAD(ai[i * V]);
        v[i] = MUL(s[i], va);
        STORE(b[i * V], v[i]);
      }
      a += ld;
      b += ld;
      a_diff += ld;
    };

    for (j = 0; j < r; j++)
    {
      vb = BROADCAST(b_norm[r - 1 - j]);
      for (i = 0; i < NR; i++)
      {
        v[i] = LOAD(a_diff[i * V]);
        s[i] = SUB(s[i], v[i]);
        va = LOAD(ai[i * V]);
        v[i] = MUL(s[i], va);
        v[i] = MUL(v[i], vb);
        STORE(b[i * V], v[i]);
      }
      b += ld;
      a_diff += ld;
    }

    a0 += NR * V;
    b0 += NR * V;
    ai += NR * V;
  }

  rest = n - (ni << POW_T);

  if (rest == 0)
    return;

  ni = rest >> POW_V; // ni = rest/V

  // printf("%ld\n", ni);

  if (rest - (ni << POW_V) > 0)
    ni = ni + 1; // based on leading dimension being divisible by V!

  a = a0;
  b = b0;
  a_diff = a0;

  s[0] = BROADCAST(zero);
  for (i = 1; i < ni; i++)
  {
    s[i] = s[0];
  }

  for (j = 0; j < r; j++)
  {
    for (i = 0; i < ni; i++)
    {
      v[i] = LOAD(a[i * V]);
      s[i] = ADD(s[i], v[i]);
    }
    a += ld;
  }

  for (j = 0; j < r + 1; j++)
  {
    vb = BROADCAST(b_norm[j]);
    for (i = 0; i < ni; i++)
    {
      v[i] = LOAD(a[i * V]);
      s[i] = ADD(s[i], v[i]);
      va = LOAD(ai[i * V]);
      v[i] = MUL(s[i], va);
      v[i] = MUL(v[i], vb);
      STORE(b[i * V], v[i]);
    }
    a += ld;
    b += ld;
  }
  while (j -= 1)
    ;

  for (j = 0; j < n - 2 * r - 1; j++)
  {
    for (i = 0; i < ni; i++)
    {
      v[i] = LOAD(a[i * V]);
      s[i] = ADD(s[i], v[i]);
      v[i] = LOAD(a_diff[i * V]);
      s[i] = SUB(s[i], v[i]);
      va = LOAD(ai[i * V]);
      v[i] = MUL(s[i], va);
      STORE(b[i * V], v[i]);
    }
    a += ld;
    b += ld;
    a_diff += ld;
  }

  for (j = 0; j < r; j++)
  {
    vb = BROADCAST(b_norm[r - 1 - j]);
    for (i = 0; i < ni; i++)
    {
      v[i] = LOAD(a_diff[i * V]);
      s[i] = SUB(s[i], v[i]);
      va = LOAD(ai[i * V]);
      v[i] = MUL(s[i], va);
      v[i] = MUL(v[i], vb);
      STORE(b[i * V], v[i]);
    }
    b += ld;
    a_diff += ld;
  }

  return;
}

void boxfilter1D(const float *x_in, float *x_out, size_t r, size_t n, size_t ld)
{

  /*
    - x_in  = pointer to 2D image n x n as 1D array

    - x_out = pointer to output 2D image n x n as 1D array
            = result of filter applied in 1 dimension to x_in

    - r     = radius of filter, made of 2*r+1 ones

    - n     = size of image, should be >= 2 * r + 1

    - ld    = leading dimension for x_in and x_out, should be divisible by V!



    NOTICE 1: Both x_in and x_out should be aligned to 32bytes for
    avx2 and 64bytes for avx512.  This can be done for example using
    aligned_alloc instead of malloc:

    float *x_in = aligned_alloc(32, (n*n)*sizeof(float));

    or using alignas(32) float x_in[N*N] for static arrays.

    NOTICE 2: If n is not divisible by V, consider a leading dimension
    ld divisible by V and save x_in as a submatrix of size n x n in a
    matrix ld x n (column-major) or n x ld (row-major).


    The function works for both column-major and row-major 2D arrays
    by using index arithmetic in a 1D array.



    Consider the row-major case.

    NR = number of registers, each with V float values (NR = 16, V = 8
    for avx2 with 256b)

    Outer loop over NR*V columns

      Inner loop: sliding window over all lines

      Go to next NR*V columns (via a0 for x_in and b0 for x_out)

    End outer loop

    Repeat once the outer loop for the rest of columns up to next number divisible by V greater than n,
    therefore still using vector acceleration but less than NR registers. This works as leading dimension ld is
    divisible by 8!

    Example for n = 1005, leading dimension ld = 1008, NR = 16, V = 8, NR*V = 128

    Outer loop with NR registers and V values - 7 times = 896 lines

    Outer loop with 13+1 registers and V values - once = 112 lines (max (NR-1)*V lines)

    Total: 1008 lines. Note that the last 3 lines are present via ld but never used in x_in and x_out.



    The inner loop over columns is divided in 4 loops - first r columns, next r+1, central part, last r columns

    The pointers a and a_diff for x_in and b for x_out start with the first column, current lines (via a0 and b0)


    Inner central loop (with j = n - 2 * r - 1):

        s contains NR*V values saved already in x_out.

        Read NR*V values from the next line starting from pointer a

        Add them to s

        Read NR*V values from line - (2*r-1)

        Substract them from s

        Save s to x_out, starting from pointer b

        Point a and b go the next column, same lines, a = a + n, b = b + n

    End central loop


  */

  float_packed v[NR];
  float_packed s[NR];

  const float *a0 = x_in;
  const float *a_diff = x_in;
  float *b0 = x_out;

  const float *a;
  float *b;

  size_t i, j, k, rest, ni;

  // Loop using NR registers with V values, with POW_T = log2(V*NR)

  ni = n >> POW_T; // ni = n/V/NR;

  //  k = ni;

  // printf("%ld\n", ni);

  for (k = 0; k < ni; k++)
  {
    a = a0;
    b = b0;
    a_diff = a0;

    s[0] = BROADCAST(zero);

    for (i = 1; i < NR; i++)
    {
      s[i] = s[0];
    }

    for (j = 0; j < r; j++)
    {
      for (i = 0; i < NR; i++)
      {
        v[i] = LOAD(a[i * V]);
        s[i] = ADD(s[i], v[i]);
      }
      a += ld;
    }

    for (j = 0; j < r + 1; j++)
    {
      for (i = 0; i < NR; i++)
      {
        v[i] = LOAD(a[i * V]);
        s[i] = ADD(s[i], v[i]);
        STORE(b[i * V], s[i]);
      }
      a += ld;
      b += ld;
    }

    for (j = 0; j < n - 2 * r - 1; j++)
    {
      for (i = 0; i < NR; i++)
      {
        v[i] = LOAD(a[i * V]);
        s[i] = ADD(s[i], v[i]);
        v[i] = LOAD(a_diff[i * V]);
        s[i] = SUB(s[i], v[i]);
        STORE(b[i * V], s[i]);
      }
      a += ld;
      b += ld;
      a_diff += ld;
    };

    for (j = 0; j < r; j++)
    {
      for (i = 0; i < NR; i++)
      {
        v[i] = LOAD(a_diff[i * V]);
        s[i] = SUB(s[i], v[i]);
        STORE(b[i * V], s[i]);
      }
      b += ld;
      a_diff += ld;
    }

    a0 += NR * V;
    b0 += NR * V;
  }

  rest = n - (ni << POW_T);

  if (rest == 0)
    return;

  ni = rest >> POW_V; // ni = rest/V

  // printf("%ld\n", ni);

  if (rest - (ni << POW_V) > 0)
    ni = ni + 1; // based on leading dimension being divisible by V!

  a = a0;
  b = b0;
  a_diff = a0;

  s[0] = BROADCAST(zero);
  for (i = 1; i < ni; i++)
  {
    s[i] = s[0];
  }

  for (j = 0; j < r; j++)
  {
    for (i = 0; i < ni; i++)
    {
      v[i] = LOAD(a[i * V]);
      s[i] = ADD(s[i], v[i]);
    }
    a += ld;
  }

  for (j = 0; j < r + 1; j++)
  {
    for (i = 0; i < ni; i++)
    {
      v[i] = LOAD(a[i * V]);
      s[i] = ADD(s[i], v[i]);
      STORE(b[i * V], s[i]);
    }
    a += ld;
    b += ld;
  }
  while (j -= 1)
    ;

  for (j = 0; j < n - 2 * r - 1; j++)
  {
    for (i = 0; i < ni; i++)
    {
      v[i] = LOAD(a[i * V]);
      s[i] = ADD(s[i], v[i]);
      v[i] = LOAD(a_diff[i * V]);
      s[i] = SUB(s[i], v[i]);
      STORE(b[i * V], s[i]);
    }
    a += ld;
    b += ld;
    a_diff += ld;
  }

  for (j = 0; j < r; j++)
  {
    for (i = 0; i < ni; i++)
    {
      v[i] = LOAD(a_diff[i * V]);
      s[i] = SUB(s[i], v[i]);
      STORE(b[i * V], s[i]);
    }
    b += ld;
    a_diff += ld;
  }
}

void matmul(const float *x1, const float *x2, float *y, size_t n, size_t ld)
{

  size_t i, ld_red;

  float_packed v1, v2, vy;

  ld_red = ld >> POW_V;

  const float *a1 = x1;
  const float *a2 = x2;
  float *b = y;

  i = n * ld_red;
  do
  {
    v1 = LOAD(a1[0]);
    v2 = LOAD(a2[0]);
    vy = MUL(v1, v2);
    STORE(b[0], vy);
    a1 += V;
    a2 += V;
    b += V;
  } while (i -= 1);
}

void diffmatmul(const float *x1, const float *x2, const float *x3, float *y, size_t n, size_t ld)
{

  size_t i, ld_red;

  float_packed v1, v2, v3, vy;

  ld_red = ld >> POW_V;

  const float *a1 = x1;
  const float *a2 = x2;
  const float *a3 = x3;
  float *b = y;

  i = n * ld_red;
  do
  {

    v2 = LOAD(a2[0]);
    v3 = LOAD(a3[0]);
    vy = MUL(v2, v3);
    v1 = LOAD(a1[0]);
    vy = SUB(v1, vy);
    STORE(b[0], vy);
    a1 += V;
    a2 += V;
    a3 += V;
    b += V;
  } while (i -= 1);
}

void addmatmul(const float *x1, const float *x2, const float *x3, float *y, size_t n, size_t ld)
{

  size_t i, ld_red;

  float_packed v1, v2, v3, vy;

  ld_red = ld >> POW_V;

  const float *a1 = x1;
  const float *a2 = x2;
  const float *a3 = x3;
  float *b = y;

  i = n * ld_red;
  do
  {

    v2 = LOAD(a2[0]);
    v3 = LOAD(a3[0]);
    vy = MUL(v2, v3);
    v1 = LOAD(a1[0]);
    vy = ADD(v1, vy);
    STORE(b[0], vy);
    a1 += V;
    a2 += V;
    a3 += V;
    b += V;
  } while (i -= 1);
}

void matdivconst(const float *x1, const float *x2, float *y, size_t n, size_t ld, float e)
{

  size_t i, ld_red;

  float_packed v1, v2, vy, ve;

  ve = BROADCAST(e);

  ld_red = ld >> POW_V;

  const float *a1 = x1;
  const float *a2 = x2;
  float *b = y;

  i = n * ld_red;
  do
  {
    v1 = LOAD(a1[0]);
    v2 = LOAD(a2[0]);
    v2 = ADD(v2, ve);
    vy = DIV(v1, v2);
    STORE(b[0], vy);
    a1 += V;
    a2 += V;
    b += V;
  } while (i -= 1);
}

void transpose_8x8(float *a, float *b, size_t n)
{
  float_packed v0, v1, v2, v3, v4, v5, v6, v7;
  float_packed s0, s1, s2, s3, s4, s5, s6, s7;

  v0 = LOAD(a[0]);
  v1 = LOAD(a[n]);
  v2 = LOAD(a[2 * n]);
  v3 = LOAD(a[3 * n]);
  v4 = LOAD(a[4 * n]);
  v5 = LOAD(a[5 * n]);
  v6 = LOAD(a[6 * n]);
  v7 = LOAD(a[7 * n]);

  s0 = _mm256_unpacklo_ps(v0, v1);
  s1 = _mm256_unpackhi_ps(v0, v1);
  s2 = _mm256_unpacklo_ps(v2, v3);
  s3 = _mm256_unpackhi_ps(v2, v3);
  s4 = _mm256_unpacklo_ps(v4, v5);
  s5 = _mm256_unpackhi_ps(v4, v5);
  s6 = _mm256_unpacklo_ps(v6, v7);
  s7 = _mm256_unpackhi_ps(v6, v7);

  v0 = _mm256_shuffle_ps(s0, s2, _MM_SHUFFLE(1, 0, 1, 0));
  v1 = _mm256_shuffle_ps(s0, s2, _MM_SHUFFLE(3, 2, 3, 2));
  v2 = _mm256_shuffle_ps(s1, s3, _MM_SHUFFLE(1, 0, 1, 0));
  v3 = _mm256_shuffle_ps(s1, s3, _MM_SHUFFLE(3, 2, 3, 2));
  v4 = _mm256_shuffle_ps(s4, s6, _MM_SHUFFLE(1, 0, 1, 0));
  v5 = _mm256_shuffle_ps(s4, s6, _MM_SHUFFLE(3, 2, 3, 2));
  v6 = _mm256_shuffle_ps(s5, s7, _MM_SHUFFLE(1, 0, 1, 0));
  v7 = _mm256_shuffle_ps(s5, s7, _MM_SHUFFLE(3, 2, 3, 2));

  s0 = _mm256_permute2f128_ps(v0, v4, 0x20);
  s1 = _mm256_permute2f128_ps(v1, v5, 0x20);
  s2 = _mm256_permute2f128_ps(v2, v6, 0x20);
  s3 = _mm256_permute2f128_ps(v3, v7, 0x20);
  s4 = _mm256_permute2f128_ps(v0, v4, 0x31);
  s5 = _mm256_permute2f128_ps(v1, v5, 0x31);
  s6 = _mm256_permute2f128_ps(v2, v6, 0x31);
  s7 = _mm256_permute2f128_ps(v3, v7, 0x31);

  STORE(b[0], s0);
  STORE(b[n], s1);
  STORE(b[2 * n], s2);
  STORE(b[3 * n], s3);
  STORE(b[4 * n], s4);
  STORE(b[5 * n], s5);
  STORE(b[6 * n], s6);
  STORE(b[7 * n], s7);
}

void transpose(float *in, float *out, size_t n, size_t ld)
{
  /*
    For the moment works only for n divisible by V
  */

  size_t ni = n >> POW_BL;
  size_t rest;

  float *a0 = in;
  float *b0 = out;

  float *a;
  float *b;

  size_t k, l, i, j;

  for (k = 0; k < ni; k++)
  {
    for (l = 0; l < ni; l++)
    {
      a = a0;
      b = b0;
      for (i = 0; i < BL_V; i++)
      {
        for (j = 0; j < BL_V; j++)
        {
          transpose_8x8(a, b, ld);
          a += V;
          b += ld * V;
        }
        a -= BL_V * V;
        a += ld * V;
        b -= BL_V * ld * V;
        b += V;
      }

      a0 += BL_V * V;
      b0 += ld * BL_V * V;
    }

    a0 -= ni * BL_V * V;
    a0 += ld * BL_V * V;

    b0 -= ni * ld * BL_V * V;
    b0 += BL_V * V;
  }

  rest = n - (ni << POW_BL);

  if (rest == 0)
    return;

  rest = rest >> POW_V;

  a0 = in + ni * BL_V * V;
  b0 = out + ni * BL_V * V * ld;

  for (k = 0; k < ni; k++)
  {
    a = a0;
    b = b0;
    for (i = 0; i < BL_V; i++)
    {
      for (j = 0; j < rest; j++)
      {
        transpose_8x8(a, b, ld);
        a += V;
        b += ld * V;
      }
      a -= rest * V;
      a += ld * V;
      b -= rest * ld * V;
      b += V;
    }
    a0 += ld * BL_V * V;
    b0 += BL_V * V;
  }

  a0 = in + ni * BL_V * V * ld;
  b0 = out + ni * BL_V * V;

  for (k = 0; k < ni; k++)
  {
    a = a0;
    b = b0;
    for (i = 0; i < BL_V; i++)
    {
      for (j = 0; j < rest; j++)
      {
        transpose_8x8(a, b, ld);
        a += ld * V;
        b += V;
      }
      a -= rest * ld * V;
      a += V;
      b -= rest * V;
      b += ld * V;
    }
    a0 += BL_V * V;
    b0 += ld * BL_V * V;
  }

  a = in + BL_V * ni * V * ld + BL_V * ni * V;
  b = out + BL_V * ni * V * ld + BL_V * ni * V;

  for (i = 0; i < rest; i++)
  {
    for (j = 0; j < rest; j++)
    {
      transpose_8x8(a, b, ld);
      a += V;
      b += ld * V;
    }
    a -= rest * V;
    a += ld * V;
    b -= rest * ld * V;
    b += V;
  }
}
