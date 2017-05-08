/**
 *  \file src/MBCSR/MatTrisolve/1x5.c
 *  \brief The \f$1\times 5\f$ MBCSR implementation
 *  of sparse triangular solve.
 *  \ingroup MATTYPE_MBCSR
 *
 *  Automatically generated by ./gen_trisolve.sh
 *  on Wed Jun  8 15:56:40 PDT 2005.
 */

#include <assert.h>

#include <oski/config.h> /* for 'restrict' keyword */
#include <oski/common.h>
#include <oski/matrix.h>
#include <oski/blas_names.h>
#include <oski/MBCSR/format.h>
#include <oski/MBCSR/module.h>

#if IS_VAL_COMPLEX
	/** Complex-valued, so do not use explicit 'register' keyword. */
	#define REGISTER
#else
	/** Real-valued, so use explicit 'register' keyword. */
	#define REGISTER register
#endif

#if defined(DO_NAME_MANGLING)
	/** Mangled name for MBCSR_MatTrisolve_Lower_v1_aX_xs1. */
	#define MBCSR_MatTrisolve_Lower_v1_aX_xs1 \
		MANGLE_MOD_(MBCSR_MatTrisolve_Lower_v1_aX_xs1_1x5)
#endif

/**
 *  \brief The \f$1\times 5\f$ MBCSR implementation
 *  of \f$L^{-1}\cdot b\f$, where x has unit-stride.
 */
void
MBCSR_MatTrisolve_Lower_v1_aX_xs1(
	oski_index_t M, oski_index_t d0,
	const oski_index_t* restrict ptr, const oski_index_t* restrict ind,
	const oski_value_t* restrict val, const oski_value_t* restrict diag,
	oski_value_t alpha, oski_value_t* restrict x )
{
	oski_index_t I;
	oski_value_t* bp = x + d0;
	const oski_value_t* dp = diag;

	for( I = 1; I != M+1; I++, bp += 1, dp += 1*1 )
	{
		oski_index_t K;
		REGISTER oski_value_t _b0;
		const oski_value_t* vp = val + ptr[I-1]*1*5;

		VAL_MUL( _b0, alpha, bp[0] );

		for( K = ptr[I-1]; K < ptr[I]; K++, vp += 1*5 )
		{
			oski_index_t j0 = ind[K];
			const oski_value_t* xp = x + j0;
			REGISTER oski_value_t _x0;
			REGISTER oski_value_t _x1;
			REGISTER oski_value_t _x2;
			REGISTER oski_value_t _x3;
			REGISTER oski_value_t _x4;

			VAL_ASSIGN( _x0, xp[0] );
			VAL_ASSIGN( _x1, xp[1] );
			VAL_ASSIGN( _x2, xp[2] );
			VAL_ASSIGN( _x3, xp[3] );
			VAL_ASSIGN( _x4, xp[4] );

			VAL_MSUB( _b0, vp[0], _x0 );
			VAL_MSUB( _b0, vp[1], _x1 );
			VAL_MSUB( _b0, vp[2], _x2 );
			VAL_MSUB( _b0, vp[3], _x3 );
			VAL_MSUB( _b0, vp[4], _x4 );
		} /* K */

		VAL_DIVEQ( _b0, dp[0*1+0] );
		VAL_ASSIGN( bp[0], _b0 );
	} /* I */
} /* MBCSR_MatTrisolve_Lower_v1_aX_xs1 */

#if defined(DO_NAME_MANGLING)
	/** Mangled name for MBCSR_MatTrisolve_Lower_v1_aX_xsX. */
	#define MBCSR_MatTrisolve_Lower_v1_aX_xsX \
		MANGLE_MOD_(MBCSR_MatTrisolve_Lower_v1_aX_xsX_1x5)
#endif

/**
 *  \brief The \f$1\times 5\f$ MBCSR implementation
 *  of \f$L^{-1}\cdot b\f$, where x has general-stride.
 */
void
MBCSR_MatTrisolve_Lower_v1_aX_xsX(
	oski_index_t M, oski_index_t d0,
	const oski_index_t* restrict ptr, const oski_index_t* restrict ind,
	const oski_value_t* restrict val, const oski_value_t* restrict diag,
	oski_value_t alpha, oski_value_t* restrict x, oski_index_t incx )
{
	oski_index_t I;
	oski_value_t* bp = x + d0*incx;
	const oski_value_t* dp = diag;

	for( I = 1; I != M+1; I++, bp += 1*incx, dp += 1*1 )
	{
		oski_index_t K;
		REGISTER oski_value_t _b0;
		const oski_value_t* vp = val + ptr[I-1]*1*5;

		VAL_MUL( _b0, alpha, bp[0] );

		for( K = ptr[I-1]; K < ptr[I]; K++, vp += 1*5 )
		{
			oski_index_t j0 = ind[K];
			const oski_value_t* xp = x + j0*incx;
			REGISTER oski_value_t _x0;
			REGISTER oski_value_t _x1;
			REGISTER oski_value_t _x2;
			REGISTER oski_value_t _x3;
			REGISTER oski_value_t _x4;

			VAL_ASSIGN( _x0, xp[0] );
			VAL_ASSIGN( _x1, xp[1*incx] );
			VAL_ASSIGN( _x2, xp[2*incx] );
			VAL_ASSIGN( _x3, xp[3*incx] );
			VAL_ASSIGN( _x4, xp[4*incx] );

			VAL_MSUB( _b0, vp[0], _x0 );
			VAL_MSUB( _b0, vp[1], _x1 );
			VAL_MSUB( _b0, vp[2], _x2 );
			VAL_MSUB( _b0, vp[3], _x3 );
			VAL_MSUB( _b0, vp[4], _x4 );
		} /* K */

		VAL_DIVEQ( _b0, dp[0*1+0] );
		VAL_ASSIGN( bp[0], _b0 );
	} /* I */
} /* MBCSR_MatTrisolve_Lower_v1_aX_xsX */

/**
 *  \brief Exported module wrapper for the \f$1\times 5\f$
 *  implementation of the sparse triangular solve operation,
 *  where the matrix is lower triangular.
 */
static void
MatTrisolve_Lower( const oski_submatMBCSR_t* T,
	oski_value_t alpha, oski_vecview_t x )
{
	assert( T != NULL );
	assert( T->r == 1 );
	assert( T->c == 5 );
	assert( x != INVALID_VEC );

	if( x->rowinc == 1 ) {
		oski_index_t j;
		oski_value_t* xp;
		for( j = 0, xp = x->val; j < x->num_cols; j++, xp += x->colinc )
			MBCSR_MatTrisolve_Lower_v1_aX_xs1( T->num_block_rows, T->offset,
				T->bptr, T->bind, T->bval, T->bdiag, alpha, xp );
	} else { /* x has general (non-unit) stride */
		oski_index_t j;
		oski_value_t* xp;
		for( j = 0, xp = x->val; j < x->num_cols; j++, xp += x->colinc )
			MBCSR_MatTrisolve_Lower_v1_aX_xsX( T->num_block_rows, T->offset,
				T->bptr, T->bind, T->bval, T->bdiag, alpha, xp, x->rowinc );
	}
}

#if !IS_VAL_COMPLEX
	/** Synonym for pure real version */
	#define MBCSR_MatConjTrisolve_Lower_v1_aX_xs1 MBCSR_MatTrisolve_Lower_v1_aX_xs1
#else /* IS_VAL_COMPLEX */

#if defined(DO_NAME_MANGLING)
	/** Mangled name for MBCSR_MatConjTrisolve_Lower_v1_aX_xs1. */
	#define MBCSR_MatConjTrisolve_Lower_v1_aX_xs1 \
		MANGLE_MOD_(MBCSR_MatConjTrisolve_Lower_v1_aX_xs1_1x5)
#endif

#endif
#if IS_VAL_COMPLEX
/**
 *  \brief The \f$1\times 5\f$ MBCSR implementation
 *  of \f$\bar{L}^{-1}\cdot b\f$, where x has unit-stride.
 */
void
MBCSR_MatConjTrisolve_Lower_v1_aX_xs1(
	oski_index_t M, oski_index_t d0,
	const oski_index_t* restrict ptr, const oski_index_t* restrict ind,
	const oski_value_t* restrict val, const oski_value_t* restrict diag,
	oski_value_t alpha, oski_value_t* restrict x )
{
	oski_index_t I;
	oski_value_t* bp = x + d0;
	const oski_value_t* dp = diag;

	for( I = 1; I != M+1; I++, bp += 1, dp += 1*1 )
	{
		oski_index_t K;
		REGISTER oski_value_t _b0;
		const oski_value_t* vp = val + ptr[I-1]*1*5;

		VAL_MUL( _b0, alpha, bp[0] );

		for( K = ptr[I-1]; K < ptr[I]; K++, vp += 1*5 )
		{
			oski_index_t j0 = ind[K];
			const oski_value_t* xp = x + j0;
			REGISTER oski_value_t _x0;
			REGISTER oski_value_t _x1;
			REGISTER oski_value_t _x2;
			REGISTER oski_value_t _x3;
			REGISTER oski_value_t _x4;

			VAL_ASSIGN( _x0, xp[0] );
			VAL_ASSIGN( _x1, xp[1] );
			VAL_ASSIGN( _x2, xp[2] );
			VAL_ASSIGN( _x3, xp[3] );
			VAL_ASSIGN( _x4, xp[4] );

			VAL_MSUB_CONJ( _b0, vp[0], _x0 );
			VAL_MSUB_CONJ( _b0, vp[1], _x1 );
			VAL_MSUB_CONJ( _b0, vp[2], _x2 );
			VAL_MSUB_CONJ( _b0, vp[3], _x3 );
			VAL_MSUB_CONJ( _b0, vp[4], _x4 );
		} /* K */

		VAL_DIVEQ_CONJ( _b0, dp[0*1+0] );
		VAL_ASSIGN( bp[0], _b0 );
	} /* I */
} /* MBCSR_MatConjTrisolve_Lower_v1_aX_xs1 */

#endif /* IS_VAL_COMPLEX */
#if !IS_VAL_COMPLEX
	/** Synonym for pure real version */
	#define MBCSR_MatConjTrisolve_Lower_v1_aX_xsX MBCSR_MatTrisolve_Lower_v1_aX_xsX
#else /* IS_VAL_COMPLEX */

#if defined(DO_NAME_MANGLING)
	/** Mangled name for MBCSR_MatConjTrisolve_Lower_v1_aX_xsX. */
	#define MBCSR_MatConjTrisolve_Lower_v1_aX_xsX \
		MANGLE_MOD_(MBCSR_MatConjTrisolve_Lower_v1_aX_xsX_1x5)
#endif

#endif
#if IS_VAL_COMPLEX
/**
 *  \brief The \f$1\times 5\f$ MBCSR implementation
 *  of \f$\bar{L}^{-1}\cdot b\f$, where x has general-stride.
 */
void
MBCSR_MatConjTrisolve_Lower_v1_aX_xsX(
	oski_index_t M, oski_index_t d0,
	const oski_index_t* restrict ptr, const oski_index_t* restrict ind,
	const oski_value_t* restrict val, const oski_value_t* restrict diag,
	oski_value_t alpha, oski_value_t* restrict x, oski_index_t incx )
{
	oski_index_t I;
	oski_value_t* bp = x + d0*incx;
	const oski_value_t* dp = diag;

	for( I = 1; I != M+1; I++, bp += 1*incx, dp += 1*1 )
	{
		oski_index_t K;
		REGISTER oski_value_t _b0;
		const oski_value_t* vp = val + ptr[I-1]*1*5;

		VAL_MUL( _b0, alpha, bp[0] );

		for( K = ptr[I-1]; K < ptr[I]; K++, vp += 1*5 )
		{
			oski_index_t j0 = ind[K];
			const oski_value_t* xp = x + j0*incx;
			REGISTER oski_value_t _x0;
			REGISTER oski_value_t _x1;
			REGISTER oski_value_t _x2;
			REGISTER oski_value_t _x3;
			REGISTER oski_value_t _x4;

			VAL_ASSIGN( _x0, xp[0] );
			VAL_ASSIGN( _x1, xp[1*incx] );
			VAL_ASSIGN( _x2, xp[2*incx] );
			VAL_ASSIGN( _x3, xp[3*incx] );
			VAL_ASSIGN( _x4, xp[4*incx] );

			VAL_MSUB_CONJ( _b0, vp[0], _x0 );
			VAL_MSUB_CONJ( _b0, vp[1], _x1 );
			VAL_MSUB_CONJ( _b0, vp[2], _x2 );
			VAL_MSUB_CONJ( _b0, vp[3], _x3 );
			VAL_MSUB_CONJ( _b0, vp[4], _x4 );
		} /* K */

		VAL_DIVEQ_CONJ( _b0, dp[0*1+0] );
		VAL_ASSIGN( bp[0], _b0 );
	} /* I */
} /* MBCSR_MatConjTrisolve_Lower_v1_aX_xsX */

#endif /* IS_VAL_COMPLEX */
/**
 *  \brief Exported module wrapper for the \f$1\times 5\f$
 *  implementation of the sparse triangular solve operation,
 *  where the matrix is lower triangular.
 */
static void
MatConjTrisolve_Lower( const oski_submatMBCSR_t* T,
	oski_value_t alpha, oski_vecview_t x )
{
	assert( T != NULL );
	assert( T->r == 1 );
	assert( T->c == 5 );
	assert( x != INVALID_VEC );

	if( x->rowinc == 1 ) {
		oski_index_t j;
		oski_value_t* xp;
		for( j = 0, xp = x->val; j < x->num_cols; j++, xp += x->colinc )
			MBCSR_MatConjTrisolve_Lower_v1_aX_xs1( T->num_block_rows, T->offset,
				T->bptr, T->bind, T->bval, T->bdiag, alpha, xp );
	} else { /* x has general (non-unit) stride */
		oski_index_t j;
		oski_value_t* xp;
		for( j = 0, xp = x->val; j < x->num_cols; j++, xp += x->colinc )
			MBCSR_MatConjTrisolve_Lower_v1_aX_xsX( T->num_block_rows, T->offset,
				T->bptr, T->bind, T->bval, T->bdiag, alpha, xp, x->rowinc );
	}
}

#if defined(DO_NAME_MANGLING)
	/** Mangled name for MBCSR_MatTransTrisolve_Lower_v1_aX_xs1. */
	#define MBCSR_MatTransTrisolve_Lower_v1_aX_xs1 \
		MANGLE_MOD_(MBCSR_MatTransTrisolve_Lower_v1_aX_xs1_1x5)
#endif

/**
 *  \brief The \f$1\times 5\f$ MBCSR implementation
 *  of \f$L^{-T}\cdot b\f$, where x has unit-stride.
 */
void
MBCSR_MatTransTrisolve_Lower_v1_aX_xs1(
	oski_index_t M, oski_index_t d0,
	const oski_index_t* restrict ptr, const oski_index_t* restrict ind,
	const oski_value_t* restrict val, const oski_value_t* restrict diag,
	oski_value_t alpha, oski_value_t* restrict x )
{
	oski_index_t I;
	oski_value_t* bp = x + (d0 + (M-1)*1);
	const oski_value_t* dp = diag + (M-1)*1*1;

	{
		oski_index_t m = M * 1;
		oski_index_t incx = 1;
		BLAS_xSCAL( &m, &alpha, x, &incx );
	}

	for( I = M; I != 0; I--, bp -= 1, dp -= 1*1 )
	{
		oski_index_t K;
		REGISTER oski_value_t _b0;
		const oski_value_t* vp = val + ptr[I-1]*1*5;

		VAL_ASSIGN( _b0, bp[0] );

		VAL_DIVEQ( _b0, dp[0*1+0] );

		for( K = ptr[I-1]; K < ptr[I]; K++, vp += 1*5 )
		{
			oski_index_t j0 = ind[K];
			oski_value_t* xp = x + j0;
			REGISTER oski_value_t _x0;
			REGISTER oski_value_t _x1;
			REGISTER oski_value_t _x2;
			REGISTER oski_value_t _x3;
			REGISTER oski_value_t _x4;

			VAL_ASSIGN( _x0, xp[0] );
			VAL_ASSIGN( _x1, xp[1] );
			VAL_ASSIGN( _x2, xp[2] );
			VAL_ASSIGN( _x3, xp[3] );
			VAL_ASSIGN( _x4, xp[4] );

			VAL_MSUB( _x0, vp[0], _b0 );
			VAL_MSUB( _x1, vp[1], _b0 );
			VAL_MSUB( _x2, vp[2], _b0 );
			VAL_MSUB( _x3, vp[3], _b0 );
			VAL_MSUB( _x4, vp[4], _b0 );
			VAL_ASSIGN( xp[0], _x0 );
			VAL_ASSIGN( xp[1], _x1 );
			VAL_ASSIGN( xp[2], _x2 );
			VAL_ASSIGN( xp[3], _x3 );
			VAL_ASSIGN( xp[4], _x4 );
		} /* K */

		VAL_ASSIGN( bp[0], _b0 );
	} /* I */
} /* MBCSR_MatTransTrisolve_Lower_v1_aX_xs1 */

#if defined(DO_NAME_MANGLING)
	/** Mangled name for MBCSR_MatTransTrisolve_Lower_v1_aX_xsX. */
	#define MBCSR_MatTransTrisolve_Lower_v1_aX_xsX \
		MANGLE_MOD_(MBCSR_MatTransTrisolve_Lower_v1_aX_xsX_1x5)
#endif

/**
 *  \brief The \f$1\times 5\f$ MBCSR implementation
 *  of \f$L^{-T}\cdot b\f$, where x has general-stride.
 */
void
MBCSR_MatTransTrisolve_Lower_v1_aX_xsX(
	oski_index_t M, oski_index_t d0,
	const oski_index_t* restrict ptr, const oski_index_t* restrict ind,
	const oski_value_t* restrict val, const oski_value_t* restrict diag,
	oski_value_t alpha, oski_value_t* restrict x, oski_index_t incx )
{
	oski_index_t I;
	oski_value_t* bp = x + (d0 + (M-1)*1)*incx;
	const oski_value_t* dp = diag + (M-1)*1*1;

	{
		oski_index_t m = M * 1;
		BLAS_xSCAL( &m, &alpha, x, &incx );
	}

	for( I = M; I != 0; I--, bp -= 1*incx, dp -= 1*1 )
	{
		oski_index_t K;
		REGISTER oski_value_t _b0;
		const oski_value_t* vp = val + ptr[I-1]*1*5;

		VAL_ASSIGN( _b0, bp[0] );

		VAL_DIVEQ( _b0, dp[0*1+0] );

		for( K = ptr[I-1]; K < ptr[I]; K++, vp += 1*5 )
		{
			oski_index_t j0 = ind[K];
			oski_value_t* xp = x + j0*incx;
			REGISTER oski_value_t _x0;
			REGISTER oski_value_t _x1;
			REGISTER oski_value_t _x2;
			REGISTER oski_value_t _x3;
			REGISTER oski_value_t _x4;

			VAL_ASSIGN( _x0, xp[0] );
			VAL_ASSIGN( _x1, xp[1*incx] );
			VAL_ASSIGN( _x2, xp[2*incx] );
			VAL_ASSIGN( _x3, xp[3*incx] );
			VAL_ASSIGN( _x4, xp[4*incx] );

			VAL_MSUB( _x0, vp[0], _b0 );
			VAL_MSUB( _x1, vp[1], _b0 );
			VAL_MSUB( _x2, vp[2], _b0 );
			VAL_MSUB( _x3, vp[3], _b0 );
			VAL_MSUB( _x4, vp[4], _b0 );
			VAL_ASSIGN( xp[0], _x0 );
			VAL_ASSIGN( xp[1*incx], _x1 );
			VAL_ASSIGN( xp[2*incx], _x2 );
			VAL_ASSIGN( xp[3*incx], _x3 );
			VAL_ASSIGN( xp[4*incx], _x4 );
		} /* K */

		VAL_ASSIGN( bp[0], _b0 );
	} /* I */
} /* MBCSR_MatTransTrisolve_Lower_v1_aX_xsX */

/**
 *  \brief Exported module wrapper for the \f$1\times 5\f$
 *  implementation of the sparse triangular solve operation,
 *  where the matrix is lower triangular.
 */
static void
MatTransTrisolve_Lower( const oski_submatMBCSR_t* T,
	oski_value_t alpha, oski_vecview_t x )
{
	assert( T != NULL );
	assert( T->r == 1 );
	assert( T->c == 5 );
	assert( x != INVALID_VEC );

	if( x->rowinc == 1 ) {
		oski_index_t j;
		oski_value_t* xp;
		for( j = 0, xp = x->val; j < x->num_cols; j++, xp += x->colinc )
			MBCSR_MatTransTrisolve_Lower_v1_aX_xs1( T->num_block_rows, T->offset,
				T->bptr, T->bind, T->bval, T->bdiag, alpha, xp );
	} else { /* x has general (non-unit) stride */
		oski_index_t j;
		oski_value_t* xp;
		for( j = 0, xp = x->val; j < x->num_cols; j++, xp += x->colinc )
			MBCSR_MatTransTrisolve_Lower_v1_aX_xsX( T->num_block_rows, T->offset,
				T->bptr, T->bind, T->bval, T->bdiag, alpha, xp, x->rowinc );
	}
}

#if !IS_VAL_COMPLEX
	/** Synonym for pure real version */
	#define MBCSR_MatHermTrisolve_Lower_v1_aX_xs1 MBCSR_MatTransTrisolve_Lower_v1_aX_xs1
#else /* IS_VAL_COMPLEX */

#if defined(DO_NAME_MANGLING)
	/** Mangled name for MBCSR_MatHermTrisolve_Lower_v1_aX_xs1. */
	#define MBCSR_MatHermTrisolve_Lower_v1_aX_xs1 \
		MANGLE_MOD_(MBCSR_MatHermTrisolve_Lower_v1_aX_xs1_1x5)
#endif

#endif
#if IS_VAL_COMPLEX
/**
 *  \brief The \f$1\times 5\f$ MBCSR implementation
 *  of \f$\bar{L}^{-T}\cdot b\f$, where x has unit-stride.
 */
void
MBCSR_MatHermTrisolve_Lower_v1_aX_xs1(
	oski_index_t M, oski_index_t d0,
	const oski_index_t* restrict ptr, const oski_index_t* restrict ind,
	const oski_value_t* restrict val, const oski_value_t* restrict diag,
	oski_value_t alpha, oski_value_t* restrict x )
{
	oski_index_t I;
	oski_value_t* bp = x + (d0 + (M-1)*1);
	const oski_value_t* dp = diag + (M-1)*1*1;

	{
		oski_index_t m = M * 1;
		oski_index_t incx = 1;
		BLAS_xSCAL( &m, &alpha, x, &incx );
	}

	for( I = M; I != 0; I--, bp -= 1, dp -= 1*1 )
	{
		oski_index_t K;
		REGISTER oski_value_t _b0;
		const oski_value_t* vp = val + ptr[I-1]*1*5;

		VAL_ASSIGN( _b0, bp[0] );

		VAL_DIVEQ_CONJ( _b0, dp[0*1+0] );

		for( K = ptr[I-1]; K < ptr[I]; K++, vp += 1*5 )
		{
			oski_index_t j0 = ind[K];
			oski_value_t* xp = x + j0;
			REGISTER oski_value_t _x0;
			REGISTER oski_value_t _x1;
			REGISTER oski_value_t _x2;
			REGISTER oski_value_t _x3;
			REGISTER oski_value_t _x4;

			VAL_ASSIGN( _x0, xp[0] );
			VAL_ASSIGN( _x1, xp[1] );
			VAL_ASSIGN( _x2, xp[2] );
			VAL_ASSIGN( _x3, xp[3] );
			VAL_ASSIGN( _x4, xp[4] );

			VAL_MSUB_CONJ( _x0, vp[0], _b0 );
			VAL_MSUB_CONJ( _x1, vp[1], _b0 );
			VAL_MSUB_CONJ( _x2, vp[2], _b0 );
			VAL_MSUB_CONJ( _x3, vp[3], _b0 );
			VAL_MSUB_CONJ( _x4, vp[4], _b0 );
			VAL_ASSIGN( xp[0], _x0 );
			VAL_ASSIGN( xp[1], _x1 );
			VAL_ASSIGN( xp[2], _x2 );
			VAL_ASSIGN( xp[3], _x3 );
			VAL_ASSIGN( xp[4], _x4 );
		} /* K */

		VAL_ASSIGN( bp[0], _b0 );
	} /* I */
} /* MBCSR_MatHermTrisolve_Lower_v1_aX_xs1 */

#endif /* IS_VAL_COMPLEX */
#if !IS_VAL_COMPLEX
	/** Synonym for pure real version */
	#define MBCSR_MatHermTrisolve_Lower_v1_aX_xsX MBCSR_MatTransTrisolve_Lower_v1_aX_xsX
#else /* IS_VAL_COMPLEX */

#if defined(DO_NAME_MANGLING)
	/** Mangled name for MBCSR_MatHermTrisolve_Lower_v1_aX_xsX. */
	#define MBCSR_MatHermTrisolve_Lower_v1_aX_xsX \
		MANGLE_MOD_(MBCSR_MatHermTrisolve_Lower_v1_aX_xsX_1x5)
#endif

#endif
#if IS_VAL_COMPLEX
/**
 *  \brief The \f$1\times 5\f$ MBCSR implementation
 *  of \f$\bar{L}^{-T}\cdot b\f$, where x has general-stride.
 */
void
MBCSR_MatHermTrisolve_Lower_v1_aX_xsX(
	oski_index_t M, oski_index_t d0,
	const oski_index_t* restrict ptr, const oski_index_t* restrict ind,
	const oski_value_t* restrict val, const oski_value_t* restrict diag,
	oski_value_t alpha, oski_value_t* restrict x, oski_index_t incx )
{
	oski_index_t I;
	oski_value_t* bp = x + (d0 + (M-1)*1)*incx;
	const oski_value_t* dp = diag + (M-1)*1*1;

	{
		oski_index_t m = M * 1;
		BLAS_xSCAL( &m, &alpha, x, &incx );
	}

	for( I = M; I != 0; I--, bp -= 1*incx, dp -= 1*1 )
	{
		oski_index_t K;
		REGISTER oski_value_t _b0;
		const oski_value_t* vp = val + ptr[I-1]*1*5;

		VAL_ASSIGN( _b0, bp[0] );

		VAL_DIVEQ_CONJ( _b0, dp[0*1+0] );

		for( K = ptr[I-1]; K < ptr[I]; K++, vp += 1*5 )
		{
			oski_index_t j0 = ind[K];
			oski_value_t* xp = x + j0*incx;
			REGISTER oski_value_t _x0;
			REGISTER oski_value_t _x1;
			REGISTER oski_value_t _x2;
			REGISTER oski_value_t _x3;
			REGISTER oski_value_t _x4;

			VAL_ASSIGN( _x0, xp[0] );
			VAL_ASSIGN( _x1, xp[1*incx] );
			VAL_ASSIGN( _x2, xp[2*incx] );
			VAL_ASSIGN( _x3, xp[3*incx] );
			VAL_ASSIGN( _x4, xp[4*incx] );

			VAL_MSUB_CONJ( _x0, vp[0], _b0 );
			VAL_MSUB_CONJ( _x1, vp[1], _b0 );
			VAL_MSUB_CONJ( _x2, vp[2], _b0 );
			VAL_MSUB_CONJ( _x3, vp[3], _b0 );
			VAL_MSUB_CONJ( _x4, vp[4], _b0 );
			VAL_ASSIGN( xp[0], _x0 );
			VAL_ASSIGN( xp[1*incx], _x1 );
			VAL_ASSIGN( xp[2*incx], _x2 );
			VAL_ASSIGN( xp[3*incx], _x3 );
			VAL_ASSIGN( xp[4*incx], _x4 );
		} /* K */

		VAL_ASSIGN( bp[0], _b0 );
	} /* I */
} /* MBCSR_MatHermTrisolve_Lower_v1_aX_xsX */

#endif /* IS_VAL_COMPLEX */
/**
 *  \brief Exported module wrapper for the \f$1\times 5\f$
 *  implementation of the sparse triangular solve operation,
 *  where the matrix is lower triangular.
 */
static void
MatHermTrisolve_Lower( const oski_submatMBCSR_t* T,
	oski_value_t alpha, oski_vecview_t x )
{
	assert( T != NULL );
	assert( T->r == 1 );
	assert( T->c == 5 );
	assert( x != INVALID_VEC );

	if( x->rowinc == 1 ) {
		oski_index_t j;
		oski_value_t* xp;
		for( j = 0, xp = x->val; j < x->num_cols; j++, xp += x->colinc )
			MBCSR_MatHermTrisolve_Lower_v1_aX_xs1( T->num_block_rows, T->offset,
				T->bptr, T->bind, T->bval, T->bdiag, alpha, xp );
	} else { /* x has general (non-unit) stride */
		oski_index_t j;
		oski_value_t* xp;
		for( j = 0, xp = x->val; j < x->num_cols; j++, xp += x->colinc )
			MBCSR_MatHermTrisolve_Lower_v1_aX_xsX( T->num_block_rows, T->offset,
				T->bptr, T->bind, T->bval, T->bdiag, alpha, xp, x->rowinc );
	}
}

#if defined(DO_NAME_MANGLING)
	/** Mangled name for MBCSR_MatTrisolve_Upper_v1_aX_xs1. */
	#define MBCSR_MatTrisolve_Upper_v1_aX_xs1 \
		MANGLE_MOD_(MBCSR_MatTrisolve_Upper_v1_aX_xs1_1x5)
#endif

/**
 *  \brief The \f$1\times 5\f$ MBCSR implementation
 *  of \f$U^{-1}\cdot b\f$, where x has unit-stride.
 */
void
MBCSR_MatTrisolve_Upper_v1_aX_xs1(
	oski_index_t M, oski_index_t d0,
	const oski_index_t* restrict ptr, const oski_index_t* restrict ind,
	const oski_value_t* restrict val, const oski_value_t* restrict diag,
	oski_value_t alpha, oski_value_t* restrict x )
{
	oski_index_t I;
	oski_value_t* bp = x + (d0 + (M-1)*1);
	const oski_value_t* dp = diag + (M-1)*1*1;

	for( I = M; I != 0; I--, bp -= 1, dp -= 1*1 )
	{
		oski_index_t K;
		REGISTER oski_value_t _b0;
		const oski_value_t* vp = val + ptr[I-1]*1*5;

		VAL_MUL( _b0, alpha, bp[0] );

		for( K = ptr[I-1]; K < ptr[I]; K++, vp += 1*5 )
		{
			oski_index_t j0 = ind[K];
			const oski_value_t* xp = x + j0;
			REGISTER oski_value_t _x0;
			REGISTER oski_value_t _x1;
			REGISTER oski_value_t _x2;
			REGISTER oski_value_t _x3;
			REGISTER oski_value_t _x4;

			VAL_ASSIGN( _x0, xp[0] );
			VAL_ASSIGN( _x1, xp[1] );
			VAL_ASSIGN( _x2, xp[2] );
			VAL_ASSIGN( _x3, xp[3] );
			VAL_ASSIGN( _x4, xp[4] );

			VAL_MSUB( _b0, vp[0], _x0 );
			VAL_MSUB( _b0, vp[1], _x1 );
			VAL_MSUB( _b0, vp[2], _x2 );
			VAL_MSUB( _b0, vp[3], _x3 );
			VAL_MSUB( _b0, vp[4], _x4 );
		} /* K */

		VAL_DIVEQ( _b0, dp[0*1+0] );
		VAL_ASSIGN( bp[0], _b0 );
	} /* I */
} /* MBCSR_MatTrisolve_Upper_v1_aX_xs1 */

#if defined(DO_NAME_MANGLING)
	/** Mangled name for MBCSR_MatTrisolve_Upper_v1_aX_xsX. */
	#define MBCSR_MatTrisolve_Upper_v1_aX_xsX \
		MANGLE_MOD_(MBCSR_MatTrisolve_Upper_v1_aX_xsX_1x5)
#endif

/**
 *  \brief The \f$1\times 5\f$ MBCSR implementation
 *  of \f$U^{-1}\cdot b\f$, where x has general-stride.
 */
void
MBCSR_MatTrisolve_Upper_v1_aX_xsX(
	oski_index_t M, oski_index_t d0,
	const oski_index_t* restrict ptr, const oski_index_t* restrict ind,
	const oski_value_t* restrict val, const oski_value_t* restrict diag,
	oski_value_t alpha, oski_value_t* restrict x, oski_index_t incx )
{
	oski_index_t I;
	oski_value_t* bp = x + (d0 + (M-1)*1)*incx;
	const oski_value_t* dp = diag + (M-1)*1*1;

	for( I = M; I != 0; I--, bp -= 1*incx, dp -= 1*1 )
	{
		oski_index_t K;
		REGISTER oski_value_t _b0;
		const oski_value_t* vp = val + ptr[I-1]*1*5;

		VAL_MUL( _b0, alpha, bp[0] );

		for( K = ptr[I-1]; K < ptr[I]; K++, vp += 1*5 )
		{
			oski_index_t j0 = ind[K];
			const oski_value_t* xp = x + j0*incx;
			REGISTER oski_value_t _x0;
			REGISTER oski_value_t _x1;
			REGISTER oski_value_t _x2;
			REGISTER oski_value_t _x3;
			REGISTER oski_value_t _x4;

			VAL_ASSIGN( _x0, xp[0] );
			VAL_ASSIGN( _x1, xp[1*incx] );
			VAL_ASSIGN( _x2, xp[2*incx] );
			VAL_ASSIGN( _x3, xp[3*incx] );
			VAL_ASSIGN( _x4, xp[4*incx] );

			VAL_MSUB( _b0, vp[0], _x0 );
			VAL_MSUB( _b0, vp[1], _x1 );
			VAL_MSUB( _b0, vp[2], _x2 );
			VAL_MSUB( _b0, vp[3], _x3 );
			VAL_MSUB( _b0, vp[4], _x4 );
		} /* K */

		VAL_DIVEQ( _b0, dp[0*1+0] );
		VAL_ASSIGN( bp[0], _b0 );
	} /* I */
} /* MBCSR_MatTrisolve_Upper_v1_aX_xsX */

/**
 *  \brief Exported module wrapper for the \f$1\times 5\f$
 *  implementation of the sparse triangular solve operation,
 *  where the matrix is upper triangular.
 */
static void
MatTrisolve_Upper( const oski_submatMBCSR_t* T,
	oski_value_t alpha, oski_vecview_t x )
{
	assert( T != NULL );
	assert( T->r == 1 );
	assert( T->c == 5 );
	assert( x != INVALID_VEC );

	if( x->rowinc == 1 ) {
		oski_index_t j;
		oski_value_t* xp;
		for( j = 0, xp = x->val; j < x->num_cols; j++, xp += x->colinc )
			MBCSR_MatTrisolve_Upper_v1_aX_xs1( T->num_block_rows, T->offset,
				T->bptr, T->bind, T->bval, T->bdiag, alpha, xp );
	} else { /* x has general (non-unit) stride */
		oski_index_t j;
		oski_value_t* xp;
		for( j = 0, xp = x->val; j < x->num_cols; j++, xp += x->colinc )
			MBCSR_MatTrisolve_Upper_v1_aX_xsX( T->num_block_rows, T->offset,
				T->bptr, T->bind, T->bval, T->bdiag, alpha, xp, x->rowinc );
	}
}

#if !IS_VAL_COMPLEX
	/** Synonym for pure real version */
	#define MBCSR_MatConjTrisolve_Upper_v1_aX_xs1 MBCSR_MatTrisolve_Upper_v1_aX_xs1
#else /* IS_VAL_COMPLEX */

#if defined(DO_NAME_MANGLING)
	/** Mangled name for MBCSR_MatConjTrisolve_Upper_v1_aX_xs1. */
	#define MBCSR_MatConjTrisolve_Upper_v1_aX_xs1 \
		MANGLE_MOD_(MBCSR_MatConjTrisolve_Upper_v1_aX_xs1_1x5)
#endif

#endif
#if IS_VAL_COMPLEX
/**
 *  \brief The \f$1\times 5\f$ MBCSR implementation
 *  of \f$\bar{U}^{-1}\cdot b\f$, where x has unit-stride.
 */
void
MBCSR_MatConjTrisolve_Upper_v1_aX_xs1(
	oski_index_t M, oski_index_t d0,
	const oski_index_t* restrict ptr, const oski_index_t* restrict ind,
	const oski_value_t* restrict val, const oski_value_t* restrict diag,
	oski_value_t alpha, oski_value_t* restrict x )
{
	oski_index_t I;
	oski_value_t* bp = x + (d0 + (M-1)*1);
	const oski_value_t* dp = diag + (M-1)*1*1;

	for( I = M; I != 0; I--, bp -= 1, dp -= 1*1 )
	{
		oski_index_t K;
		REGISTER oski_value_t _b0;
		const oski_value_t* vp = val + ptr[I-1]*1*5;

		VAL_MUL( _b0, alpha, bp[0] );

		for( K = ptr[I-1]; K < ptr[I]; K++, vp += 1*5 )
		{
			oski_index_t j0 = ind[K];
			const oski_value_t* xp = x + j0;
			REGISTER oski_value_t _x0;
			REGISTER oski_value_t _x1;
			REGISTER oski_value_t _x2;
			REGISTER oski_value_t _x3;
			REGISTER oski_value_t _x4;

			VAL_ASSIGN( _x0, xp[0] );
			VAL_ASSIGN( _x1, xp[1] );
			VAL_ASSIGN( _x2, xp[2] );
			VAL_ASSIGN( _x3, xp[3] );
			VAL_ASSIGN( _x4, xp[4] );

			VAL_MSUB_CONJ( _b0, vp[0], _x0 );
			VAL_MSUB_CONJ( _b0, vp[1], _x1 );
			VAL_MSUB_CONJ( _b0, vp[2], _x2 );
			VAL_MSUB_CONJ( _b0, vp[3], _x3 );
			VAL_MSUB_CONJ( _b0, vp[4], _x4 );
		} /* K */

		VAL_DIVEQ_CONJ( _b0, dp[0*1+0] );
		VAL_ASSIGN( bp[0], _b0 );
	} /* I */
} /* MBCSR_MatConjTrisolve_Upper_v1_aX_xs1 */

#endif /* IS_VAL_COMPLEX */
#if !IS_VAL_COMPLEX
	/** Synonym for pure real version */
	#define MBCSR_MatConjTrisolve_Upper_v1_aX_xsX MBCSR_MatTrisolve_Upper_v1_aX_xsX
#else /* IS_VAL_COMPLEX */

#if defined(DO_NAME_MANGLING)
	/** Mangled name for MBCSR_MatConjTrisolve_Upper_v1_aX_xsX. */
	#define MBCSR_MatConjTrisolve_Upper_v1_aX_xsX \
		MANGLE_MOD_(MBCSR_MatConjTrisolve_Upper_v1_aX_xsX_1x5)
#endif

#endif
#if IS_VAL_COMPLEX
/**
 *  \brief The \f$1\times 5\f$ MBCSR implementation
 *  of \f$\bar{U}^{-1}\cdot b\f$, where x has general-stride.
 */
void
MBCSR_MatConjTrisolve_Upper_v1_aX_xsX(
	oski_index_t M, oski_index_t d0,
	const oski_index_t* restrict ptr, const oski_index_t* restrict ind,
	const oski_value_t* restrict val, const oski_value_t* restrict diag,
	oski_value_t alpha, oski_value_t* restrict x, oski_index_t incx )
{
	oski_index_t I;
	oski_value_t* bp = x + (d0 + (M-1)*1)*incx;
	const oski_value_t* dp = diag + (M-1)*1*1;

	for( I = M; I != 0; I--, bp -= 1*incx, dp -= 1*1 )
	{
		oski_index_t K;
		REGISTER oski_value_t _b0;
		const oski_value_t* vp = val + ptr[I-1]*1*5;

		VAL_MUL( _b0, alpha, bp[0] );

		for( K = ptr[I-1]; K < ptr[I]; K++, vp += 1*5 )
		{
			oski_index_t j0 = ind[K];
			const oski_value_t* xp = x + j0*incx;
			REGISTER oski_value_t _x0;
			REGISTER oski_value_t _x1;
			REGISTER oski_value_t _x2;
			REGISTER oski_value_t _x3;
			REGISTER oski_value_t _x4;

			VAL_ASSIGN( _x0, xp[0] );
			VAL_ASSIGN( _x1, xp[1*incx] );
			VAL_ASSIGN( _x2, xp[2*incx] );
			VAL_ASSIGN( _x3, xp[3*incx] );
			VAL_ASSIGN( _x4, xp[4*incx] );

			VAL_MSUB_CONJ( _b0, vp[0], _x0 );
			VAL_MSUB_CONJ( _b0, vp[1], _x1 );
			VAL_MSUB_CONJ( _b0, vp[2], _x2 );
			VAL_MSUB_CONJ( _b0, vp[3], _x3 );
			VAL_MSUB_CONJ( _b0, vp[4], _x4 );
		} /* K */

		VAL_DIVEQ_CONJ( _b0, dp[0*1+0] );
		VAL_ASSIGN( bp[0], _b0 );
	} /* I */
} /* MBCSR_MatConjTrisolve_Upper_v1_aX_xsX */

#endif /* IS_VAL_COMPLEX */
/**
 *  \brief Exported module wrapper for the \f$1\times 5\f$
 *  implementation of the sparse triangular solve operation,
 *  where the matrix is upper triangular.
 */
static void
MatConjTrisolve_Upper( const oski_submatMBCSR_t* T,
	oski_value_t alpha, oski_vecview_t x )
{
	assert( T != NULL );
	assert( T->r == 1 );
	assert( T->c == 5 );
	assert( x != INVALID_VEC );

	if( x->rowinc == 1 ) {
		oski_index_t j;
		oski_value_t* xp;
		for( j = 0, xp = x->val; j < x->num_cols; j++, xp += x->colinc )
			MBCSR_MatConjTrisolve_Upper_v1_aX_xs1( T->num_block_rows, T->offset,
				T->bptr, T->bind, T->bval, T->bdiag, alpha, xp );
	} else { /* x has general (non-unit) stride */
		oski_index_t j;
		oski_value_t* xp;
		for( j = 0, xp = x->val; j < x->num_cols; j++, xp += x->colinc )
			MBCSR_MatConjTrisolve_Upper_v1_aX_xsX( T->num_block_rows, T->offset,
				T->bptr, T->bind, T->bval, T->bdiag, alpha, xp, x->rowinc );
	}
}

#if defined(DO_NAME_MANGLING)
	/** Mangled name for MBCSR_MatTransTrisolve_Upper_v1_aX_xs1. */
	#define MBCSR_MatTransTrisolve_Upper_v1_aX_xs1 \
		MANGLE_MOD_(MBCSR_MatTransTrisolve_Upper_v1_aX_xs1_1x5)
#endif

/**
 *  \brief The \f$1\times 5\f$ MBCSR implementation
 *  of \f$U^{-T}\cdot b\f$, where x has unit-stride.
 */
void
MBCSR_MatTransTrisolve_Upper_v1_aX_xs1(
	oski_index_t M, oski_index_t d0,
	const oski_index_t* restrict ptr, const oski_index_t* restrict ind,
	const oski_value_t* restrict val, const oski_value_t* restrict diag,
	oski_value_t alpha, oski_value_t* restrict x )
{
	oski_index_t I;
	oski_value_t* bp = x + d0;
	const oski_value_t* dp = diag;

	{
		oski_index_t m = M * 1;
		oski_index_t incx = 1;
		BLAS_xSCAL( &m, &alpha, x, &incx );
	}

	for( I = 1; I != M+1; I++, bp += 1, dp += 1*1 )
	{
		oski_index_t K;
		REGISTER oski_value_t _b0;
		const oski_value_t* vp = val + ptr[I-1]*1*5;

		VAL_ASSIGN( _b0, bp[0] );

		VAL_DIVEQ( _b0, dp[0*1+0] );

		for( K = ptr[I-1]; K < ptr[I]; K++, vp += 1*5 )
		{
			oski_index_t j0 = ind[K];
			oski_value_t* xp = x + j0;
			REGISTER oski_value_t _x0;
			REGISTER oski_value_t _x1;
			REGISTER oski_value_t _x2;
			REGISTER oski_value_t _x3;
			REGISTER oski_value_t _x4;

			VAL_ASSIGN( _x0, xp[0] );
			VAL_ASSIGN( _x1, xp[1] );
			VAL_ASSIGN( _x2, xp[2] );
			VAL_ASSIGN( _x3, xp[3] );
			VAL_ASSIGN( _x4, xp[4] );

			VAL_MSUB( _x0, vp[0], _b0 );
			VAL_MSUB( _x1, vp[1], _b0 );
			VAL_MSUB( _x2, vp[2], _b0 );
			VAL_MSUB( _x3, vp[3], _b0 );
			VAL_MSUB( _x4, vp[4], _b0 );
			VAL_ASSIGN( xp[0], _x0 );
			VAL_ASSIGN( xp[1], _x1 );
			VAL_ASSIGN( xp[2], _x2 );
			VAL_ASSIGN( xp[3], _x3 );
			VAL_ASSIGN( xp[4], _x4 );
		} /* K */

		VAL_ASSIGN( bp[0], _b0 );
	} /* I */
} /* MBCSR_MatTransTrisolve_Upper_v1_aX_xs1 */

#if defined(DO_NAME_MANGLING)
	/** Mangled name for MBCSR_MatTransTrisolve_Upper_v1_aX_xsX. */
	#define MBCSR_MatTransTrisolve_Upper_v1_aX_xsX \
		MANGLE_MOD_(MBCSR_MatTransTrisolve_Upper_v1_aX_xsX_1x5)
#endif

/**
 *  \brief The \f$1\times 5\f$ MBCSR implementation
 *  of \f$U^{-T}\cdot b\f$, where x has general-stride.
 */
void
MBCSR_MatTransTrisolve_Upper_v1_aX_xsX(
	oski_index_t M, oski_index_t d0,
	const oski_index_t* restrict ptr, const oski_index_t* restrict ind,
	const oski_value_t* restrict val, const oski_value_t* restrict diag,
	oski_value_t alpha, oski_value_t* restrict x, oski_index_t incx )
{
	oski_index_t I;
	oski_value_t* bp = x + d0*incx;
	const oski_value_t* dp = diag;

	{
		oski_index_t m = M * 1;
		BLAS_xSCAL( &m, &alpha, x, &incx );
	}

	for( I = 1; I != M+1; I++, bp += 1*incx, dp += 1*1 )
	{
		oski_index_t K;
		REGISTER oski_value_t _b0;
		const oski_value_t* vp = val + ptr[I-1]*1*5;

		VAL_ASSIGN( _b0, bp[0] );

		VAL_DIVEQ( _b0, dp[0*1+0] );

		for( K = ptr[I-1]; K < ptr[I]; K++, vp += 1*5 )
		{
			oski_index_t j0 = ind[K];
			oski_value_t* xp = x + j0*incx;
			REGISTER oski_value_t _x0;
			REGISTER oski_value_t _x1;
			REGISTER oski_value_t _x2;
			REGISTER oski_value_t _x3;
			REGISTER oski_value_t _x4;

			VAL_ASSIGN( _x0, xp[0] );
			VAL_ASSIGN( _x1, xp[1*incx] );
			VAL_ASSIGN( _x2, xp[2*incx] );
			VAL_ASSIGN( _x3, xp[3*incx] );
			VAL_ASSIGN( _x4, xp[4*incx] );

			VAL_MSUB( _x0, vp[0], _b0 );
			VAL_MSUB( _x1, vp[1], _b0 );
			VAL_MSUB( _x2, vp[2], _b0 );
			VAL_MSUB( _x3, vp[3], _b0 );
			VAL_MSUB( _x4, vp[4], _b0 );
			VAL_ASSIGN( xp[0], _x0 );
			VAL_ASSIGN( xp[1*incx], _x1 );
			VAL_ASSIGN( xp[2*incx], _x2 );
			VAL_ASSIGN( xp[3*incx], _x3 );
			VAL_ASSIGN( xp[4*incx], _x4 );
		} /* K */

		VAL_ASSIGN( bp[0], _b0 );
	} /* I */
} /* MBCSR_MatTransTrisolve_Upper_v1_aX_xsX */

/**
 *  \brief Exported module wrapper for the \f$1\times 5\f$
 *  implementation of the sparse triangular solve operation,
 *  where the matrix is upper triangular.
 */
static void
MatTransTrisolve_Upper( const oski_submatMBCSR_t* T,
	oski_value_t alpha, oski_vecview_t x )
{
	assert( T != NULL );
	assert( T->r == 1 );
	assert( T->c == 5 );
	assert( x != INVALID_VEC );

	if( x->rowinc == 1 ) {
		oski_index_t j;
		oski_value_t* xp;
		for( j = 0, xp = x->val; j < x->num_cols; j++, xp += x->colinc )
			MBCSR_MatTransTrisolve_Upper_v1_aX_xs1( T->num_block_rows, T->offset,
				T->bptr, T->bind, T->bval, T->bdiag, alpha, xp );
	} else { /* x has general (non-unit) stride */
		oski_index_t j;
		oski_value_t* xp;
		for( j = 0, xp = x->val; j < x->num_cols; j++, xp += x->colinc )
			MBCSR_MatTransTrisolve_Upper_v1_aX_xsX( T->num_block_rows, T->offset,
				T->bptr, T->bind, T->bval, T->bdiag, alpha, xp, x->rowinc );
	}
}

#if !IS_VAL_COMPLEX
	/** Synonym for pure real version */
	#define MBCSR_MatHermTrisolve_Upper_v1_aX_xs1 MBCSR_MatTransTrisolve_Upper_v1_aX_xs1
#else /* IS_VAL_COMPLEX */

#if defined(DO_NAME_MANGLING)
	/** Mangled name for MBCSR_MatHermTrisolve_Upper_v1_aX_xs1. */
	#define MBCSR_MatHermTrisolve_Upper_v1_aX_xs1 \
		MANGLE_MOD_(MBCSR_MatHermTrisolve_Upper_v1_aX_xs1_1x5)
#endif

#endif
#if IS_VAL_COMPLEX
/**
 *  \brief The \f$1\times 5\f$ MBCSR implementation
 *  of \f$\bar{U}^{-T}\cdot b\f$, where x has unit-stride.
 */
void
MBCSR_MatHermTrisolve_Upper_v1_aX_xs1(
	oski_index_t M, oski_index_t d0,
	const oski_index_t* restrict ptr, const oski_index_t* restrict ind,
	const oski_value_t* restrict val, const oski_value_t* restrict diag,
	oski_value_t alpha, oski_value_t* restrict x )
{
	oski_index_t I;
	oski_value_t* bp = x + d0;
	const oski_value_t* dp = diag;

	{
		oski_index_t m = M * 1;
		oski_index_t incx = 1;
		BLAS_xSCAL( &m, &alpha, x, &incx );
	}

	for( I = 1; I != M+1; I++, bp += 1, dp += 1*1 )
	{
		oski_index_t K;
		REGISTER oski_value_t _b0;
		const oski_value_t* vp = val + ptr[I-1]*1*5;

		VAL_ASSIGN( _b0, bp[0] );

		VAL_DIVEQ_CONJ( _b0, dp[0*1+0] );

		for( K = ptr[I-1]; K < ptr[I]; K++, vp += 1*5 )
		{
			oski_index_t j0 = ind[K];
			oski_value_t* xp = x + j0;
			REGISTER oski_value_t _x0;
			REGISTER oski_value_t _x1;
			REGISTER oski_value_t _x2;
			REGISTER oski_value_t _x3;
			REGISTER oski_value_t _x4;

			VAL_ASSIGN( _x0, xp[0] );
			VAL_ASSIGN( _x1, xp[1] );
			VAL_ASSIGN( _x2, xp[2] );
			VAL_ASSIGN( _x3, xp[3] );
			VAL_ASSIGN( _x4, xp[4] );

			VAL_MSUB_CONJ( _x0, vp[0], _b0 );
			VAL_MSUB_CONJ( _x1, vp[1], _b0 );
			VAL_MSUB_CONJ( _x2, vp[2], _b0 );
			VAL_MSUB_CONJ( _x3, vp[3], _b0 );
			VAL_MSUB_CONJ( _x4, vp[4], _b0 );
			VAL_ASSIGN( xp[0], _x0 );
			VAL_ASSIGN( xp[1], _x1 );
			VAL_ASSIGN( xp[2], _x2 );
			VAL_ASSIGN( xp[3], _x3 );
			VAL_ASSIGN( xp[4], _x4 );
		} /* K */

		VAL_ASSIGN( bp[0], _b0 );
	} /* I */
} /* MBCSR_MatHermTrisolve_Upper_v1_aX_xs1 */

#endif /* IS_VAL_COMPLEX */
#if !IS_VAL_COMPLEX
	/** Synonym for pure real version */
	#define MBCSR_MatHermTrisolve_Upper_v1_aX_xsX MBCSR_MatTransTrisolve_Upper_v1_aX_xsX
#else /* IS_VAL_COMPLEX */

#if defined(DO_NAME_MANGLING)
	/** Mangled name for MBCSR_MatHermTrisolve_Upper_v1_aX_xsX. */
	#define MBCSR_MatHermTrisolve_Upper_v1_aX_xsX \
		MANGLE_MOD_(MBCSR_MatHermTrisolve_Upper_v1_aX_xsX_1x5)
#endif

#endif
#if IS_VAL_COMPLEX
/**
 *  \brief The \f$1\times 5\f$ MBCSR implementation
 *  of \f$\bar{U}^{-T}\cdot b\f$, where x has general-stride.
 */
void
MBCSR_MatHermTrisolve_Upper_v1_aX_xsX(
	oski_index_t M, oski_index_t d0,
	const oski_index_t* restrict ptr, const oski_index_t* restrict ind,
	const oski_value_t* restrict val, const oski_value_t* restrict diag,
	oski_value_t alpha, oski_value_t* restrict x, oski_index_t incx )
{
	oski_index_t I;
	oski_value_t* bp = x + d0*incx;
	const oski_value_t* dp = diag;

	{
		oski_index_t m = M * 1;
		BLAS_xSCAL( &m, &alpha, x, &incx );
	}

	for( I = 1; I != M+1; I++, bp += 1*incx, dp += 1*1 )
	{
		oski_index_t K;
		REGISTER oski_value_t _b0;
		const oski_value_t* vp = val + ptr[I-1]*1*5;

		VAL_ASSIGN( _b0, bp[0] );

		VAL_DIVEQ_CONJ( _b0, dp[0*1+0] );

		for( K = ptr[I-1]; K < ptr[I]; K++, vp += 1*5 )
		{
			oski_index_t j0 = ind[K];
			oski_value_t* xp = x + j0*incx;
			REGISTER oski_value_t _x0;
			REGISTER oski_value_t _x1;
			REGISTER oski_value_t _x2;
			REGISTER oski_value_t _x3;
			REGISTER oski_value_t _x4;

			VAL_ASSIGN( _x0, xp[0] );
			VAL_ASSIGN( _x1, xp[1*incx] );
			VAL_ASSIGN( _x2, xp[2*incx] );
			VAL_ASSIGN( _x3, xp[3*incx] );
			VAL_ASSIGN( _x4, xp[4*incx] );

			VAL_MSUB_CONJ( _x0, vp[0], _b0 );
			VAL_MSUB_CONJ( _x1, vp[1], _b0 );
			VAL_MSUB_CONJ( _x2, vp[2], _b0 );
			VAL_MSUB_CONJ( _x3, vp[3], _b0 );
			VAL_MSUB_CONJ( _x4, vp[4], _b0 );
			VAL_ASSIGN( xp[0], _x0 );
			VAL_ASSIGN( xp[1*incx], _x1 );
			VAL_ASSIGN( xp[2*incx], _x2 );
			VAL_ASSIGN( xp[3*incx], _x3 );
			VAL_ASSIGN( xp[4*incx], _x4 );
		} /* K */

		VAL_ASSIGN( bp[0], _b0 );
	} /* I */
} /* MBCSR_MatHermTrisolve_Upper_v1_aX_xsX */

#endif /* IS_VAL_COMPLEX */
/**
 *  \brief Exported module wrapper for the \f$1\times 5\f$
 *  implementation of the sparse triangular solve operation,
 *  where the matrix is upper triangular.
 */
static void
MatHermTrisolve_Upper( const oski_submatMBCSR_t* T,
	oski_value_t alpha, oski_vecview_t x )
{
	assert( T != NULL );
	assert( T->r == 1 );
	assert( T->c == 5 );
	assert( x != INVALID_VEC );

	if( x->rowinc == 1 ) {
		oski_index_t j;
		oski_value_t* xp;
		for( j = 0, xp = x->val; j < x->num_cols; j++, xp += x->colinc )
			MBCSR_MatHermTrisolve_Upper_v1_aX_xs1( T->num_block_rows, T->offset,
				T->bptr, T->bind, T->bval, T->bdiag, alpha, xp );
	} else { /* x has general (non-unit) stride */
		oski_index_t j;
		oski_value_t* xp;
		for( j = 0, xp = x->val; j < x->num_cols; j++, xp += x->colinc )
			MBCSR_MatHermTrisolve_Upper_v1_aX_xsX( T->num_block_rows, T->offset,
				T->bptr, T->bind, T->bval, T->bdiag, alpha, xp, x->rowinc );
	}
}


#if defined(DO_NAME_MANGLING)
/** Mangled name for primary exportable symbol */
#define SubmatReprTrisolve MANGLE_MOD_(SubmatReprTrisolve_1x5)
#endif

/**
 *  \brief Entry point to the 1x5 kernel that implements
 *  simultaneous multiplication by sparse \f$A\f$ and
 *  \f$\mathrm{op}(A)\f$.
 */
int
SubmatReprTrisolve( const oski_submatMBCSR_t* T, int is_lower,
	oski_matop_t opT, oski_value_t alpha, oski_vecview_t x )
{
	int err = 0;
	if( is_lower )
		switch( opT ) {
			case OP_NORMAL:
				MatTrisolve_Lower( T, alpha, x );
				break;
			case OP_TRANS:
				MatTransTrisolve_Lower( T, alpha, x );
				break;
			case OP_CONJ:
				MatConjTrisolve_Lower( T, alpha, x );
				break;
			case OP_CONJ_TRANS:
				MatHermTrisolve_Lower( T, alpha, x );
				break;
			default:
				err = ERR_NOT_IMPLEMENTED;
		}
	else
		switch( opT ) {
			case OP_NORMAL:
				MatTrisolve_Upper( T, alpha, x );
				break;
			case OP_TRANS:
				MatTransTrisolve_Upper( T, alpha, x );
				break;
			case OP_CONJ:
				MatConjTrisolve_Upper( T, alpha, x );
				break;
			case OP_CONJ_TRANS:
				MatHermTrisolve_Upper( T, alpha, x );
				break;
			default:
				err = ERR_NOT_IMPLEMENTED;
		}
	return err;
}

/* eof */
