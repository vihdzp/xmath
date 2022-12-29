// Implements and tests various matrix algorithms.

use crate::traits::*;
use xmath_traits::*;

/// Transposes a matrix and writes the output in a new matrix. No size check is
/// performed.
///
/// The inputs are imagined as having zeros in all nonexistent entries, and
/// the output is trimmed to fit.
pub fn transpose_trim<M: Matrix, N: Matrix<Item = M::Item>>(m: &M) -> N
where
    M::Item: Ring,
{
    N::from_fn(m.width(), m.height(), |i, j| m.coeff_or_zero(j, i))
}

/// Does a compile-time size check and then calls [`transpose_trim`].
///
/// Various types have their own `transpose` convenience method which
/// autocompletes the matrix types, at a slight cost to generality.
///
/// ## Panics
///
/// This function will panic if any of these are false:
///
/// - The height of `M` equals the width of `N`.
/// - The width of `M` equals the height of `N`.
pub fn transpose_gen<M: Matrix, N: Matrix<Item = M::Item>>(m: &M) -> N
where
    M::Item: Ring,
{
    if M::SIZE == N::SIZE.swap() {
        transpose_trim(m)
    } else {
        panic!(
            "incompatible matrix sizes for transposition M → N (M: {}, N: {})",
            M::SIZE,
            N::SIZE,
        )
    }
}

/// Transposes a matrix in place. No size check is performed.
///
/// The inputs are imagined as having zeros in all nonexistent entries, and
/// the output is trimmed to fit.
///
/// TODO: this can probably be made more efficient.
pub fn transpose_trim_mut<M: Matrix>(m: &mut M)
where
    M::Item: Ring,
{
    *m = transpose_trim(m);
}

/// Does a compile-time size check and then calls [`transpose_trim_mut`].
///
/// Various types have their own `transpose_mut` convenience method which
/// autocompletes the matrix types, at a slight cost to generality.
///
/// ## Panics
///
/// This function will panic if the height and width of `M` are different.
pub fn transpose_gen_mut<M: Matrix>(m: &mut M)
where
    M::Item: Ring,
{
    if M::SIZE == M::SIZE.swap() {
        transpose_trim_mut(m)
    } else {
        panic!("incompatible matrix size for transposition {}", M::SIZE)
    }
}

/// Computes the addition of two matrices and writes the output in a new
/// matrix. No size check is performed.
///
/// The inputs are imagined as having zeros in all nonexistent entries, and
/// the output is trimmed to fit.
pub fn madd_trim<M: Matrix, N: Matrix<Item = M::Item>, K: Matrix<Item = M::Item>>(m: &M, n: &N) -> K
where
    M::Item: Ring,
{
    K::from_fn(
        m.height().max(n.height()),
        m.width().max(n.width()),
        |i, j| {
            if let (Some(x), Some(y)) = (m.coeff_ref(i, j), n.coeff_ref(i, j)) {
                x.add(y)
            } else {
                M::Item::zero()
            }
        },
    )
}

/// Does a compile-time size check and then calls [`madd_trim`].
///
/// Various types have their own `madd` convenience method which
/// autocompletes the matrix types, at a slight cost to generality.
///
/// ## Panics
///
/// This function will panic if the sizes of `M`, `N`, `K` are not equal.
pub fn madd_gen<M: Matrix, N: Matrix<Item = M::Item>, K: Matrix<Item = M::Item>>(m: &M, n: &N) -> K
where
    M::Item: Ring,
{
    if M::SIZE == N::SIZE && N::SIZE == K::SIZE {
        madd_trim(m, n)
    } else {
        panic!(
            "incompatible matrix sizes for addition M + N → K (M: {}, N: {}, K: {})",
            M::SIZE,
            N::SIZE,
            K::SIZE,
        )
    }
}

/// Computes the product of two matrices and writes the output in a new matrix.
/// No size check is performed.
///
/// The inputs are imagined as having zeros in all nonexistent entries, and the
/// output is trimmed to fit.
pub fn mmul_trim<M: Matrix, N: Matrix<Item = M::Item>, K: Matrix<Item = M::Item>>(m: &M, n: &N) -> K
where
    M::Item: Ring,
{
    K::from_fn(m.height(), n.width(), |i, j| {
        let mut z = M::Item::zero();

        for k in 0..m.row_support(i).min(n.col_support(j)) {
            if let (Some(x), Some(y)) = (m.coeff_ref(i, k), n.coeff_ref(k, j)) {
                z.add_mut(&x.mul(y));
            }
        }

        z
    })
}

/// Does a compile-time size check and then calls [`mmul_trim`].
///
/// Various types have their own `mmul` convenience method which autocompletes
/// the matrix types, at a slight cost to generality.
///
/// ## Panics
///
/// This function will panic if any of these are false:
///
/// - The height of `M` equals the height of `K`.
/// - The width of `N` equals the width of `K`.
/// - The width of `M` equals the height of `N`.
pub fn mmul_gen<M: Matrix, N: Matrix<Item = M::Item>, K: Matrix<Item = M::Item>>(m: &M, n: &N) -> K
where
    M::Item: Ring,
{
    if M::SIZE.height() == K::SIZE.height()
        && N::SIZE.width() == K::SIZE.width()
        && M::SIZE.width() == N::SIZE.height()
    {
        mmul_trim(m, n)
    } else {
        panic!(
            "incompatible matrix sizes for multiplication M × N → K (M: {}, N: {}, K: {})",
            M::SIZE,
            N::SIZE,
            K::SIZE,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::aliases::Transpose;
    use crate::data::*;
    use crate::{matrix_dyn, matrix_mn};
    use std::num::Wrapping;

    /// Tests transposition.
    #[test]
    fn transpose_1() {
        let m1: MatrixDyn<Wu8> = matrix_dyn!(
            Wrapping(1), Wrapping(2);
            Wrapping(3), Wrapping(4), Wrapping(5)
        );

        let m2: MatrixDyn<Wu8> = matrix_dyn!(
            Wrapping(1), Wrapping(3);
            Wrapping(2), Wrapping(4);
            Wrapping(0), Wrapping(5)
        );

        assert_eq!(transpose_gen::<_, MatrixDyn<_>>(&m1), m2);
    }

    /// Tests the bound checks for transposition.
    #[test]
    #[should_panic = "incompatible matrix sizes for transposition"]
    fn transpose_2() {
        let m1: MatrixMN<Wu8, 1, 2> = matrix_mn!(Wrapping(1), Wrapping(2));
        transpose_gen::<_, MatrixMN<Wu8, 1, 2>>(&m1);
    }

    /// Tests addition of statically-sized matrices.
    #[test]
    fn add_1() {
        let m1: MatrixMN<Wu8, 2, 3> = matrix_mn!(
            Wrapping(1), Wrapping(2), Wrapping(3);
            Wrapping(4), Wrapping(5), Wrapping(6)
        );

        let m2: MatrixMN<Wu8, 2, 3> = matrix_mn!(
            Wrapping(7), Wrapping(8), Wrapping(9);
            Wrapping(10), Wrapping(11), Wrapping(12)
        );

        let m3: MatrixMN<Wu8, 2, 3> = matrix_mn!(
            Wrapping(8), Wrapping(10), Wrapping(12);
            Wrapping(14), Wrapping(16), Wrapping(18)
        );

        assert_eq!(m1.madd(&m2), m3);
    }

    /// Tests addition of dynamically-sized matrices.
    #[test]
    fn add_2() {
        let m1: MatrixDyn<Wu8> = matrix_dyn!(
            Wrapping(1), Wrapping(2), Wrapping(3);
            Wrapping(4), Wrapping(5), Wrapping(6)
        );

        let m2: Transpose<MatrixDyn<Wu8>> = Transpose(matrix_dyn!(
            Wrapping(7), Wrapping(10);
            Wrapping(8), Wrapping(11);
            Wrapping(9), Wrapping(12)
        ));

        let m3: MatrixDyn<Wu8> = matrix_dyn!(
            Wrapping(8), Wrapping(10), Wrapping(12);
            Wrapping(14), Wrapping(16), Wrapping(18)
        );

        assert_eq!(madd_gen::<_, _, MatrixDyn<_>>(&m1, &m2), m3);
    }

    /// Tests the bound checks for addition.
    #[test]
    #[should_panic = "incompatible matrix sizes for addition"]
    fn add_3() {
        let m1: MatrixMN<Wu8, 2, 3> = matrix_mn!(
            Wrapping(1), Wrapping(2), Wrapping(3);
            Wrapping(4), Wrapping(5), Wrapping(6)
        );

        let m2: MatrixMN<Wu8, 2, 2> = matrix_mn!(
            Wrapping(7), Wrapping(8);
            Wrapping(9), Wrapping(10)
        );

        madd_gen::<_, _, MatrixMN<Wu8, 2, 3>>(&m1, &m2);
    }

    /// Tests multiplication of statically-sized matrices.
    #[test]
    fn mul_1() {
        let m1: MatrixMN<Wu8, 2, 3> = matrix_mn!(
            Wrapping(1), Wrapping(2), Wrapping(3);
            Wrapping(4), Wrapping(5), Wrapping(6)
        );

        let m2: MatrixMN<Wu8, 3, 2> = matrix_mn!(
            Wrapping(7), Wrapping(10);
            Wrapping(8), Wrapping(11);
            Wrapping(9), Wrapping(12)
        );

        let m3: MatrixMN<Wu8, 2, 2> = matrix_mn!(
            Wrapping(50), Wrapping(68);
            Wrapping(122), Wrapping(167)
        );

        assert_eq!(m1.mmul(&m2), m3);
    }

    /// Tests multiplication of dynamically-sized matrices.
    #[test]
    fn mul_2() {
        let m1: MatrixDyn<Wu8> = matrix_dyn!(
            Wrapping(1), Wrapping(2), Wrapping(3);
            Wrapping(4), Wrapping(5), Wrapping(6)
        );

        let m2: Transpose<MatrixDyn<Wu8>> = Transpose(matrix_dyn!(
            Wrapping(7), Wrapping(8), Wrapping(9);
            Wrapping(10), Wrapping(11), Wrapping(12)
        ));

        let m3: MatrixDyn<Wu8> = matrix_dyn!(
            Wrapping(50), Wrapping(68);
            Wrapping(122), Wrapping(167)
        );

        assert_eq!(mmul_gen::<_, _, MatrixDyn<_>>(&m1, &m2), m3);
    }

    /// Tests the bound checks for multiplication.
    #[test]
    #[should_panic = "incompatible matrix sizes for multiplication"]
    fn mul_3() {
        let m1: MatrixMN<Wu8, 2, 3> = matrix_mn!(
            Wrapping(1), Wrapping(2), Wrapping(3);
            Wrapping(4), Wrapping(5), Wrapping(6)
        );

        let m2: MatrixMN<Wu8, 2, 2> = matrix_mn!(
            Wrapping(7), Wrapping(8);
            Wrapping(9), Wrapping(10)
        );

        mmul_gen::<_, _, MatrixMN<Wu8, 2, 3>>(&m1, &m2);
    }
}
