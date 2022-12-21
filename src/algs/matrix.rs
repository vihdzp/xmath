use crate::traits::{
    basic::*,
    matrix::{size_height, size_width, Matrix},
};

/// Computes the addition of two matrices and writes the output in a new
/// matrix. No size check is performed.
///
/// The inputs are imagined as having zeros in all nonexistent entries, and
/// the output is trimmed to fit.
pub fn madd_trim<
    M: Matrix,
    N: Matrix<Item = M::Item>,
    K: Matrix<Item = M::Item>,
>(
    m: &M,
    n: &N,
) -> K
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
/// This function will panic if any of these are false:
///
/// - The heights of `M`, `N`, `K` are equal.
/// - The widths of `M`, `N`, `K` are equal.
pub fn madd_gen<
    M: Matrix,
    N: Matrix<Item = M::Item>,
    K: Matrix<Item = M::Item>,
>(
    m: &M,
    n: &N,
) -> K
where
    M::Item: Ring,
{
    if size_height::<M>() == size_height::<N>()
        && size_height::<M>() == size_height::<K>()
        && size_width::<M>() == size_width::<N>()
        && size_width::<M>() == size_width::<K>()
    {
        madd_trim(m, n)
    } else {
        panic!("{}", crate::SIZE_MISMATCH)
    }
}

/// Computes the product of two matrices and writes the output in a new matrix.
/// No size check is performed.
///
/// The inputs are imagined as having zeros in all nonexistent entries, and the
/// output is trimmed to fit.
pub fn mmul_trim<
    M: Matrix,
    N: Matrix<Item = M::Item>,
    K: Matrix<Item = M::Item>,
>(
    m: &M,
    n: &N,
) -> K
where
    M::Item: Ring,
{
    K::from_fn(m.height(), n.width(), |i, j| {
        let mut z = M::Item::zero();

        for k in 0..m.row_support(i).min(n.col_support(j)) {
            if let (Some(x), Some(y)) = (m.coeff_ref(i, k), n.coeff_ref(i, j)) {
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
/// - The width of `M` equals the height of `N`.
/// - The width of `N` equals the width of `K`.
pub fn mmul_gen<
    M: Matrix,
    N: Matrix<Item = M::Item>,
    K: Matrix<Item = M::Item>,
>(
    m: &M,
    n: &N,
) -> K
where
    M::Item: Ring,
{
    if size_height::<M>() == size_height::<K>()
        && size_width::<N>() == size_width::<K>()
        && size_width::<M>() == size_height::<N>()
    {
        mmul_trim(m, n)
    } else {
        panic!("{}", crate::SIZE_MISMATCH)
    }
}
