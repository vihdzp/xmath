use crate::traits::{matrix::Matrix, basic::*};

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
            if let (Some(x), Some(y)) =
                (m.coeff_ref((i, j)), n.coeff_ref((i, j)))
            {
                x.add(y)
            } else {
                M::Item::zero()
            }
        },
    )
}

/// Does a compile-time size check and then calls [`Matrix::madd_trim`].
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
    N: Matrix<
        Item = M::Item,
        HeightType = M::HeightType,
        WidthType = M::WidthType,
    >,
    K: Matrix<
        Item = M::Item,
        HeightType = M::HeightType,
        WidthType = M::WidthType,
    >,
>(
    m: &M,
    n: &N,
) -> K
where
    M::Item: Ring,
{
    if M::HEIGHT == N::HEIGHT
        && M::HEIGHT == K::HEIGHT
        && M::WIDTH == N::WIDTH
        && M::WIDTH == K::WIDTH
    {
        madd_trim(m, n)
    } else {
        panic!("{}", crate::DIM_MISMATCH)
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
            if let (Some(x), Some(y)) =
                (m.coeff_ref((i, k)), n.coeff_ref((k, j)))
            {
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
    N: Matrix<Item = M::Item, HeightType = M::WidthType>,
    K: Matrix<
        Item = M::Item,
        HeightType = M::HeightType,
        WidthType = N::WidthType,
    >,
>(
    m: &M,
    n: &N,
) -> K
where
    M::Item: Ring,
{
    if M::HEIGHT == K::HEIGHT && M::WIDTH == N::HEIGHT && N::WIDTH == K::WIDTH {
        mmul_trim(m, n)
    } else {
        panic!("{}", crate::DIM_MISMATCH)
    }
}
