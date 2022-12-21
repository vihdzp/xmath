//! Implements matrices.
//!
//! ## Important note
//!
//! Due to compiler bug
//! [#37748](https://github.com/rust-lang/rust/issues/37748), you'll need to
//! provide type arguments to most functions explicitly.

use super::array::Array;
use super::poly::Poly;
use crate::algs::matrix::*;
use crate::traits::dim::{CPair, TypeNum};
use crate::traits::matrix::*;
use crate::{data::aliases::Transpose, traits::basic::*};
use std::cmp::Ordering::*;
use xmath_core::Succ;

/// An alias for an M × N statically sized matrix.
pub type MatrixMN<T, const M: usize, const N: usize> = Array<Array<T, N>, M>;

impl<T: Ring, const M: usize, const N: usize> MatrixMN<T, M, N> {
    /// Adds two statically sized matrices.
    pub fn madd(&self, m: &Self) -> Self {
        madd_gen(self, m)
    }

    /// Multiplies two statically sized matrices.
    pub fn mmul<const K: usize>(
        &self,
        m: &MatrixMN<T, N, K>,
    ) -> MatrixMN<T, M, K> {
        mmul_gen(self, m)
    }

    /// Transmutes a matrix as another matrix of the same size. The size check
    /// is performed at compile time.
    ///
    /// ## Panics
    ///
    /// Panics if both matrices don't have the same size.
    pub fn transmute<const K: usize, const L: usize>(
        self,
    ) -> MatrixMN<T, K, L> {
        if M * N == K * L {
            // Safety: matrices of the same size have the same layout.
            unsafe { crate::transmute_gen(self) }
        } else {
            panic!("{}", crate::SIZE_MISMATCH)
        }
    }

    /// Transmutes a reference to a matrix as a refrence to another matrix of
    /// the same size. The size check is performed at compile time.
    ///
    /// ## Panics
    ///
    /// Panics if both matrices don't have the same size.
    pub fn transmute_ref<const K: usize, const L: usize>(
        &self,
    ) -> &MatrixMN<T, K, L> {
        if M * N == K * L {
            // Safety: we've performed the size check.
            unsafe { &*(self as *const Self).cast() }
        } else {
            panic!("{}", crate::SIZE_MISMATCH)
        }
    }

    /// Transmutes a mutable reference to a matrix as a mutable refrence to
    /// another matrix of the same size. The size check is performed at
    /// compile time.
    ///
    /// ## Panics
    ///
    /// Panics if both matrices don't have the same size.
    pub fn transmute_mut<const K: usize, const L: usize>(
        &mut self,
    ) -> &mut MatrixMN<T, K, L> {
        if M * N == K * L {
            // Safety: we've performed the size check.
            unsafe { &mut *(self as *mut Self).cast() }
        } else {
            panic!("{}", crate::SIZE_MISMATCH)
        }
    }
}

/// An alias for a dynamically sized matrix.
pub type MatrixDyn<T> = Poly<Poly<T>>;

impl<T: Ring> MatrixDyn<T> {
    /// Adds two dynamically sized matrices.
    pub fn madd(&self, m: &Self) -> Self {
        madd_gen(self, m)
    }

    /// Multiplies two dynamically sized matrices.
    pub fn mmul<const K: usize>(&self, m: &Self) -> Self {
        mmul_gen(self, m)
    }
}

/// Interprets a vector as a row vector.
pub type RowVec<T> = Array<T, 1>;

impl<T> RowVec<T> {
    /// Creates a new row vector.
    pub fn new_row(x: T) -> Self {
        Self([x])
    }

    /// Gets the inner vector.
    pub fn inner(self) -> T {
        crate::from_array(self.0)
    }
}

/// Interprets a vector as a column vector.
pub type ColVec<T> = Transpose<RowVec<T>>;

impl<T> ColVec<T> {
    /// Creates a new column vector.
    pub fn new(x: T) -> Self {
        Self(RowVec::new_row(x))
    }

    /// Gets the inner vector.
    pub fn inner(self) -> T {
        self.0.inner()
    }
}

#[cfg(test)]
mod tests {
    use std::num::Wrapping;

    use super::*;

    #[test]
    fn mul() {
        let m1: MatrixMN<Wu8, 2, 3> = Array([
            Array([Wrapping(1), Wrapping(2), Wrapping(3)]),
            Array([Wrapping(4), Wrapping(5), Wrapping(6)]),
        ]);

        let m2: MatrixMN<Wu8, 3, 2> = Array([
            Array([Wrapping(1), Wrapping(4)]),
            Array([Wrapping(2), Wrapping(5)]),
            Array([Wrapping(3), Wrapping(6)]),
        ]);

        let _m3: MatrixMN<Wu8, 2, 3> = Array([
            Array([Wrapping(1), Wrapping(2), Wrapping(3)]),
            Array([Wrapping(4), Wrapping(5), Wrapping(6)]),
        ]);

        println!("{:?}", m1.mmul(&m2));
    }
}
