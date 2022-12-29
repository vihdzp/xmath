use xmath_macro::{Transparent, ArrayFromIter};
use xmath_traits::*;

use crate::traits::*;


/// A wrapper for a matrix that is to be interpreted with the rows and columns
/// swapped.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Transparent, ArrayFromIter)]
#[repr(transparent)]
pub struct Transpose<T>(pub T);

impl<M: List<U2>> List<U2> for Transpose<M> {
    type Item = M::Item;
    const SIZE: Array2<Dim> = M::SIZE.swap();

    fn coeff_ref_gen(&self, index: &Array2<usize>) -> Option<&Self::Item> {
        self.0.coeff_ref_gen(&index.swap())
    }

    unsafe fn coeff_set_unchecked_gen(&mut self, index: &Array2<usize>, value: Self::Item) {
        self.0.coeff_set_unchecked_gen(&index.swap(), value)
    }

    fn map<F: Fn(&Self::Item) -> Self::Item>(&self, f: F) -> Self {
        Self(self.0.map(f))
    }

    fn map_mut<F: Fn(&mut Self::Item)>(&mut self, f: F) {
        self.0.map_mut(f);
    }
}

/// The zero value is just the transpose of 0.
impl<T: Zero> Zero for Transpose<T> {
    fn zero() -> Self {
        Self(T::zero())
    }

    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

/// The one value is just the transpose of 1.
impl<T: One> One for Transpose<T> {
    fn one() -> Self {
        Self(T::one())
    }

    fn is_one(&self) -> bool {
        self.0.is_one()
    }
}

/// Negates the inner value.
impl<T: Neg> Neg for Transpose<T> {
    fn neg(&self) -> Self {
        Self(self.0.neg())
    }
}

/// Adds the inner values.
impl<T: Add> Add for Transpose<T> {
    fn add(&self, x: &Self) -> Self {
        Self(self.0.add(&x.0))
    }

    fn add_mut(&mut self, x: &Self) {
        self.0.add_mut(&x.0);
    }

    fn double(&self) -> Self {
        Self(self.0.double())
    }

    fn double_mut(&mut self) {
        self.0.double_mut();
    }
}

/// Subtracts the inner values.
impl<T: Sub> Sub for Transpose<T> {
    fn sub(&self, x: &Self) -> Self {
        Self(self.0.sub(&x.0))
    }

    fn sub_mut(&mut self, x: &Self) {
        self.0.sub_mut(&x.0);
    }
}

impl<T: AddMonoid> AddMonoid for Transpose<T> {}
impl<T: AddGroup> AddGroup for Transpose<T> {}

impl<M: Module<U2>> Module<U2> for Transpose<M>
where
    Self::Item: Ring,
{
    fn smul(&self, x: &M::Item) -> Self {
        Self(self.0.smul(x))
    }

    fn smul_mut(&mut self, x: &M::Item) {
        self.0.smul_mut(x);
    }

    fn dot(&self, x: &Self) -> Self::Item {
        self.0.dot(&x.0)
    }
}

impl<M: Matrix> Matrix for Transpose<M>
where
    M::Item: Ring,
{
    const DIR: Direction = M::DIR.transpose();

    fn col_support(&self, j: usize) -> usize {
        self.0.row_support(j)
    }

    fn row_support(&self, i: usize) -> usize {
        self.0.col_support(i)
    }

    fn height(&self) -> usize {
        self.0.width()
    }

    fn width(&self) -> usize {
        self.0.height()
    }

    fn collect_row<I: Iterator<Item = M::Item>, J: Iterator<Item = I>>(iter: J) -> Self {
        Self(M::collect_col(iter))
    }

    fn collect_col<I: Iterator<Item = M::Item>, J: Iterator<Item = I>>(iter: J) -> Self {
        Self(M::collect_row(iter))
    }
}