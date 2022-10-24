//! Implements the type of arrays.

use crate::traits::{
    basic::*,
    matrix::{Iter, LinearModule, List, ListIter, Module},
};

/// A statically-sized array of elements of a single type.
///
/// ## Internal representation
///
/// This stores a single `[T; N]` field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct Array<T, const N: usize>(pub [T; N]);

impl<T: Default, const N: usize> Default for Array<T, N> {
    fn default() -> Self {
        Self(std::array::from_fn(|_| T::default()))
    }
}

impl<T, const N: usize> Array<T, N> {
    /// Performs a pairwise operation on two arrays.
    pub fn pairwise<F: FnMut(&T, &T) -> T>(&self, x: &Self, mut f: F) -> Self {
        Self(std::array::from_fn(|i| f(&self[i], &x[i])))
    }

    /// Performs a mutable pairwise operation on two arrays.
    pub fn pairwise_mut<F: FnMut(&mut T, &T)>(&mut self, x: &Self, mut f: F) {
        for i in 0..N {
            f(&mut self[i], &x[i]);
        }
    }
}

impl<T, const N: usize> AsRef<[T; N]> for Array<T, N> {
    fn as_ref(&self) -> &[T; N] {
        &self.0
    }
}

impl<T, const N: usize> AsMut<[T; N]> for Array<T, N> {
    fn as_mut(&mut self) -> &mut [T; N] {
        &mut self.0
    }
}

impl<T, const N: usize> std::ops::Index<usize> for Array<T, N> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.as_ref()[index]
    }
}

impl<T, const N: usize> std::ops::IndexMut<usize> for Array<T, N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.as_mut()[index]
    }
}

impl<T, const N: usize> IntoIterator for Array<T, N> {
    type Item = T;
    type IntoIter = std::array::IntoIter<T, N>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a, T, const N: usize> IntoIterator for Iter<'a, &'a Array<T, N>, usize> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.as_ref().iter()
    }
}

impl<'a, C, T: ListIter<C>, const N: usize> IntoIterator
    for Iter<'a, &'a Array<T, N>, (usize, C)>
{
    type Item = &'a T::Item;
    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        todo!()
    }
}

impl<T: Zero, const N: usize> Zero for Array<T, N> {
    fn zero() -> Self {
        Self(std::array::from_fn(|_| T::zero()))
    }

    fn is_zero(&self) -> bool {
        self.as_ref().iter().all(T::is_zero)
    }
}

impl<T: One, const N: usize> One for Array<T, N> {
    fn one() -> Self {
        Self(std::array::from_fn(|_| T::one()))
    }

    fn is_one(&self) -> bool {
        self.as_ref().iter().all(T::is_one)
    }
}

impl<T: Neg, const N: usize> Neg for Array<T, N> {
    fn neg(&self) -> Self {
        self.map(T::neg)
    }

    fn neg_mut(&mut self) {
        self.map_mut(T::neg_mut);
    }
}

impl<T: Inv, const N: usize> Inv for Array<T, N> {
    fn inv(&self) -> Self {
        self.map(T::inv)
    }

    fn inv_mut(&mut self) {
        self.map_mut(T::inv_mut)
    }
}

impl<T: Add, const N: usize> Add for Array<T, N> {
    fn add(&self, x: &Self) -> Self {
        self.pairwise(x, T::add)
    }

    fn add_mut(&mut self, x: &Self) {
        self.pairwise_mut(x, T::add_mut);
    }

    fn double(&self) -> Self {
        self.map(T::double)
    }

    fn double_mut(&mut self) {
        self.map_mut(T::double_mut);
    }
}

impl<T: Mul, const N: usize> Mul for Array<T, N> {
    fn mul(&self, x: &Self) -> Self {
        self.pairwise(x, T::mul)
    }

    fn mul_mut(&mut self, x: &Self) {
        self.pairwise_mut(x, T::mul_mut);
    }

    fn sq(&self) -> Self {
        self.map(T::sq)
    }

    fn sq_mut(&mut self) {
        self.map_mut(T::sq_mut);
    }
}

impl<T: CommAdd, const N: usize> CommAdd for Array<T, N> {}
impl<T: CommMul, const N: usize> CommMul for Array<T, N> {}

impl<T: Sub, const N: usize> Sub for Array<T, N> {
    fn sub(&self, x: &Self) -> Self {
        self.pairwise(x, T::sub)
    }

    fn sub_mut(&mut self, x: &Self) {
        self.pairwise_mut(x, T::sub_mut);
    }
}

impl<T: Div, const N: usize> Div for Array<T, N> {
    fn div(&self, x: &Self) -> Self {
        self.pairwise(x, T::div)
    }

    fn div_mut(&mut self, x: &Self) {
        self.pairwise_mut(x, T::div_mut);
    }
}

impl<T: AddMonoid, const N: usize> AddMonoid for Array<T, N> {}
impl<T: MulMonoid, const N: usize> MulMonoid for Array<T, N> {}
impl<T: AddGroup, const N: usize> AddGroup for Array<T, N> {}
impl<T: MulGroup, const N: usize> MulGroup for Array<T, N> {}
impl<T: Ring, const N: usize> Ring for Array<T, N> {}

impl<T, const N: usize> List<usize> for Array<T, N> {
    type Item = T;

    fn is_valid_coeff(i: usize) -> bool {
        i < N
    }

    fn coeff_ref(&self, i: usize) -> Option<&Self::Item> {
        self.as_ref().get(i)
    }

    fn coeff_set(&mut self, i: usize, x: Self::Item) {
        self.as_mut()[i] = x;
    }
}

impl<T, const N: usize> ListIter<usize> for Array<T, N> {
    fn map<F: FnMut(&T) -> T>(&self, mut f: F) -> Self {
        Self(std::array::from_fn(|i| f(&self[i])))
    }

    fn map_mut<F: FnMut(&mut T)>(&mut self, mut f: F) {
        for i in 0..N {
            f(&mut self[i]);
        }
    }

    /* fn pairwise<F: FnMut(&Self::Item, &Self::Item) -> Self::Item>(
        &self,
        x: &Self,
        mut f: F,
    ) -> Self {
        Self(std::array::from_fn(|i| f(&self[i], &x[i])))
    }

    fn pairwise_mut<F: FnMut(&mut Self::Item, &Self::Item)>(
        &mut self,
        x: &Self,
        mut f: F,
    ) {
        for i in 0..N {
            f(&mut self[i], &x[i]);
        }
    }*/
}

impl<T: Ring, const N: usize> Module<usize> for Array<T, N> {
    fn smul(&self, x: &T) -> Self {
        self.map(|y| y.mul(x))
    }

    fn smul_mut(&mut self, x: &T) {
        self.map_mut(|y| y.mul_mut(x));
    }
}

impl<T: Ring, const N: usize> FromIterator<T> for Array<T, N> {
    fn from_iter<J: IntoIterator<Item = T>>(iter: J) -> Self {
        let mut iter = iter.into_iter();
        Self(std::array::from_fn(|_| iter.next().unwrap_or_else(T::zero)))
    }
}

impl<T, const N: usize> LinearModule for Array<T, N>
where
    T: Ring,
{
    type DimType = usize;
    const DIM: Self::DimType = N;

    fn support(&self) -> usize {
        N
    }
}
