//! Contains the definitions for basic matrix types, including [`Array`] and
//! [`Poly`].

mod array;
mod poly;

pub use array::*;
pub use poly::*;

use crate::traits::{basic::*, dim::*, matrix::*};

/// The empty list of a given type. This is a unit type.
///
/// ## Internal representation
///
/// This contains a single `PhantomData<T>` field.
#[derive(Debug)]
pub struct Empty<T>(std::marker::PhantomData<T>);

impl<T> Default for Empty<T> {
    fn default() -> Self {
        Self(Default::default())
    }
}

impl<T> PartialEq for Empty<T> {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl<T> Eq for Empty<T> {}

impl<T> Empty<T> {
    /// Returns the unique empty list for the type.
    pub fn new() -> Self {
        Self::default()
    }
}

impl<T> Clone for Empty<T> {
    fn clone(&self) -> Self {
        Self::new()
    }
}

impl<T> Copy for Empty<T> {}

/// Returns the unit.
impl<T> Zero for Empty<T> {
    fn zero() -> Self {
        Self::new()
    }
}

/// Returns the unit.
impl<T> One for Empty<T> {
    fn one() -> Self {
        Self::new()
    }
}

/// The identity function.
impl<T> Neg for Empty<T> {
    fn neg(&self) -> Self {
        Self::new()
    }
}

/// The identity function.
impl<T> Inv for Empty<T> {
    fn inv(&self) -> Self {
        Self::new()
    }
}

/// The identity function.
impl<T> Add for Empty<T> {
    fn add(&self, _: &Self) -> Self {
        Self::new()
    }
}

/// The identity function.
impl<T> Mul for Empty<T> {
    fn mul(&self, _: &Self) -> Self {
        Self::new()
    }
}

impl<T> IntegralDomain for Empty<T> {}
impl<T> CommAdd for Empty<T> {}
impl<T> CommMul for Empty<T> {}
impl<T> Sub for Empty<T> {}
impl<T> Div for Empty<T> {}
impl<T> AddMonoid for Empty<T> {}
impl<T> MulMonoid for Empty<T> {}
impl<T> AddGroup for Empty<T> {}
impl<T> MulGroup for Empty<T> {}
impl<T> Ring for Empty<T> {}

impl<T, C: TypeNum> List<C> for Empty<T> {
    type Item = T;
    const SIZE: C::Array<Dim> = C::Array::<Dim>::ZERO;

    fn coeff_ref_gen(
        &self,
        _: &<C as TypeNum>::Array<usize>,
    ) -> Option<&Self::Item> {
        None
    }

    unsafe fn coeff_set_unchecked_gen(
        &mut self,
        _: &<C as TypeNum>::Array<usize>,
        _: Self::Item,
    ) {
        unreachable!()
    }

    fn map<F: Fn(&Self::Item) -> Self::Item>(&self, _: F) -> Self {
        Self::new()
    }

    fn map_mut<F: Fn(&mut Self::Item)>(&mut self, _: F) {}
}

impl<T: Ring, C: TypeNum> Module<C> for Empty<T> {
    fn dot(&self, _: &Self) -> Self::Item {
        Self::Item::zero()
    }
}

impl<T> FromIterator<T> for Empty<T> {
    fn from_iter<I: IntoIterator<Item = T>>(_: I) -> Self {
        Self::new()
    }
}

impl<T: Ring> LinearModule for Empty<T> {
    fn support(&self) -> usize {
        0
    }
}

impl<T: Ring> Matrix for Empty<T> {
    const DIR: Direction = Direction::Either;

    fn col_support(&self, _: usize) -> usize {
        0
    }

    fn row_support(&self, _: usize) -> usize {
        0
    }

    fn height(&self) -> usize {
        0
    }

    fn width(&self) -> usize {
        0
    }

    fn collect_row<I: Iterator<Item = Self::Item>, J: Iterator<Item = I>>(
        _: J,
    ) -> Self {
        Self::new()
    }

    fn collect_col<I: Iterator<Item = Self::Item>, J: Iterator<Item = I>>(
        _: J,
    ) -> Self {
        Self::new()
    }
}
