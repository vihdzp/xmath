use std::fmt::Display;

use super::basic::Zero;

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct CSingle<T>(pub T);

impl<T> FromIterator<T> for CSingle<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self(iter.into_iter().next().unwrap())
    }
}

impl<T> AsRef<[T]> for CSingle<T> {
    fn as_ref(&self) -> &[T] {
        std::slice::from_ref(&self.0)
    }
}

impl<T> AsMut<[T]> for CSingle<T> {
    fn as_mut(&mut self) -> &mut [T] {
        std::slice::from_mut(&mut self.0)
    }
}

/// A pair of elements with `repr(C)`.
///
/// This doesn't implement any algebraic traits, see
/// [`Pair`](crate::data::basic::Pair) for that usage.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CPair<T, U>(pub T, pub U);

impl<T: Copy> CPair<T, T> {
    pub const fn swap(x: Self) -> Self {
        CPair(x.1, x.0)
    }
}

pub trait CTuple:
    FromIterator<Self::Item> + AsRef<[Self::Item]> + AsMut<[Self::Item]>
{
    const LEN: usize;
    type Item;
}

impl<T> CTuple for CSingle<T> {
    const LEN: usize = 1;
    type Item = T;
}

impl<T: CTuple> FromIterator<T::Item> for CPair<T::Item, T> {
    fn from_iter<I: IntoIterator<Item = T::Item>>(iter: I) -> Self {
        let mut iter = iter.into_iter();
        Self(iter.next().unwrap(), iter.collect())
    }
}

impl<T: CTuple> AsRef<[T::Item]> for CPair<T::Item, T> {
    fn as_ref(&self) -> &[T::Item] {
        unsafe {
            std::slice::from_raw_parts(
                self as *const Self as *const _,
                Self::LEN,
            )
        }
    }
}

impl<T: CTuple> AsMut<[T::Item]> for CPair<T::Item, T> {
    fn as_mut(&mut self) -> &mut [T::Item] {
        unsafe {
            std::slice::from_raw_parts_mut(
                self as *mut Self as *mut _,
                Self::LEN,
            )
        }
    }
}

impl<T: CTuple> CTuple for CPair<T::Item, T> {
    const LEN: usize = T::LEN + 1;
    type Item = T::Item;
}

impl<T: Zero, U: CTuple<Item = T> + Eq> Zero for U {
    fn zero() -> Self {
        std::iter::repeat_with(T::zero).collect()
    }
}

/// A type level integer, starting from 1.
pub trait TypeNum {
    const VAL: usize;

    /// An array of the corresponding length.
    type Array<T>: CTuple<Item = T>;
}

/// The type-level integer `1`.
pub struct U1;

impl TypeNum for U1 {
    const VAL: usize = 1;
    type Array<T> = CSingle<T>;
}

/// The type-level integer `D + 1`.
pub struct Succ<D>(D);

impl<D: TypeNum> TypeNum for Succ<D> {
    const VAL: usize = D::VAL + 1;
    type Array<T> = CPair<T, D::Array<T>>;
}

pub type U2 = Succ<U1>;

pub mod ctuple {
    use super::*;

    pub fn pairwise<C: TypeNum, T, U, F: Fn(&T, &U) -> bool>(
        x: &C::Array<T>,
        y: &C::Array<U>,
        f: F,
    ) -> bool {
        for i in 0..C::VAL {
            if !f(&x.as_ref()[i], &y.as_ref()[i]) {
                return false;
            }
        }

        true
    }
}

pub fn from_cpair<T>(x: CPair<T, T>) -> [T; 2] {
    // Safety: both types have the same layout.
    unsafe { crate::transmute_gen(x) }
}
