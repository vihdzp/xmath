use super::{basic::{Zero, One}, matrix::Dim};
pub use xmath_core::{Succ, U1};

/// A single element with `repr(transparent)`.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct CSingle<T>(pub T);

/// Reads a single element from the iterator and returns the resulting
/// `CSingle`.
impl<T> FromIterator<T> for CSingle<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self(iter.into_iter().next().unwrap())
    }
}

/// Returns a reference to the single entry.
impl<T> AsRef<[T]> for CSingle<T> {
    fn as_ref(&self) -> &[T] {
        std::slice::from_ref(&self.0)
    }
}

/// Returns a mutable reference to the single entry.
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

impl<T: Copy> CPair<T, CSingle<T>> {
    /// Given the tuple `(x, y)`, returns `(y, x)`.
    pub const fn swap(self) -> Self {
        crate::tuple!(self.1 .0, self.0)
    }
}

/// A trait for a type with a `const` value of zero. This differs from [`Zero`]
/// since that traits allows the zero value to be dynamically allocated (albeit
/// still constant).
pub trait ConstZero {
    /// The constant zero value of the type.
    const ZERO: Self;
}

impl ConstZero for usize {
    const ZERO: Self = 0;
}

impl ConstZero for Dim {
    const ZERO: Self = Self::Fin(0);
}

impl<T: ConstZero> ConstZero for CSingle<T> {
    const ZERO: Self = Self(T::ZERO);
}

impl<T: ConstZero, U: ConstZero> ConstZero for CPair<T, U> {
    const ZERO: Self = Self(T::ZERO, U::ZERO);
}

/// Represents an array of elements associated to a type-level integer, with the
/// same layout and alignment as `[Self::Item; Self::Len]`.
///
/// This trait is only implemented for `CSingle<T>`, `CPair<T, CSingle<T>>`, 
/// etc. and shouldn't be implemented for types outside of this crate.
pub trait CTuple:
    ConstZero + FromIterator<Self::Item> + AsRef<[Self::Item]> + AsMut<[Self::Item]>
{
    /// The length of the tuple.
    const LEN: usize;

    /// The item in the tuple.
    type Item: ConstZero;
}

impl<T: ConstZero> CTuple for CSingle<T> {
    const LEN: usize = 1;
    type Item = T;
}

/// Reads elements from the iterator until the tuple is filled.
impl<T: CTuple> FromIterator<T::Item> for CPair<T::Item, T> {
    fn from_iter<I: IntoIterator<Item = T::Item>>(iter: I) -> Self {
        let mut iter = iter.into_iter();
        Self(iter.next().unwrap(), iter.collect())
    }
}

/// Returns the underlying slice of elements for the tuple.
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

/// Returns the underlying mutable slice of elements for the tuple.
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

/// The zero element is the tuple all of whose entries are 0.
impl<T: Zero, U: CTuple<Item = T> + Eq> Zero for U {
    fn zero() -> Self {
        std::iter::repeat_with(T::zero).collect()
    }
}

/// The one element is the tuple all of whose entries are 1.
impl<T: One, U: CTuple<Item = T> + Eq> One for U {
    fn one() -> Self {
        std::iter::repeat_with(T::one).collect()
    }
}

/// A type level integer, starting from 1.
///
/// Each type level integer `N` has an associated array type with guaranteed
/// same layout as `[T; N]`.
pub trait TypeNum {
    /// The constant associated with the type.
    const VAL: usize;

    /// An array of the corresponding length.
    type Array<T: ConstZero>: CTuple<Item = T>;
}

impl TypeNum for U1 {
    const VAL: usize = 1;
    type Array<T: ConstZero> = CSingle<T>;
}

impl<D: TypeNum> TypeNum for Succ<D> {
    const VAL: usize = D::VAL + 1;
    type Array<T: ConstZero> = CPair<T, D::Array<T>>;
}

/// Returns the type-level integer associated with a literal.
#[macro_export]
macro_rules! dim {
    (0) => {
        compile_error!("dimension 0 is invalid")
    };
    ($n: literal) => {
        xmath_macro::dim!($n)
    };
}

pub type U2 = dim!(2);
pub type U3 = dim!(3);

/// Returns a [`CTuple`] from a list of values.
#[macro_export]
macro_rules! tuple {
    ($t: expr) => {
        $crate::traits::dim::CSingle($t)
    };
    ($t: expr, $($ts: expr),*) => {
        $crate::traits::dim::CPair($t, $crate::tuple!($($ts),*))
    };
}

/// `ctuple!(T; N)` returns the [`CTuple`] type of item `T` and length `N`.
#[macro_export]
macro_rules! ctuple {
    ($t: ty; $n: literal) => {
        <$crate::dim!($n) as $crate::traits::dim::TypeNum>::Array<$t>
    };
}

pub mod ctuple {
    use super::*;

    /// Returns whether `f` holds for all pairs of entries with the same index.
    pub fn pairwise<
        C: TypeNum,
        T: ConstZero,
        U: ConstZero,
        F: Fn(&T, &U) -> bool,
    >(
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

#[cfg(test)]
mod test {
    use crate::{ctuple, tuple};

    #[test]
    fn tuple() {
        // Check that the types match.
        let _: ctuple!(usize; 3) = tuple!(1, 2, 3);
    }
}
