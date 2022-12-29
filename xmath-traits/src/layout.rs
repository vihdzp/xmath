//! Traits relating to the layout of types.

use xmath_core::{transmute_gen, transmute_mut, transmute_ref};

/// A trait for a type with an underlying slice.
///
/// There is no requirement that this slice is the only field for the type. In
/// particular, it might not always be possible to turn `&[Self::Item]` into
/// `&Self`.
///
/// There's no hard distinction between implementing this trait and implementing
/// [`AsRef<[T]>`](std::convert::AsRef) and [`AsMut<[T]>`](std::convert::AsMut).
/// However, there's cases where it may be appropriate to implement one but not
/// the other. For instance, [`Transpose`](crate::data::aliases::Transpose)
/// implements `SliceLike`, as it's a requirement to implement [`Transparent`],
/// but it does not implement [`AsRef<[T]>`](std::convert::AsRef) as it's not
/// intuitive what this would do.
pub trait SliceLike {
    /// The item of the array.
    type Item;

    /// Returns a reference to the underlying slice.
    fn as_slice(&self) -> &[Self::Item];

    /// Returns a mutable reference to the underlying slice.
    fn as_mut_slice(&mut self) -> &mut [Self::Item];

    /// A default [`Index`](std::ops::Index) implementation.
    fn index(&self, index: usize) -> &Self::Item {
        &self.as_slice()[index]
    }

    /// A default [`IndexMut`](std::ops::IndexMut) implementation.
    fn index_mut(&mut self, index: usize) -> &mut Self::Item {
        &mut self.as_mut_slice()[index]
    }
}

impl<T, const N: usize> SliceLike for [T; N] {
    type Item = T;

    fn as_slice(&self) -> &[Self::Item] {
        self.as_slice()
    }

    fn as_mut_slice(&mut self) -> &mut [Self::Item] {
        self.as_mut_slice()
    }
}

impl<T> SliceLike for Vec<T> {
    type Item = T;

    fn as_slice(&self) -> &[Self::Item] {
        self.as_slice()
    }

    fn as_mut_slice(&mut self) -> &mut [Self::Item] {
        self.as_mut_slice()
    }
}

/// A trait for a type with the same layout and alignment as
/// `[Self::Item; Self::Len]`.
///
/// The [`FromIterator`] implementation must read from the iterator until no
/// more entries can be filled.
pub trait ArrayLike: SliceLike + FromIterator<Self::Item> {
    //// The length of the array.
    const LEN: usize;

    /// Reads exactly `LEN` entries from the iterator, uses them to build an
    /// instance of the type.
    fn from_iter_mut<I: Iterator<Item = Self::Item>>(iter: &mut I) -> Self;

    /// A default [`FromIterator`] implementation.
    fn from_iter<I: IntoIterator<Item = Self::Item>>(iter: I) -> Self {
        Self::from_iter_mut(&mut iter.into_iter())
    }

    /// Performs a compile-time check and transmutes `self` into an array with a
    /// fixed size.
    fn to_array<const N: usize>(self) -> [Self::Item; N] {
        assert_eq!(Self::LEN, N);
        unsafe { transmute_gen(self) }
    }

    /// Performs a compile-time check and transmutes an array with a fixed size
    /// into this type.
    fn from_array<const N: usize>(array: [Self::Item; N]) -> Self {
        assert_eq!(Self::LEN, N);
        unsafe { transmute_gen(array) }
    }

    /// Performs a compile-time check and transmutes a reference to `self` into
    /// a reference to an array with a fixed size.
    fn as_const_slice<const N: usize>(&self) -> &[Self::Item; N] {
        assert_eq!(Self::LEN, N);
        unsafe { transmute_ref(self) }
    }

    /// Performs a compile-time check and transmutes a reference to an array
    /// with a fixed size into a reference to `self`.
    fn from_const_slice<const N: usize>(value: &[Self::Item; N]) -> &Self {
        assert_eq!(Self::LEN, N);
        unsafe { transmute_ref(value) }
    }

    /// Performs a compile-time check and transmutes a mutable reference to
    /// `self` into a mutable reference to an array with a fixed size.
    fn as_const_mut_slice<const N: usize>(&mut self) -> &mut [Self::Item; N] {
        assert_eq!(Self::LEN, N);
        unsafe { transmute_mut(self) }
    }

    /// Performs a compile-time check and transmutes a mutable reference to an
    /// array with a fixed size into a mutable reference to `self`.
    fn from_const_mut_slice<const N: usize>(value: &mut [Self::Item; N]) -> &mut Self {
        assert_eq!(Self::LEN, N);
        unsafe { transmute_mut(value) }
    }

    /// A default [`AsRef`] implementation.
    fn as_ref(&self) -> &[Self::Item] {
        unsafe { std::slice::from_raw_parts((self as *const Self).cast(), Self::LEN) }
    }

    /// A default [`AsMut`] implementation.
    fn as_mut(&mut self) -> &mut [Self::Item] {
        unsafe { std::slice::from_raw_parts_mut((self as *mut Self).cast(), Self::LEN) }
    }
}

/// A trait for a type with the same layout and alignment as `Self::Item`, or
/// equivalently, as `[Self::Item, 1]`.
pub trait Transparent: ArrayLike {
    /// Transmutes an element of the type into the inner type.
    fn to_single(self) -> Self::Item {
        xmath_core::from_array(Self::to_array(self))
    }

    /// Transmutes the inner value into the type.
    fn from_single(value: Self::Item) -> Self {
        Self::from_array([value])
    }

    /// Returns a reference to the inner value.
    fn as_single_ref(&self) -> &Self::Item {
        &self.as_slice()[0]
    }

    /// Transmutes a reference to an inner value into a reference to `self`.
    fn from_single_ref(value: &Self::Item) -> &Self {
        unsafe { transmute_ref(value) }
    }

    /// Returns a mutable reference to the inner value.
    fn as_single_mut(&mut self) -> &mut Self::Item {
        &mut self.as_mut_slice()[0]
    }

    /// Transmutes a mutable reference to an inner value into a mutable
    /// reference to `self`.
    fn from_single_mut(value: &mut Self::Item) -> &mut Self {
        unsafe { transmute_mut(value) }
    }

    /// A default [`ArrayLike::from_iter_mut`] implementation.
    fn from_iter_mut_def<I: Iterator<Item = Self::Item>>(iter: &mut I) -> Self {
        Self::from_single(iter.into_iter().next().unwrap())
    }
}
