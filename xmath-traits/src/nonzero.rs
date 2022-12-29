//! Defines a trait for wrappers like
//! ```rs
//! #[repr(transparent)]
//! struct NonZero<T: Zero>(T);
//! ```
//! These must enforce, as a runtime invariant, that they never hold a zero 
//! value, as defined by the [`Zero`] implementation for `T`.
//! 
//! Ideally we would have a single `NonZero` wrapper, thus circumventing the 
//! need for this trait. However, it wouldn't be possible to implement any 
//! algebraic traits on `NonZero<T>` even for local types `T`, as this will 
//! trigger error [E0117](https://doc.rust-lang.org/error_codes/E0117.html).

use crate::*;

pub trait NonZeroWrapper: Sized {
    type Inner: Zero;

    fn as_ref(&self) -> &Self::Inner {
        unsafe { xmath_core::transmute_ref(self) }
    }

    unsafe fn as_mut(&mut self) -> &mut Self::Inner {
        xmath_core::transmute_mut(self)
    }

    fn to_ref(value: &Self::Inner) -> &Self {
        unsafe { xmath_core::transmute_ref(value) }
    }

    unsafe fn to_mut(value: &mut Self::Inner) -> &mut Self {
        xmath_core::transmute_mut(value)
    }

    unsafe fn new_unchecked(x: Self::Inner) -> Self {
        xmath_core::transmute_gen(x)
    }

    fn new(x: Self::Inner) -> Option<Self> {
        if x.is_zero() {
            None
        } else {
            unsafe { Some(Self::new_unchecked(x)) }
        }
    }
}

/*
/// A type alias which guarantees that the enclosed value is nonzero.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct NonZero<T>(T);

/// Returns a reference to the underlying value.
impl<T> AsRef<T> for NonZero<T> {
    fn as_ref(&self) -> &T {
        &self.0
    }
}

impl<T: Zero> NonZero<T> {
    /// Returns a mutable reference to the inner value.
    ///
    /// ## Safety
    ///
    /// The caller must ensure this value remains nonzero.
    pub unsafe fn as_mut(&mut self) -> &mut T {
        &mut self.0
    }

    /// Gets a `NonZero<T>` reference from a nonzero `T` reference.
    ///
    /// ## Safety
    ///
    /// The caller must ensure `x` is indeed nonzero.
    pub unsafe fn to_ref(x: &T) -> &Self {
        xmath_core::transmute_ref(x)
    }

    /// Gets a mutable `NonZero<T>` reference from a nonzero `T` reference.
    ///
    /// ## Safety
    ///
    /// The caller must ensure `x` is indeed nonzero.
    pub unsafe fn to_mut(x: &mut T) -> &mut Self {
        xmath_core::transmute_mut(x)
    }

    /// Initializes a new `NonZero` from a given nonzero value.
    ///
    /// ## Safety
    ///
    /// The caller must ensure `x` is indeed nonzero.
    pub const unsafe fn new_unchecked(x: T) -> Self {
        Self(x)
    }

    /// Initializes a new `NonZero` from a given value. Returns `None` if the
    /// input is zero.
    pub fn new(x: T) -> Option<NonZero<T>> {
        if x.is_zero() {
            None
        } else {
            // Safety: we just checked that `x` is nonzero.
            unsafe { Some(Self::new_unchecked(x)) }
        }
    }
}

impl<T: ZeroNeOne> One for NonZero<T> {
    fn one() -> Self {
        // Safety: we assume `0 != 1`.
        unsafe { Self::new_unchecked(T::one()) }
    }
}

impl<T: IntegralDomain> Mul for NonZero<T> {
    fn mul(&self, x: &Self) -> Self {
        // Safety: we assume the product of nonzero elements is nonzero.
        unsafe { Self::new_unchecked(self.as_ref().mul(x.as_ref())) }
    }

    fn mul_mut(&mut self, x: &Self) {
        // Safety: we assume the product of nonzero elements is nonzero.
        unsafe { self.as_mut().mul_mut(x.as_ref()) };
    }

    fn sq(&self) -> Self {
        // Safety: we assume the product of nonzero elements is nonzero.
        unsafe { Self::new_unchecked(self.as_ref().sq()) }
    }

    fn sq_mut(&mut self) {
        // Safety: we assume the product of nonzero elements is nonzero.
        unsafe { self.as_mut().sq_mut() };
    }
}

impl<T: IntegralDomain + MulMonoid + ZeroNeOne> MulMonoid for NonZero<T> {}
 */
