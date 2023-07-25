//! Defines a trait for wrappers like
//! ```rs
//! #[repr(transparent)]
//! struct NonZero<T: Zero>(T);
//! ```
//! These must enforce, as a runtime invariant, that they never hold a zero
//! value, as defined by the [`Zero`] implementation for `T`.
//!
//! **The wrapper must be `repr(transparent)`, or the default implementations
//! will invoke undefined behavior.**
//!
//! Ideally we would have a single `NonZero` wrapper, thus circumventing the
//! need for this trait. However, it wouldn't be possible to implement any
//! algebraic traits on `NonZero<T>` even for local types `T`, as this will
//! trigger error [E0117](https://doc.rust-lang.org/error_codes/E0117.html).
//! To automatically implement this wrapper, see the [`impl_nonzero!`] macro.

use crate::*;

/// A trait for a wrapper that enforces that it holds a nonzero value. See also
/// the module documentation for [`nonzero`].
pub trait NonZeroWrapper: Sized {
    //// The type being wrapped.
    type Inner: Zero;

    /// Returns a reference to the inner value.
    fn as_ref(&self) -> &Self::Inner {
        unsafe { xmath_core::transmute_ref(self) }
    }

    /// Returns a mutable reference to the inner value.
    ///
    /// ## Safety
    ///
    /// This value must not be modified to zero.
    unsafe fn as_mut(&mut self) -> &mut Self::Inner {
        xmath_core::transmute_mut(self)
    }

    /// Converts a reference to the inner value to a reference to `Self`.
    fn to_ref(value: &Self::Inner) -> &Self {
        // Safety: implementing this type means these have the same layout.
        unsafe { xmath_core::transmute_ref(value) }
    }

    /// Converts a mutable reference to the inner value to a mutable reference
    /// to `Self`.
    ///
    /// ## Safety
    ///
    /// This value must not be modified to zero.
    unsafe fn to_mut(value: &mut Self::Inner) -> &mut Self {
        xmath_core::transmute_mut(value)
    }

    /// Initializes an instance of the wrapper.
    ///
    /// ## Safety
    ///
    /// The passed value must be nonzero.
    unsafe fn new_unchecked(x: Self::Inner) -> Self {
        xmath_core::transmute_gen(x)
    }

    /// Initializes an instance of the wrapper. Returns `None` if a zero value
    /// is passed.
    fn new(x: Self::Inner) -> Option<Self> {
        if x.is_zero() {
            None
        } else {
            unsafe { Some(Self::new_unchecked(x)) }
        }
    }
}

/// Automatically implements a `NonZero` wrapper. See also the module docs for
/// [`nonzero`].
#[macro_export]
macro_rules! impl_nonzero {
    ($name: ident) => {
        /// A type alias which guarantees that the enclosed value is nonzero.
        /// This has been auto-implemented by
        /// [`impl_nonzero`](xmath_traits::impl_nonzero).
        #[repr(transparent)]
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name<T: xmath_traits::Zero>(T);

        impl<T: xmath_traits::Zero> $name<T> {
            /// Initializes an instance of the wrapper.
            ///
            /// ## Safety
            ///
            /// The passed value must be nonzero.
            pub const unsafe fn new_unchecked(x: T) -> Self {
                Self(x)
            }
        }

        impl<T: xmath_traits::Zero> xmath_traits::NonZeroWrapper for $name<T> {
            type Inner = T;
        }

        impl<T: xmath_traits::ZeroNeOne> xmath_traits::One for $name<T> {
            fn one() -> Self {
                // Safety: we assume `0 != 1`.
                unsafe { Self::new_unchecked(T::one()) }
            }
        }

        impl<T: xmath_traits::IntegralDomain> xmath_traits::Mul for $name<T> {
            fn mul(&self, x: &Self) -> Self {
                use xmath_traits::NonZeroWrapper;
                // Safety: we assume the product of nonzero elements is nonzero.
                unsafe { Self::new_unchecked(self.as_ref().mul(x.as_ref())) }
            }

            fn mul_mut(&mut self, x: &Self) {
                use xmath_traits::NonZeroWrapper;
                // Safety: we assume the product of nonzero elements is nonzero.
                unsafe { self.as_mut().mul_mut(x.as_ref()) };
            }

            fn sq(&self) -> Self {
                use xmath_traits::NonZeroWrapper;
                // Safety: we assume the product of nonzero elements is nonzero.
                unsafe { Self::new_unchecked(self.as_ref().sq()) }
            }

            fn sq_mut(&mut self) {
                use xmath_traits::NonZeroWrapper;
                // Safety: we assume the product of nonzero elements is nonzero.
                unsafe { self.as_mut().sq_mut() };
            }
        }

        impl<
                T: xmath_traits::IntegralDomain + xmath_traits::MulMonoid + xmath_traits::ZeroNeOne,
            > xmath_traits::MulMonoid for $name<T>
        {
        }
    };
    () => {
        xmath_traits::impl_nonzero!(NonZero);
    };
}
