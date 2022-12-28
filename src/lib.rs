//! # χ-math
//!
//! A library for **exact** mathematical computation. We prioritize, in order:
//!
//! - Correctness
//! - Generality
//! - Speed
//!
//! ## Features
//!
//! - Basic matrix operations for both [statically](crate::data::MatrixMN) and
//!   [dynamically](crate::data::MatrixDyn) sized matrices.
//! - A bignum type for [naturals](crate::data::Nat),
//!   [integers](crate::data::Int), and [rationals](crate::data::Rat) (all works
//!   in progress).
//!
//! ## Design
//!
//! χ-math is built out of a large hierarchy of traits. For instance, an
//! [`AddGroup`](crate::traits::AddGroup) is an
//! [`AddMonoid`](crate::traits::AddMonoid), which itself implements
//! [`Add`](crate::traits::Add) and [`Zero`](crate::traits::Zero). Each
//! successive trait implements new methods, or asserts new facts about the
//! type, or both.
//!
//! **This crate uses unsafe code. Erroneous implementations of the traits in
//! the library can cause undefined behavior.**
//!
//! See the [`traits`] module for more information.
//!
//! ## Why the name?
//!
//! It looked cool.

extern crate self as xmath;

pub mod algs;
pub mod data;
pub mod traits;

pub use xmath_macro::*;

/// Transmutes a reference `&T` into a reference `&U`.
///
/// ## Safety
///
/// Both types must have the same layout.
unsafe fn transmute_ref<T: ?Sized, U>(x: &T) -> &U {
    &*(x as *const T).cast()
}

/// Transmutes a mutable reference `&mut T` into a mutable reference `&mut U`.
///
/// ## Safety
///
/// Both types must have the same layout.
unsafe fn transmute_mut<T: ?Sized, U>(x: &mut T) -> &mut U {
    &mut *(x as *mut T).cast()
}

/// A workaround for transmuting a generic type into another.
///
/// Borrowed from https://github.com/rust-lang/rust/issues/61956.
///
/// ## Safety
///
/// All the same safety considerations for [`std::mem::transmute`] still apply.
unsafe fn transmute_gen<T, U>(x: T) -> U {
    (*transmute_ref::<_, std::mem::MaybeUninit<U>>(
        &std::mem::MaybeUninit::new(x),
    ))
    .assume_init_read()
}

/// Transmutes `[T; 1]` into `T`.
fn from_array<T>(x: [T; 1]) -> T {
    // Safety: both types have the same layout.
    unsafe { crate::transmute_gen(x) }
}
