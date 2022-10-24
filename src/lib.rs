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
//! (list some cool stuff here)
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

pub mod algs;
pub mod data;
pub mod traits;

/// Generic error message.
const DIM_MISMATCH: &str = "matrix dimension mismatch";

/// A workaround for transmuting a generic type into another.
///
/// Borrowed from https://github.com/rust-lang/rust/issues/61956.
///
/// ## Safety
///
/// All the same safety considerations for [`std::mem::transmute`] still apply.
unsafe fn transmute_gen<T, U>(x: T) -> U {
    (&*(&std::mem::MaybeUninit::new(x) as *const _
        as *const std::mem::MaybeUninit<_>))
        .assume_init_read()
}

/// Transmutes `[T; 1]` into `T`.
fn from_array<T>(x: [T; 1]) -> T {
    // Safety: both types have the same layout.
    unsafe { crate::transmute_gen(x) }
}
