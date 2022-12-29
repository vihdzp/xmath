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
//! See the [`xmath_traits`] crate for more information.
//!
//! ## Crates
//!
//! For internal organization, the χ-math codebase is split up into various
//! crates. For the moment being, these are subject to sudden and radical
//! change. The current hierarchy from top to down is as follows:
//!
//! - [`xmath`]: the main crate.
//! - [`num_bigint`]: a fork of [`num_bigint`](https://docs.rs/num-bigint/latest/num_bigint/index.html#)
//!    designed to interface better with the χ-math code.
//! - [`xmath_matrix`]: code for arrays and matrices.
//! - [`xmath_traits`]: the basic χ-math traits.
//! - [`xmath_macro`]: for any procedural macros.
//! - [`xmath_core`]: miscellaneous code that's used by the other crates.
//!
//! ## Why the name?
//!
//! The letter χ shares its first phoneme with the x from exact. Besides, it
//! really just looked cool.

extern crate self as xmath;

pub mod algs;
pub mod data;

pub use xmath_macro::*;
use xmath_traits::{IntegralDomain, Mul, MulMonoid, NonZeroWrapper, One, Zero, ZeroNeOne};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct NonZero<T>(T);

impl<T> NonZero<T> {
    pub const unsafe fn new_unchecked(x: T) -> Self {
        Self(x)
    }
}

impl<T: Zero> NonZeroWrapper for NonZero<T> {
    type Inner = T;
}

impl<T: IntegralDomain> Mul for NonZero<T> {
    fn mul(&self, rhs: &Self) -> Self {
        todo!()
    }
}

impl<T: ZeroNeOne> One for NonZero<T> {
    fn one() -> Self {
        todo!()
    }
}

impl<T: ZeroNeOne + IntegralDomain> MulMonoid for NonZero<T> {}
