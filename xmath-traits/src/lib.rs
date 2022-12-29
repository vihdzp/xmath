//! Defines all the most basic traits, representing different mathematical
//! structures, and implements them for the primitive types.
//!
//! ## Safety
//!
//! These traits have many conditions that cannot be enforced statically.
//! Despite this, we choose not to implement them as `unsafe` traits for
//! convenience. It is up to the user to verify all necessary conditions.
//! **Undefined behavior can occur** if these guarantees aren't upheld.
//!
//! Universal conditions that apply for all of the traits in this module are as
//! follows.
//!
//! - All functions are assumed to be pure, in the sense that the same arguments
//!   will always give the same outputs.
//! - Mutable and immutable versions of the same function must always give the
//!   same outputs.
//! - Functions always return the same values as default implementations, when
//!   these exist.
//! - Any valid inputs must return a valid output without panicking or invoking
//!   undefined behavior.
//!
//! This last condition restricts the possible trait implementations quite
//! heavily. For instance, `u8` can't implement [`Add`](basic::Add) since
//! addition can overflow. We make a sole exception for running out of memory,
//! with the understanding that it should only happen in extreme cases.

extern crate self as xmath_traits;

mod basic;
mod dim;
mod layout;
mod nonzero;

pub use basic::*;
pub use dim::*;
pub use layout::*;
pub use nonzero::*;

use std::num::Wrapping;

/// The ring of integers modulo 2<sup>8</sup>.
pub type Wu8 = Wrapping<u8>;

/// The ring of integers modulo 2<sup>16</sup>.
pub type Wu16 = Wrapping<u16>;

/// The ring of integers modulo 2<sup>32</sup>.
pub type Wu32 = Wrapping<u32>;

/// The ring of integers modulo 2<sup>64</sup>.
pub type Wu64 = Wrapping<u64>;

/// The ring of integers modulo 2<sup>128</sup>.
pub type Wu128 = Wrapping<u128>;

/// The ring of integers modulo the pointer size.
pub type Wusize = Wrapping<usize>;

/// The ring of integers modulo 2<sup>8</sup>.
pub type Wi8 = Wrapping<i8>;

/// The ring of integers modulo 2<sup>16</sup>.
pub type Wi16 = Wrapping<i16>;

/// The ring of integers modulo 2<sup>32</sup>.
pub type Wi32 = Wrapping<i32>;

/// The ring of integers modulo 2<sup>64</sup>.
pub type Wi64 = Wrapping<i64>;

/// The ring of integers modulo 2<sup>128</sup>.
pub type Wi128 = Wrapping<i128>;

/// The ring of integers modulo the pointer size.
pub type Wisize = Wrapping<isize>;
