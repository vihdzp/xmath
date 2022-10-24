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
//! - Any valid inputs must return a valid output without invoking undefined
//!   behavior.
//!
//! This last condition restricts the possible trait implementations quite
//! heavily. For instance, `u8` can't implement [`Add`](basic::Add) since
//! addition can overflow.

pub mod basic;
pub mod matrix;
