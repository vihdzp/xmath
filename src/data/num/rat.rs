use super::{Int, Nat};
use crate::{data::aliases::NonZero, traits::basic::*};

/// A variable-sized rational number.
///
/// These are stored as a fraction in lowest terms, so operations using this
/// type will generally be much more expensive than their floating-point
/// counterparts.
///
/// ## Internal representation
///
/// This contains an [`Int`] field for the numerator and a [`NonZero<Nat>`]
/// field for the denominator. These must always be coprime, that is, they must
/// not share any prime factors.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rat {
    /// The numerator.
    num: Int,

    /// The denominator.
    den: NonZero<Nat>,
}

impl Rat {
    /// Initializes a rational number with the given numerator and denominator.
    ///
    /// ## Safety
    ///
    /// These numbers must be coprime.
    pub unsafe fn new_unchecked(num: Int, den: NonZero<Nat>) -> Self {
        Self { num, den }
    }

    /// Initializes the rational `num / den`.
    pub fn new(_num: Int, _den: NonZero<Nat>) -> Self {
        todo!()
    }

    /// The numerator of the rational.
    pub fn num(&self) -> &Int {
        &self.num
    }

    /// The denominator of the rational.
    pub fn den(&self) -> &NonZero<Nat> {
        &self.den
    }

    /// A mutable reference to the numerator of the rational.
    ///
    /// ## Safety
    ///
    /// The type invariant must be preserved.
    pub unsafe fn num_mut(&mut self) -> &mut Int {
        &mut self.num
    }

    /// A mutable reference to the sign of the integer.
    ///
    /// ## Safety
    ///
    /// The type invariant must be preserved.
    pub unsafe fn den_mut(&mut self) -> &mut NonZero<Nat> {
        &mut self.den
    }
}

impl From<Int> for Rat {
    fn from(num: Int) -> Self {
        // Safety: 1 is coprime with any integer.
        unsafe { Self::new_unchecked(num, NonZero::one()) }
    }
}

impl From<Nat> for Rat {
    fn from(value: Nat) -> Self {
        Int::into(value.into())
    }
}

impl Zero for Rat {
    fn zero() -> Self {
        Int::zero().into()
    }

    fn is_zero(&self) -> bool {
        self.num().is_zero()
    }
}

impl One for Rat {
    fn one() -> Self {
        Int::one().into()
    }

    fn is_one(&self) -> bool {
        self.num().is_one() && self.den().is_one()
    }
}
