use super::Nat;
use crate::{data::Sign, traits::basic::*};

/// A variable-sized (signed) integer.
///
/// ## Internal representation
///
/// This contains a [`Nat`] field with the absolute value of the number, and a
/// [`Sign`] field with its sign. Any combinations of these are allowed, with
/// the following caveat:
///
/// - If either of the absolute value or sign is zero, so must the other be.
///
/// This guarantees that equality of the type coincides with element-wise
/// equality.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Int {
    /// The absolute value of the integer.
    abs: Nat,

    /// The sign of the integer.
    sgn: Sign,
}

impl Zero for Int {
    fn zero() -> Self {
        Self {
            abs: Nat::zero(),
            sgn: Sign::ZERO,
        }
    }

    fn is_zero(&self) -> bool {
        self.sgn().is_zero()
    }
}

impl Int {
    /// Initializes a new integer with the given absolute value and sign.
    ///
    /// ## Safety
    ///
    /// If either of the absolute value or sign is zero, so must the other be.
    pub unsafe fn new_unchecked(abs: Nat, sgn: Sign) -> Self {
        Self { abs, sgn }
    }

    /// Initializes a nonnegative integer with a given absolute value.
    pub fn new_pos(abs: Nat) -> Self {
        if abs.is_zero() {
            Self::zero()
        } else {
            // Safety: the type invariant has been verified.
            unsafe { Self::new_unchecked(abs, Sign::POS) }
        }
    }

    /// Initializes a nonpositive integer with a given absolute value.
    pub fn new_neg(abs: Nat) -> Self {
        if abs.is_zero() {
            Self::zero()
        } else {
            // Safety: the type invariant has been verified.
            unsafe { Self::new_unchecked(abs, Sign::NEG) }
        }
    }

    /// The absolute value of the integer.
    pub fn abs(&self) -> &Nat {
        &self.abs
    }

    /// The sign of the integer.
    pub fn sgn(&self) -> Sign {
        self.sgn
    }

    /// A mutable reference to the absolute value of the integer.
    ///
    /// ## Safety
    ///
    /// The type invariant must be preserved.
    pub unsafe fn abs_mut(&mut self) -> &mut Nat {
        &mut self.abs
    }

    /// A mutable reference to the sign of the integer.
    ///
    /// ## Safety
    ///
    /// The type invariant must be preserved.
    pub unsafe fn sgn_mut(&mut self) -> &mut Sign {
        &mut self.sgn
    }
}

impl From<Nat> for Int {
    fn from(abs: Nat) -> Self {
        Self::new_pos(abs)
    }
}

impl One for Int {
    fn one() -> Self {
        Nat::one().into()
    }

    fn is_one(&self) -> bool {
        self.sgn().is_one() && self.abs().is_one()
    }
}

impl ZeroNeOne for Int {}

impl Neg for Int {
    fn neg(&self) -> Self {
        // Safety: this operation preserves the sign invariant.
        unsafe { Self::new_unchecked(self.abs().clone(), self.sgn().neg()) }
    }
}

// missing ADD

impl Mul for Int {
    fn mul(&self, x: &Self) -> Self {
        // Safety: this operation preserves the sign invariant.
        unsafe {
            Self::new_unchecked(self.abs().mul(x.abs()), self.sgn() * x.sgn())
        }
    }
}
