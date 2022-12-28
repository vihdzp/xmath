use super::Nat;
use crate::{data::Sign, traits::*};
use std::cmp::Ordering::*;

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

impl From<u32> for Int {
    fn from(abs: u32) -> Self {
        Self::new_pos(abs.into())
    }
}

impl From<i32> for Int {
    fn from(abs: i32) -> Self {
        if abs < 0 {
            Self::new_neg(((-abs) as u32).into())
        } else {
            (abs as u32).into()
        }
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

/// A convenience macro for defining an integer from its underlying array, in a
/// manner similar to [`vec`]. 
/// 
/// Prefixing the list by `M` yields the negative.
///
/// ## Examples
///
/// ```
/// # use xmath::{int, nat};
/// # use xmath::data::Int;
/// # use xmath::traits::Zero;
/// assert_eq!(int!(0), Int::zero());
/// assert_eq!(int!(M 0), Int::zero());
/// assert_eq!(int!(5), Int::from(5u32));
/// assert_eq!(int!(0, 1), Int::from(nat![0, 1]));
/// assert_eq!(int!(M 2, 5), Int::new_neg(nat![2, 5]));
/// ```
#[macro_export]
macro_rules! int {
    ($($xs: expr),*) => {
        xmath::data::Int::from(xmath::nat!($($xs),*))
    };
    (M $($xs: expr),*) => {
        xmath::data::Int::new_neg(xmath::nat!($($xs),*))
    }
}

impl Neg for Int {
    fn neg(&self) -> Self {
        // Safety: this operation preserves the sign invariant.
        unsafe { Self::new_unchecked(self.abs().clone(), self.sgn().neg()) }
    }

    fn neg_mut(&mut self) {
        // Safety: this operation preserves the sign invariant.
        unsafe {
            self.sgn_mut().neg_mut();
        }
    }
}

/// Adds two integers. The absolute values and signs are unbundled, which means
/// this works for both addition and subtraction.
///
/// ## Safety
///
/// The natural numbers and their corresponding signs are subject to the same
/// invariant as [`Int`].
unsafe fn add_sign(x: &Nat, xs: Sign, y: &Nat, ys: Sign) -> Int {
    match xs {
        Sign::ZERO => Int::new_unchecked(y.clone(), ys),
        Sign::POS => match ys {
            Sign::ZERO => Int::new_unchecked(x.clone(), xs),
            Sign::POS => Int::new_unchecked(x.add(y), Sign::POS),
            Sign::NEG => match x.cmp(y) {
                Greater => Int::new_unchecked(x.monus(y), Sign::POS),
                Equal => Int::zero(),
                Less => Int::new_unchecked(y.monus(x), Sign::NEG),
            },
        },
        Sign::NEG => match ys {
            Sign::ZERO => Int::new_unchecked(x.clone(), xs),
            Sign::NEG => Int::new_unchecked(x.add(y), Sign::NEG),
            Sign::POS => match x.cmp(y) {
                Greater => Int::new_unchecked(x.monus(y), Sign::NEG),
                Equal => Int::zero(),
                Less => Int::new_unchecked(y.monus(x), Sign::POS),
            },
        },
    }
}

/// Adds two integers, assigns to the first. The absolute values and signs are
/// unbundled, which means this works for both addition and subtraction.
///
/// ## Safety
///
/// The natural numbers and their corresponding signs are subject to the same
/// invariant as [`Int`].
unsafe fn add_sign_mut(x: &mut Nat, xs: &mut Sign, y: &Nat, ys: Sign) {
    match xs {
        Sign::ZERO => (*x, *xs) = (y.clone(), ys),
        Sign::POS => match ys {
            Sign::ZERO => {}
            Sign::POS => x.add_mut(y),
            Sign::NEG => match (*x).cmp(y) {
                Greater => x.monus_mut(y),
                Equal => (*x, *xs) = (Nat::zero(), Sign::ZERO),
                Less => {
                    *xs = Sign::NEG;
                    y.monus_rhs_mut(x);
                }
            },
        },
        Sign::NEG => match ys {
            Sign::ZERO => {}
            Sign::NEG => x.add_mut(y),
            Sign::POS => match (*x).cmp(y) {
                Greater => x.monus_mut(y),
                Equal => (*x, *xs) = (Nat::zero(), Sign::ZERO),
                Less => {
                    *xs = Sign::POS;
                    y.monus_rhs_mut(x);
                }
            },
        },
    }
}

impl Add for Int {
    fn add(&self, rhs: &Self) -> Self {
        unsafe { add_sign(self.abs(), self.sgn(), rhs.abs(), rhs.sgn()) }
    }

    fn add_mut(&mut self, rhs: &Self) {
        unsafe {
            add_sign_mut(&mut self.abs, &mut self.sgn, rhs.abs(), rhs.sgn())
        }
    }

    fn add_rhs_mut(&self, rhs: &mut Self) {
        unsafe {
            add_sign_mut(&mut rhs.abs, &mut rhs.sgn, self.abs(), self.sgn())
        }
    }
}

impl Sub for Int {
    fn sub(&self, rhs: &Self) -> Self {
        unsafe { add_sign(self.abs(), self.sgn(), rhs.abs(), -rhs.sgn()) }
    }

    fn sub_mut(&mut self, rhs: &Self) {
        unsafe {
            add_sign_mut(&mut self.abs, &mut self.sgn, rhs.abs(), -rhs.sgn())
        }
    }

    fn sub_rhs_mut(&self, rhs: &mut Self) {
        unsafe {
            rhs.sgn_mut().neg_mut();
            add_sign_mut(&mut rhs.abs, &mut rhs.sgn, self.abs(), self.sgn())
        }
    }
}

impl Mul for Int {
    fn mul(&self, x: &Self) -> Self {
        // Safety: this operation preserves the sign invariant.
        unsafe {
            Self::new_unchecked(self.abs().mul(x.abs()), self.sgn() * x.sgn())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Auxiliary method for testing addition.
    fn add_test(mut x: Int, y: Int, z: Int) {
        // Immutable addition.
        assert_eq!(x.add(&y), z);

        // Mutable addition.
        let mut x0 = x.clone();
        x0.add_mut(&y);
        assert_eq!(x0, z);

        // Mutable addition (rhs).
        y.add_rhs_mut(&mut x);
        assert_eq!(x, z);
    }

    /// Performs various test additions.
    #[test]
    fn add() {
        add_test(int!(5), int!(M 5), int!(0));
        add_test(int!(1), int!(M 2), int!(M 1));
        add_test(int!(M u32::MAX), int!(M 1), int!(M 0, 1));

        add_test(
            int!(M 2990080069, 3717786483, 3347647318),
            int!(2174112409, 4183093485, 906683806),
            int!(M 815967660, 3829660294, 2440963511),
        )
    }
}
