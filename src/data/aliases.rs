//! Declares various type aliases, which change the structure a type is endowed
//! with.

use xmath_macro::{ArrayFromIter, Transparent};
use xmath_traits::*;

/// A type alias that endows a type with additive operations instead of
/// multiplicative ones.
///
/// This allows us to implement algorithms for addition, and have them
/// immediately in multiplicative contexts.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Transparent, ArrayFromIter,
)]
pub struct Additive<T>(pub T);

impl<T> Additive<T> {
    /// Initializes a new value.
    pub fn new(x: T) -> Self {
        Self(x)
    }
}

/// Multiplicative one becomes additive zero.
impl<T: One> Zero for Additive<T> {
    fn zero() -> Self {
        Self::new(T::one())
    }

    fn is_zero(&self) -> bool {
        self.0.is_one()
    }
}

/// Multiplicative inverse becomes negation.
impl<T: Inv> Neg for Additive<T> {
    fn neg(&self) -> Self {
        Self::new(self.0.inv())
    }

    fn neg_mut(&mut self) {
        self.0.inv_mut()
    }
}

/// Multiplication becomes addition.
impl<T: Mul> Add for Additive<T> {
    fn add(&self, x: &Self) -> Self {
        Self::new(self.0.mul(&x.0))
    }

    fn add_mut(&mut self, x: &Self) {
        self.0.mul_mut(&x.0)
    }

    fn double_mut(&mut self) {
        self.0.sq_mut()
    }
}

impl<T: MulMonoid> AddMonoid for Additive<T> {}

/// Division becomes subtraction.
impl<T: Div> Sub for Additive<T> {
    fn sub(&self, x: &Self) -> Self {
        Self::new(self.0.div(&x.0))
    }

    fn sub_mut(&mut self, x: &Self) {
        self.0.div_mut(&x.0)
    }
}

impl<T: MulGroup> AddGroup for Additive<T> {}

/// A type alias that endows a type with multiplicative operations instead of
/// additive ones.
///
/// This allows us to implement algorithms for multiplication, and have them
/// immediately in additive contexts.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Transparent, ArrayFromIter,
)]
pub struct Multiplicative<T>(pub T);

impl<T> Multiplicative<T> {
    /// Initializes a new value.
    pub fn new(x: T) -> Self {
        Self(x)
    }
}

/// Multiplicative zero becomes one.
impl<T: Zero> One for Multiplicative<T> {
    fn one() -> Self {
        Self::new(T::zero())
    }

    fn is_one(&self) -> bool {
        self.0.is_zero()
    }
}

/// Negation becomes multiplicative inverse.
impl<T: Neg> Inv for Multiplicative<T> {
    fn inv(&self) -> Self {
        Self::new(self.0.neg())
    }

    fn inv_mut(&mut self) {
        self.0.neg_mut()
    }
}

/// Addition becomes multiplication.
impl<T: Add> Mul for Multiplicative<T> {
    fn mul(&self, x: &Self) -> Self {
        Self::new(self.0.add(&x.0))
    }

    fn mul_mut(&mut self, x: &Self) {
        self.0.add_mut(&x.0)
    }

    fn sq_mut(&mut self) {
        self.0.double_mut()
    }
}

impl<T: AddMonoid> MulMonoid for Multiplicative<T> {}

/// Subtraction becomes division.
impl<T: Sub> Div for Multiplicative<T> {
    fn div(&self, x: &Self) -> Self {
        Self::new(self.0.sub(&x.0))
    }

    fn div_mut(&mut self, x: &Self) {
        self.0.sub_mut(&x.0)
    }
}

impl<T: AddGroup> MulGroup for Multiplicative<T> {}

