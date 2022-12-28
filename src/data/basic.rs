//! This file contains all of the most basic types and their trait
//! implementations.

use super::aliases::NonZero;
use crate::{traits::*, *};
use std::fmt::{Display, Write};
use std::num::Wrapping;

/// The ring of integers modulo 2⁸.
pub type Wu8 = Wrapping<u8>;

/// The ring of integers modulo 2¹⁶.
pub type Wu16 = Wrapping<u16>;

/// The ring of integers modulo 2³².
pub type Wu32 = Wrapping<u32>;

/// The ring of integers modulo 2⁶⁴.
pub type Wu64 = Wrapping<u64>;

/// The ring of integers modulo 2¹²⁸.
pub type Wu128 = Wrapping<u128>;

/// The ring of integers modulo the pointer size.
pub type Wusize = Wrapping<usize>;

/// The ring of integers modulo 2⁸.
pub type Wi8 = Wrapping<i8>;

/// The ring of integers modulo 2¹⁶.
pub type Wi16 = Wrapping<i16>;

/// The ring of integers modulo 2³².
pub type Wi32 = Wrapping<i32>;

/// The ring of integers modulo 2⁶⁴.
pub type Wi64 = Wrapping<i64>;

/// The ring of integers modulo 2¹²⁸.
pub type Wi128 = Wrapping<i128>;

/// The ring of integers modulo the pointer size.
pub type Wisize = Wrapping<isize>;

/// The trivial structure with a single element.
///
/// ## Internal representation
///
/// This is a unit type.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct I;

impl Zero for I {
    fn zero() -> Self {
        Self
    }
}

impl One for I {
    fn one() -> Self {
        Self
    }
}

impl Neg for I {
    fn neg(&self) -> Self {
        Self
    }
}

derive_neg!(I);

impl Inv for I {
    fn inv(&self) -> Self {
        Self
    }
}

impl Add for I {
    fn add(&self, _: &Self) -> Self {
        Self
    }
}

derive_add!(I);

impl Mul for I {
    fn mul(&self, _: &Self) -> Self {
        Self
    }
}

derive_mul!(I);

impl IntegralDomain for I {}
impl CommAdd for I {}
impl CommMul for I {}
impl Sub for I {}

derive_sub!(I);

impl Div for I {}

derive_div!(I);

impl AddMonoid for I {}
impl MulMonoid for I {}
impl AddGroup for I {}
impl MulGroup for I {}
impl Ring for I {}

/// A pair of elements.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pair<T, U>(pub T, pub U);

/// The pair (0, 0).
impl<T: Zero, U: Zero> Zero for Pair<T, U> {
    fn zero() -> Self {
        Self(T::zero(), U::zero())
    }

    fn is_zero(&self) -> bool {
        self.0.is_zero() && self.1.is_zero()
    }
}

/// The pair (1, 1).
impl<T: One, U: One> One for Pair<T, U> {
    fn one() -> Self {
        Self(T::one(), U::one())
    }

    fn is_one(&self) -> bool {
        self.0.is_one() && self.1.is_one()
    }
}

// Really one of these suffices, but unfortunately we can't implement this.
impl<T: ZeroNeOne, U: ZeroNeOne> ZeroNeOne for Pair<T, U> {}

/// Element-wise negation.
impl<T: Neg, U: Neg> Neg for Pair<T, U> {
    fn neg(&self) -> Self {
        Self(self.0.neg(), self.1.neg())
    }

    fn neg_mut(&mut self) {
        self.0.neg_mut();
        self.1.neg_mut();
    }
}

/// Element-wise inverse.
impl<T: Inv, U: Inv> Inv for Pair<T, U> {
    fn inv(&self) -> Self {
        Self(self.0.inv(), self.1.inv())
    }

    fn inv_mut(&mut self) {
        self.0.inv_mut();
        self.1.inv_mut();
    }
}

impl<T: Add, U: Add> Add for Pair<T, U> {
    fn add(&self, x: &Self) -> Self {
        Self(self.0.add(&x.0), self.1.add(&x.1))
    }

    fn add_mut(&mut self, x: &Self) {
        self.0.add_mut(&x.0);
        self.1.add_mut(&x.1);
    }

    fn double(&self) -> Self {
        Self(self.0.double(), self.1.double())
    }

    fn double_mut(&mut self) {
        self.0.double_mut();
        self.1.double_mut();
    }
}

impl<T: CommAdd, U: CommAdd> CommAdd for Pair<T, U> {}
impl<T: AddMonoid, U: AddMonoid> AddMonoid for Pair<T, U> {}

impl<T: Sub, U: Sub> Sub for Pair<T, U> {
    fn sub(&self, x: &Self) -> Self {
        Self(self.0.sub(&x.0), self.1.sub(&x.1))
    }

    fn sub_mut(&mut self, x: &Self) {
        self.0.sub_mut(&x.0);
        self.1.sub_mut(&x.1);
    }
}

impl<T: AddGroup, U: AddGroup> AddGroup for Pair<T, U> {}

impl<T: Mul, U: Mul> Mul for Pair<T, U> {
    fn mul(&self, x: &Self) -> Self {
        Self(self.0.mul(&x.0), self.1.mul(&x.1))
    }

    fn mul_mut(&mut self, x: &Self) {
        self.0.mul_mut(&x.0);
        self.1.mul_mut(&x.1);
    }

    fn sq(&self) -> Self {
        Self(self.0.sq(), self.1.sq())
    }

    fn sq_mut(&mut self) {
        self.0.sq_mut();
        self.1.sq_mut();
    }
}

impl<T: CommMul, U: CommMul> CommMul for Pair<T, U> {}
impl<T: MulMonoid, U: MulMonoid> MulMonoid for Pair<T, U> {}

impl<T: Div, U: Div> Div for Pair<T, U> {
    fn div(&self, x: &Self) -> Self {
        Self(self.0.div(&x.0), self.1.div(&x.1))
    }

    fn div_mut(&mut self, x: &Self) {
        self.0.div_mut(&x.0);
        self.1.div_mut(&x.1);
    }
}

impl<T: MulGroup, U: MulGroup> MulGroup for Pair<T, U> {}
impl<T: Ring, U: Ring> Ring for Pair<T, U> {}

/// The field with two elements. This may be thought of as the integers mod 2.
///
/// ## Internal representation
///
/// This is a structure with a single `bool` field. The value `false` stands for
/// `0` while `true` stands for `1`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct F2(bool);

impl From<bool> for F2 {
    fn from(x: bool) -> Self {
        Self(x)
    }
}

impl From<F2> for bool {
    fn from(x: F2) -> Self {
        x.0
    }
}

impl Display for F2 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if bool::from(*self) {
            f.write_char('1')
        } else {
            f.write_char('0')
        }
    }
}

impl F2 {
    /// Returns a reference to the inner boolean value.
    pub fn as_bool_mut(&mut self) -> &mut bool {
        &mut self.0
    }

    /// An alias for `add`.
    pub fn xor(self, x: Self) -> Self {
        (bool::from(self) ^ bool::from(x)).into()
    }

    /// An alias for `add_mut`.
    pub fn xor_mut(&mut self, x: Self) {
        *self = self.xor(x);
    }

    /// An alias for `mul`.
    pub fn and(self, x: Self) -> Self {
        (bool::from(self) && bool::from(x)).into()
    }

    /// An alias for `mul_mut`.
    pub fn and_mut(&mut self, x: Self) {
        *self = self.and(x);
    }

    /// Returns `1` if either value is `1`, `0` otherwise.
    pub fn or(self, x: Self) -> Self {
        (bool::from(self) || bool::from(x)).into()
    }

    /// Assigns `1` if either value is `1`, `0` otherwise.
    pub fn or_mut(&mut self, x: Self) {
        *self = self.or(x);
    }
}

impl std::ops::Not for F2 {
    type Output = F2;

    fn not(self) -> Self::Output {
        (!bool::from(self)).into()
    }
}

impl Zero for F2 {
    fn zero() -> Self {
        false.into()
    }

    fn is_zero(&self) -> bool {
        !bool::from(*self)
    }
}

impl One for F2 {
    fn one() -> Self {
        true.into()
    }

    fn is_one(&self) -> bool {
        bool::from(*self)
    }
}

impl ZeroNeOne for F2 {}

impl Neg for F2 {
    fn neg(&self) -> Self {
        *self
    }
}

derive_neg!(F2);

impl Add for F2 {
    fn add(&self, x: &Self) -> Self {
        self.xor(*x)
    }

    fn double(&self) -> Self {
        Self::zero()
    }
}

derive_add!(F2);

impl Mul for F2 {
    fn mul(&self, x: &Self) -> Self {
        self.and(*x)
    }

    fn sq(&self) -> Self {
        *self
    }
}

derive_mul!(F2);

impl CommAdd for F2 {}
impl CommMul for F2 {}
impl Sub for F2 {}

derive_sub!(F2);

impl AddMonoid for F2 {}
impl MulMonoid for F2 {}
impl AddGroup for F2 {}
impl IntegralDomain for F2 {}

derive_mul!(NonZero<F2>);

impl Default for NonZero<F2> {
    fn default() -> Self {
        Self::one()
    }
}

impl Inv for NonZero<F2> {
    fn inv(&self) -> Self {
        *self
    }
}

impl Div for NonZero<F2> {
    fn div(&self, _: &Self) -> Self {
        Self::one()
    }

    fn div_mut(&mut self, _: &Self) {}
}

derive_div!(NonZero<F2>);

impl MulGroup for NonZero<F2> {}
impl ZeroGroup for F2 {}
impl Ring for F2 {}
impl Field for F2 {}

/// The field with three elements. This may be thought of as the integers mod 3.
///
/// ## Internal representation
///
/// This is an enum backed by a `u8` with values `ZERO = 0`, `ONE = 1`, and
/// `TWO = 2`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum F3 {
    /// The zero value.
    #[default]
    ZERO = 0,

    /// The one value.
    ONE = 1,

    /// The two value.
    TWO = 2,
}

impl From<u8> for F3 {
    fn from(x: u8) -> Self {
        // Safety: `F3` has the same representation as `u8`, and any possible
        // value of `x % 3` is valid for `F3`.
        unsafe { std::mem::transmute(x % 3) }
    }
}

impl Display for F3 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            F3::ZERO => f.write_char('0'),
            F3::ONE => f.write_char('1'),
            F3::TWO => f.write_char('2'),
        }
    }
}

impl Zero for F3 {
    fn zero() -> Self {
        Self::ZERO
    }
}

impl One for F3 {
    fn one() -> Self {
        Self::ONE
    }
}

impl ZeroNeOne for F3 {}

impl Neg for F3 {
    fn neg(&self) -> Self {
        (3 - *self as u8).into()
    }
}

derive_neg!(F3);

impl Add for F3 {
    fn add(&self, x: &Self) -> Self {
        (*self as u8 + *x as u8).into()
    }
}

derive_add!(F3);

impl Mul for F3 {
    fn mul(&self, x: &Self) -> Self {
        (*self as u8 * *x as u8).into()
    }
}

derive_mul!(F3);

impl CommAdd for F3 {}
impl CommMul for F3 {}
impl Sub for F3 {}

derive_sub!(F3);

impl AddMonoid for F3 {}
impl MulMonoid for F3 {}
impl AddGroup for F3 {}
impl IntegralDomain for F3 {}

impl NonZero<F3> {
    /// The one value.
    pub const ONE: Self = unsafe { NonZero::new_unchecked(F3::ONE) };

    /// The two value.
    pub const TWO: Self = unsafe { NonZero::new_unchecked(F3::TWO) };
}

derive_mul!(NonZero<F3>);

impl Inv for NonZero<F3> {
    fn inv(&self) -> Self {
        *self
    }
}

impl Div for NonZero<F3> {}

derive_div!(NonZero<F3>);

impl MulGroup for NonZero<F3> {}
impl ZeroGroup for F3 {}
impl Ring for F3 {}
impl Field for F3 {}

/// The sign of an integer. As a mathematical structure, this is a [`Field`]
/// isomorphic to [`F3`].
///
/// ## Internal representation
///
/// This is an enum backed by an `i8` with values `ZERO = 0`, `POS = 1`, and
/// `NEG = -1`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(i8)]
pub enum Sign {
    /// The zero value.
    #[default]
    ZERO = 0,

    /// The positive value.
    POS = 1,

    /// The negative value.
    NEG = -1,
}

impl From<i8> for Sign {
    fn from(x: i8) -> Self {
        // Safety: `F3` has the same representation as `u8`, and any possible
        // value of this expression is valid for `F3` (we cast to `i16` to avoid
        // overflows).
        unsafe { std::mem::transmute(((x as i16 + 1) % 3 - 1) as i8) }
    }
}

/// The field isomorphism between [`F3`] and [`Sign`].
impl From<F3> for Sign {
    fn from(value: F3) -> Self {
        match value {
            F3::ZERO => Self::ZERO,
            F3::ONE => Self::POS,
            F3::TWO => Self::NEG,
        }
    }
}

/// The field isomorphism between [`F3`] and [`Sign`].
impl From<Sign> for F3 {
    fn from(value: Sign) -> Self {
        match value {
            Sign::ZERO => Self::ZERO,
            Sign::POS => Self::ONE,
            Sign::NEG => Self::TWO,
        }
    }
}

impl Display for Sign {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Sign::ZERO => f.write_char('0'),
            Sign::POS => f.write_char('1'),
            Sign::NEG => f.write_str("-1"),
        }
    }
}

impl Zero for Sign {
    fn zero() -> Self {
        Self::ZERO
    }
}

impl One for Sign {
    fn one() -> Self {
        Self::POS
    }
}

impl ZeroNeOne for Sign {}

impl Neg for Sign {
    fn neg(&self) -> Self {
        (-*self as i8).into()
    }
}

derive_neg!(Sign);

impl Add for Sign {
    fn add(&self, x: &Self) -> Self {
        (*self as i8 + *x as i8).into()
    }
}

derive_add!(Sign);

impl Mul for Sign {
    fn mul(&self, x: &Self) -> Self {
        (*self as i8 * *x as i8).into()
    }
}

derive_mul!(Sign);

impl CommAdd for Sign {}
impl CommMul for Sign {}
impl Sub for Sign {}

derive_sub!(Sign);

impl AddMonoid for Sign {}
impl MulMonoid for Sign {}
impl AddGroup for Sign {}
impl IntegralDomain for Sign {}

impl NonZero<Sign> {
    /// The one value.
    pub const POS: Self = unsafe { NonZero::new_unchecked(Sign::POS) };

    /// The negative one value.
    pub const NEG: Self = unsafe { NonZero::new_unchecked(Sign::NEG) };
}

derive_mul!(NonZero<Sign>);

impl Inv for NonZero<Sign> {
    fn inv(&self) -> Self {
        *self
    }
}

impl Div for NonZero<Sign> {}

derive_div!(NonZero<Sign>);

impl MulGroup for NonZero<Sign> {}
impl ZeroGroup for Sign {}
impl Ring for Sign {}
impl Field for Sign {}

/// The field with four elements. It consists of values `0`, `1`, `X`, `Y`.
///
/// ## Internal representation
///
/// This is an enum with values `ZERO = 0`, `ONE = 1`, `X = 2`, `Y = 3`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum F4 {
    /// The zero value.
    #[default]
    ZERO = 0,

    /// The one value.
    ONE = 1,

    /// Either of the values other than `0` and `1`.
    X = 2,

    /// Either of the values other than `0` and `1`.
    Y = 3,
}

impl F4 {
    /// Takes the last two bytes of an `u8` as an `F4`. This is not the same as
    /// the `u8` cast which can only take either `0` or `1`.
    pub fn from_u8_raw(x: u8) -> Self {
        // Safety: `F4` has the same representation as `u8`, and any possible
        // value of `x % 4` is valid for `F4`.
        unsafe { std::mem::transmute(x % 4) }
    }
}

impl From<u8> for F4 {
    fn from(x: u8) -> Self {
        Self::from_u8_raw(x % 2)
    }
}

impl Display for F4 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            F4::ZERO => f.write_char('0'),
            F4::ONE => f.write_char('1'),
            F4::X => f.write_char('X'),
            F4::Y => f.write_char('Y'),
        }
    }
}

impl Zero for F4 {
    fn zero() -> Self {
        Self::ZERO
    }
}

impl One for F4 {
    fn one() -> Self {
        Self::ONE
    }
}

impl ZeroNeOne for F4 {}

impl Neg for F4 {
    fn neg(&self) -> Self {
        *self
    }
}

derive_neg!(F4);

impl Add for F4 {
    fn add(&self, x: &Self) -> Self {
        Self::from_u8_raw(*self as u8 ^ *x as u8)
    }

    fn double(&self) -> Self {
        Self::ZERO
    }
}

derive_add!(F4);

impl Mul for F4 {
    fn mul(&self, x: &Self) -> Self {
        match self {
            Self::ZERO => Self::ZERO,
            Self::ONE => *x,
            Self::X => match *x {
                Self::ZERO => Self::ZERO,
                Self::ONE => Self::X,
                Self::X => Self::Y,
                Self::Y => Self::ONE,
            },
            Self::Y => match *x {
                Self::ZERO => Self::ZERO,
                Self::ONE => Self::Y,
                Self::X => Self::ONE,
                Self::Y => Self::X,
            },
        }
    }
}

derive_mul!(F4);

impl CommAdd for F4 {}
impl CommMul for F4 {}
impl Sub for F4 {}

derive_sub!(F4);

impl AddMonoid for F4 {}
impl MulMonoid for F4 {}
impl AddGroup for F4 {}
impl IntegralDomain for F4 {}

impl NonZero<F4> {
    /// The one value.
    pub const ONE: Self = unsafe { NonZero::new_unchecked(F4::ONE) };

    /// Either of the values other than `0` and `1`.
    pub const X: Self = unsafe { NonZero::new_unchecked(F4::X) };

    /// Either of the values other than `0` and `1`.
    pub const Y: Self = unsafe { NonZero::new_unchecked(F4::Y) };
}

derive_mul!(NonZero<F4>);

impl Inv for NonZero<F4> {
    fn inv(&self) -> Self {
        match self.as_ref() {
            F4::ZERO => unreachable!(),
            F4::ONE => Self::ONE,
            F4::X => Self::Y,
            F4::Y => Self::X,
        }
    }
}

impl Div for NonZero<F4> {}

derive_div!(NonZero<F4>);

impl MulGroup for NonZero<F4> {}
impl ZeroGroup for F4 {}
impl Ring for F4 {}
impl Field for F4 {}
