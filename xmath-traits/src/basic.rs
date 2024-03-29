//! Defines the basic traits forming the algebraic hierarchy of χ-math.
//!
//! ## [`std::ops`] traits
//!
//! There are obvious parallels between traits in this module and traits in Rust
//! itself. For instance [`Add`] corresponds to [`std::ops::Add`], and [`Neg`]
//! corresponds to [`std::ops::Neg`], and so on. Aside from the extra guarantees
//! that this library's traits have, the main difference is that Rust's traits
//! pass by value, while ours pass by reference. This means ours are more suited
//! to working with dynamically sized types.
//!
//! We also provide extra functions such as [`double`](Add::double),
//! which for certain types can be optimized more than the default
//! implementation.
//!
//! As a rule of thumb, we only implement the Rust traits on our custom types
//! whenever these are statically sized.
//!
//! ## Additive and multiplicative traits
//!
//! Instead of implementing traits for arbitrary operations, we choose to only
//! consider "additive" and "multiplicative" operations. This means that many
//! traits have an additive or multiplicative counterpart, such as [`Add`] and
//! [`Mul`], or [`Zero`] and [`One`].

use super::*;
use std::num::Wrapping;

/// A trait for a `0` value.
///
/// The difference between this trait and
/// [`num_traits::Zero`](https://docs.rs/num-traits/latest/num_traits/identities/trait.Zero.html)
/// is that we don't require any algebraic properties on the zero value – it can
/// be entirely arbitrary. See [`AddMonoid`] for that usage.
///
/// Its multiplicative counterpart is the [`One`] trait.
pub trait Zero: PartialEq + Sized {
    /// The zero value of the type.
    fn zero() -> Self;

    /// Compares a value to zero.
    fn is_zero(&self) -> bool {
        self == &Self::zero()
    }

    /// Sets a value to zero.
    fn set_zero(&mut self) {
        *self = Self::zero();
    }
}

/// Implements [`Zero`] for the primitive types.
macro_rules! impl_zero {
    ($($x: ty), *) => {
        $(impl Zero for $x {
            fn zero() -> Self {
                0 as Self
            }
        }

        impl Zero for Wrapping<$x> {
            fn zero() -> Self {
                Self(<$x>::zero())
            }
        })*
    };
}

impl_zero!(u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize, f32, f64);

/// A trait for a `1` value.
///
/// The difference between this trait and
/// [`num_traits::One`](https://docs.rs/num-traits/latest/num_traits/identities/trait.One.html)
/// is that we don't require any algebraic properties on the one value – it can
/// be entirely arbitrary. See [`MulMonoid`] for that usage.
///
/// Its additive counterpart is the [`Zero`] trait.
pub trait One: PartialEq + Sized {
    /// The one value of the type.
    fn one() -> Self;

    /// Compares a value to one.
    fn is_one(&self) -> bool {
        self == &Self::one()
    }

    /// Sets a value to one.
    fn set_one(&mut self) {
        *self = Self::one();
    }
}

/// Implements [`One`] for the primitive types.
macro_rules! impl_one {
    ($($x: ty), *) => {
        $(impl One for $x {
            fn one() -> Self {
                1 as Self
            }
        }

        impl One for Wrapping<$x> {
            fn one() -> Self {
                Self(<$x>::one())
            }
        })*
    };
}

impl_one!(u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize, f32, f64);

/// Types where `0 != 1`.
pub trait ZeroNeOne: Zero + One + Eq {}

/// Implements [`ZeroNeOne`] for the primitive types.
macro_rules! impl_zero_ne_one {
    ($($x: ty), *) => {
        $(impl ZeroNeOne for $x {}
        impl ZeroNeOne for Wrapping<$x> {})*
    };
}

impl_zero_ne_one!(u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize);

/// Negation on a type.
///
/// If you want to define [`std::ops::Neg`] along with this trait, you should
/// define this one first, and then use the [`derive_neg`](crate::derive_neg)
/// macro.
///
/// Its multiplicative counterpart is the [`Inv`] trait.
pub trait Neg: Sized {
    /// The negation function on the type.
    fn neg(&self) -> Self;

    /// The negation-assignment function on the type.
    ///
    /// The default implementation is `*self = self.neg()`.
    fn neg_mut(&mut self) {
        *self = self.neg();
    }
}

/// Implements [`Neg`] for the primitive types.
macro_rules! impl_neg {
    ($($x: ty), *) => {
        $(impl Neg for $x {
            fn neg(&self) -> Self {
                -*self
            }
        })*
    };
}

impl_neg!(
    i8, i16, i32, i64, i128, isize, f32, f64, Wu8, Wu16, Wu32, Wu64, Wu128, Wusize, Wi8, Wi16,
    Wi32, Wi64, Wi128, Wisize
);

/// Derives [`std::ops::Neg`] from [`Neg`].
#[macro_export]
macro_rules! derive_neg {
    ($($x: ty), *) => {
        $(impl std::ops::Neg for &$x {
            type Output = $x;

            fn neg(self) -> Self::Output {
                xmath_traits::Neg::neg(self)
            }
        }

        impl std::ops::Neg for $x {
            type Output = Self;

            fn neg(self) -> Self::Output {
                -&self
            }
        })*
    };
}

/// A multiplicative inverse on a type.
///
/// Its additive counterpart is the [`Neg`] trait.
pub trait Inv: Sized {
    /// The multiplicative inverse on the type.
    fn inv(&self) -> Self;

    /// The multiplicative inverse assignment function on the type.
    fn inv_mut(&mut self) {
        *self = self.inv();
    }
}

/// Implements [`Inv`] for the primitive types.
macro_rules! impl_inv {
    ($($x: ty), *) => {
        $(impl Inv for $x {
            fn inv(&self) -> Self {
                1.0 / *self
            }
        })*
    };
}

impl_inv!(f32, f64);

/// Addition on a type.
///
/// If you want to define [`std::ops::Add`] along with this trait, you should
/// define this one first, and then use the [`derive_add`](crate::derive_add)
/// macro.
///
/// Its multiplicative counterpart is the [`Mul`] trait.
pub trait Add: Clone {
    /// The addition function on the type.
    fn add(&self, rhs: &Self) -> Self;

    /// The (left) addition-assignment function on the type.
    fn add_mut(&mut self, rhs: &Self) {
        *self = self.add(rhs);
    }

    /// The (right) addition-assignment function on the type.
    fn add_rhs_mut(&self, rhs: &mut Self) {
        *rhs = self.add(rhs);
    }

    /// Doubles a value.
    fn double(&self) -> Self {
        self.add(self)
    }

    /// Doubles a value in place.
    fn double_mut(&mut self) {
        *self = self.double();
    }
}

/// Implements [`Add`] for the primitive types.
macro_rules! impl_add {
    ($($x: ty), *) => {
        $(impl Add for $x {
            fn add(&self, x : &Self) -> Self {
                *self + *x
            }
        })*
    };
}

impl_add!(Wu8, Wu16, Wu32, Wu64, Wu128, Wusize, Wi8, Wi16, Wi32, Wi64, Wi128, Wisize, f32, f64);

/// Derives [`std::ops::Add`] and [`std::ops::AddAssign`] from [`Add`].
#[macro_export]
macro_rules! derive_add {
    ($($x: ty), *) => {
        $(impl std::ops::Add for &$x {
            type Output = $x;

            fn add(self, rhs: Self) -> Self::Output {
                xmath_traits::Add::add(self, rhs)
            }
        }

        impl std::ops::Add<$x> for &$x {
            type Output = $x;

            fn add(self, rhs: $x) -> Self::Output {
                self + &rhs
            }
        }

        impl std::ops::Add<&$x> for $x {
            type Output = Self;

            fn add(self, rhs: &Self) -> Self::Output {
                &self + rhs
            }
        }

        impl std::ops::Add for $x {
            type Output = Self;

            fn add(self, rhs: Self) -> Self::Output {
                &self + &rhs
            }
        }

        impl std::ops::AddAssign<&$x> for $x {
            fn add_assign(&mut self, rhs: &Self) {
                xmath_traits::Add::add_mut(self, &rhs);
            }
        }

        impl std::ops::AddAssign for $x {
            fn add_assign(&mut self, rhs: Self) {
                *self += &rhs;
            }
        })*
    };
}

/// Multiplication on a type.
///
/// If you want to define [`std::ops::Mul`] along with this trait, you should
/// define this one first, and then use the [`derive_mul`](crate::derive_mul)
/// macro.
///
/// Its additive counterpart is the [`Add`] trait.
pub trait Mul: Clone {
    /// The multiplication function on the type.
    fn mul(&self, rhs: &Self) -> Self;

    /// The (left) multiplication-assignment function on the type.
    fn mul_mut(&mut self, rhs: &Self) {
        *self = self.mul(rhs);
    }

    /// The (right) multiplication-assignment function on the type.
    fn mul_rhs_mut(&self, rhs: &mut Self) {
        *rhs = self.mul(rhs);
    }

    /// Squares a value.
    fn sq(&self) -> Self {
        self.mul(self)
    }

    /// Squares a value in place.
    fn sq_mut(&mut self) {
        *self = self.sq();
    }
}

/// Implements [`Mul`] for the primitive types.
macro_rules! impl_mul {
    ($($x: ty), *) => {
        $(impl Mul for $x {
            fn mul(&self, rhs: &Self) -> Self {
                *self * *rhs
            }
        })*
    };
}

impl_mul!(Wu8, Wu16, Wu32, Wu64, Wu128, Wusize, Wi8, Wi16, Wi32, Wi64, Wi128, Wisize, f32, f64);

/// Derives [`std::ops::Mul`] and [`std::ops::MulAssign`] from [`Mul`].
#[macro_export]
macro_rules! derive_mul {
    ($($x: ty), *) => {
        $(impl std::ops::Mul for &$x {
            type Output = $x;

            fn mul(self, rhs: Self) -> Self::Output {
                xmath_traits::Mul::mul(self, rhs)
            }
        }

        impl std::ops::Mul<$x> for &$x {
            type Output = $x;

            fn mul(self, rhs: $x) -> Self::Output {
                self * &rhs
            }
        }

        impl std::ops::Mul<&$x> for $x {
            type Output = Self;

            fn mul(self, rhs: &Self) -> Self::Output {
                &self * rhs
            }
        }

        impl std::ops::Mul for $x {
            type Output = Self;

            fn mul(self, rhs: Self) -> Self::Output {
                &self * &rhs
            }
        }

        impl std::ops::MulAssign<&$x> for $x {
            fn mul_assign(&mut self, rhs: &Self) {
                xmath_traits::Mul::mul_mut(self, &rhs);
            }
        }

        impl std::ops::MulAssign for $x {
            fn mul_assign(&mut self, rhs: Self) {
                *self *= &rhs;
            }
        })*
    };
}

/// Commutative addition.
///
/// Its multiplicative counterpart is the [`CommMul`] trait.
pub trait CommAdd: Add + Eq {}

/// Implements [`CommAdd`] for the primitive types.
macro_rules! impl_comm_add {
    ($($x: ty), *) => {
        $(impl CommAdd for $x {})*
    };
}

impl_comm_add!(Wu8, Wu16, Wu32, Wu64, Wu128, Wusize, Wi8, Wi16, Wi32, Wi64, Wi128, Wisize);

/// Commutative multiplication.
///
/// Its additive counterpart is the [`CommAdd`] trait.
pub trait CommMul: Mul + Eq {}

/// Implements [`CommMul`] for the primitive types.
macro_rules! impl_comm_mul {
    ($($x: ty), *) => {
        $(impl CommMul for $x {})*
    };
}

impl_comm_mul!(Wu8, Wu16, Wu32, Wu64, Wu128, Wusize, Wi8, Wi16, Wi32, Wi64, Wi128, Wisize);

/// Subtraction on a type.
///
/// If you want to define [`std::ops::Sub`] along with this trait, you should
/// define this one first, and then use the [`derive_sub`](crate::derive_div)
/// macro.
///
/// Its multiplicative counterpart is the [`Div`] trait.
pub trait Sub: Add + Neg {
    /// The subtraction function on the type.
    fn sub(&self, rhs: &Self) -> Self {
        self.add(&rhs.neg())
    }

    /// The (left) subtraction-assignment function on the type.
    fn sub_mut(&mut self, rhs: &Self) {
        self.add_mut(&rhs.neg());
    }

    /// The (right) subtraction-assignment function on the type.
    fn sub_rhs_mut(&self, rhs: &mut Self) {
        rhs.neg_mut();
        self.add_rhs_mut(rhs);
    }
}

/// Implements [`Sub`] for the primitive types.
macro_rules! impl_sub {
    ($($x: ty), *) => {
        $(impl Sub for $x {})*
    };
}

impl_sub!(Wu8, Wu16, Wu32, Wu64, Wu128, Wusize, Wi8, Wi16, Wi32, Wi64, Wi128, Wisize, f32, f64);

/// Derives [`std::ops::Sub`] and [`std::ops::SubAssign`] from [`Sub`].
#[macro_export]
macro_rules! derive_sub {
    ($($x: ty), *) => {
        $(impl std::ops::Sub for &$x {
            type Output = $x;

            fn sub(self, rhs: Self) -> Self::Output {
                xmath_traits::Sub::sub(self, rhs)
            }
        }

        impl std::ops::Sub<$x> for &$x {
            type Output = $x;

            fn sub(self, rhs: $x) -> Self::Output {
                self - &rhs
            }
        }

        impl std::ops::Sub<&$x> for $x {
            type Output = Self;

            fn sub(self, rhs: &Self) -> Self::Output {
                &self - rhs
            }
        }

        impl std::ops::Sub for $x {
            type Output = Self;

            fn sub(self, rhs: Self) -> Self::Output {
                &self - &rhs
            }
        }

        impl std::ops::SubAssign<&$x> for $x {
            fn sub_assign(&mut self, rhs: &Self) {
                xmath_traits::Sub::sub_mut(self, &rhs);
            }
        }

        impl std::ops::SubAssign for $x {
            fn sub_assign(&mut self, rhs: Self) {
                *self -= &rhs;
            }
        })*
    };
}

/// Division on a type.
///
/// If you want to define [`std::ops::Div`] along with this trait, you should
/// define this one first, and then use the [`derive_div`](crate::derive_div)
/// macro.
///
/// Its additive counterpart is the [`Sub`] trait.
pub trait Div: Mul + Inv {
    /// The division function on the type.
    fn div(&self, rhs: &Self) -> Self {
        self.mul(&rhs.inv())
    }

    /// The (left) division-assignment function on the type.
    fn div_mut(&mut self, rhs: &Self) {
        self.mul_mut(&rhs.inv())
    }

    /// The (right) division-assignment function on the type.
    fn div_rhs_mut(&self, rhs: &mut Self) {
        rhs.inv_mut();
        self.mul_rhs_mut(rhs);
    }
}

/// Implements [`Div`] for the primitive types.
macro_rules! impl_div {
    ($($x: ty), *) => {
        $(impl Div for $x {})*
    };
}

impl_div!(f32, f64);

/// Derives [`std::ops::Div`] and [`std::ops::DivAssign`] from [`Div`].
#[macro_export]
macro_rules! derive_div {
    ($($x: ty), *) => {
        $(impl std::ops::Div for &$x {
            type Output = $x;

            fn div(self, rhs: Self) -> Self::Output {
                xmath_traits::Div::div(self, rhs)
            }
        }

        impl std::ops::Div<$x> for &$x {
            type Output = $x;

            fn div(self, rhs: $x) -> Self::Output {
                self / &rhs
            }
        }

        impl std::ops::Div<&$x> for $x {
            type Output = Self;

            fn div(self, rhs: &Self) -> Self::Output {
                &self / rhs
            }
        }

        impl std::ops::Div for $x {
            type Output = Self;

            fn div(self, rhs: Self) -> Self::Output {
                &self / &rhs
            }
        }

        impl std::ops::DivAssign<&$x> for $x {
            fn div_assign(&mut self, rhs: &Self) {
                xmath_traits::Div::div_mut(self, &rhs);
            }
        }

        impl std::ops::DivAssign for $x {
            fn div_assign(&mut self, rhs: Self) {
                *self /= &rhs;
            }
        })*
    };
}

/// A trait for integral domains.
///
/// These are types implementing [`Mul`] and [`Zero`] such that the product of
/// nonzero elements is nonzero.
pub trait IntegralDomain: Mul + Zero + Eq {}

/// A trait for additive monoids.
///
/// These are types implementing [`Add`] and [`Zero`] such that
///
/// - The addition function is associative, meaning
///   `(x + y) + z == x + (y + z)` for any `x`, `y`, `z`.
/// - The identity `x + 0 == 0 + x == x` holds for any `x`.
///
/// Its multiplicative counterpart is [`MulMonoid`].
pub trait AddMonoid: Add + Zero + Eq {}

/// Implements [`AddMonoid`] for the primitive types.
macro_rules! impl_add_monoid {
    ($($x: ty), *) => {
        $(impl AddMonoid for $x {})*
    };
}

impl_add_monoid!(Wu8, Wu16, Wu32, Wu64, Wu128, Wi8, Wi16, Wi32, Wi64, Wi128);

/// A trait for multiplicative monoids.
///
/// These are types implementing [`Mul`] and [`One`] such that
///
/// - The multiplication function is associative, meaning
///   `(x * y) * z == x * (y * z)` for any `x`, `y`, `z`.
/// - The identity `x * 1 == 1 * x == x` holds for any `x`.
///
/// Its additive counterpart is [`AddMonoid`].
pub trait MulMonoid: Mul + One + Eq {}

/// Implements [`MulMonoid`] for the primitive types.
macro_rules! impl_mul_monoid {
    ($($x: ty), *) => {
        $(impl MulMonoid for $x {})*
    };
}

impl_mul_monoid!(Wu8, Wu16, Wu32, Wu64, Wu128, Wi8, Wi16, Wi32, Wi64, Wi128);

/// A trait for additive groups.
///
/// These are types implementing [`AddMonoid`] and [`Neg`] such that
/// `x - x == 0` for all `x`.
///
/// Its multiplicative counterpart is [`MulGroup`].
pub trait AddGroup: AddMonoid + Sub {}

/// Implements [`AddGroup`] for the primitive types.
macro_rules! impl_add_group {
    ($($x: ty), *) => {
        $(impl AddGroup for $x {})*
    };
}

impl_add_group!(Wu8, Wu16, Wu32, Wu64, Wu128, Wi8, Wi16, Wi32, Wi64, Wi128);

/// A trait for multiplicative groups.
///
/// These are types implementing [`MulMonoid`] and [`Inv`] such that
/// `x / x == 1` for all `x`.
///
/// Its additive counterpart is [`AddGroup`].
pub trait MulGroup: MulMonoid + Div {}

/// A trait for (multiplicative) groups with zero.
///
/// These are multiplicative monoids with a `0` such that
///
/// - `0 * x == x * 0 == 0` for any `x`.
/// - Nonzero elements are invertible.
pub trait ZeroGroup: MulMonoid + IntegralDomain + Zero {
    /// The type of nonzero elements of this type.
    type Nonzero: NonZeroWrapper<Inner = Self> + MulGroup;

    /// Inverts a nonzero element.
    ///
    /// ## Safety
    ///
    /// The caller must ensure the element is indeed nonzero.
    unsafe fn try_inv_unchecked(&self) -> Self::Nonzero {
        Self::Nonzero::to_ref(self).inv()
    }

    /// Attempts to invert an element. Returns `None` if zero.
    fn try_inv(&self) -> Option<Self::Nonzero> {
        if self.is_zero() {
            None
        } else {
            // Safety: we just verified the element is nonzero.
            unsafe { Some(self.try_inv_unchecked()) }
        }
    }

    /// Inverts a nonzero element in place.
    ///
    /// ## Safety
    ///
    /// The caller must ensure the element is indeed nonzero.
    unsafe fn try_inv_mut_unchecked(&mut self) {
        Self::Nonzero::to_mut(self).inv_mut();
    }

    /// Attempts to invert an element in place. Does nothing if zero. Returns
    /// whether the element was inverted.
    fn try_inv_mut(&mut self) -> bool {
        if self.is_zero() {
            false
        } else {
            // Safety: we just verified the element is nonzero.
            unsafe { self.try_inv_mut_unchecked() };
            true
        }
    }

    /// Divides an element by a nonzero element.
    ///
    /// ## Safety
    ///
    /// The caller must ensure the `rhs` is indeed nonzero.
    unsafe fn try_div_unchecked(&self, rhs: &Self) -> Self {
        self.mul(rhs.try_inv_unchecked().as_ref())
    }

    /// Attempts to divide an element by another. Returns `None` if the second
    /// element is zero.
    fn try_div(&self, rhs: &Self) -> Option<Self> {
        if rhs.is_zero() {
            None
        } else {
            // Safety: we just verified the element is nonzero.
            unsafe { Some(self.try_div_unchecked(rhs)) }
        }
    }

    /// Divides an element by a nonzero element in place.
    ///
    /// ## Safety
    ///
    /// The caller must ensure the `rhs` is indeed nonzero.
    unsafe fn try_div_mut_unchecked(&mut self, rhs: &Self) {
        self.mul_mut(rhs.try_inv_unchecked().as_ref());
    }

    /// Attempts to divide an element by another in place. Does nothing if the
    /// second element is zero. Returns whether a division was performed.
    fn try_div_mut(&mut self, rhs: &Self) -> bool {
        if rhs.is_zero() {
            false
        } else {
            // Safety: we just verified the element is nonzero.
            unsafe { self.try_div_mut_unchecked(rhs) };
            true
        }
    }

    /// Divides an element by a nonzero element in place.
    ///
    /// ## Safety
    ///
    /// The caller must ensure the `rhs` is indeed nonzero.
    unsafe fn try_div_rhs_mut_unchecked(&self, rhs: &mut Self) {
        rhs.try_inv_mut_unchecked();
        self.mul_rhs_mut(rhs);
    }

    /// Attempts to divide an element by another in place. Does nothing if the
    /// second element is zero. Returns whether a division was performed.
    fn try_div_rhs_mut(&self, rhs: &mut Self) -> bool {
        if rhs.is_zero() {
            false
        } else {
            // Safety: we just verified the element is nonzero.
            unsafe { self.try_div_rhs_mut_unchecked(rhs) };
            true
        }
    }
}

/// A trait for (commutative and unitary) rings.
///
/// These are types implementing [`AddGroup`], [`MulMonoid`], [`CommAdd`], and
/// [`CommMul`], such that `x * (y + z) = x * y + x * z` for any `x`, `y`, `z`.
pub trait Ring: AddGroup + MulMonoid + CommAdd + CommMul {}

/// Implements [`Ring`] for the primitive types.
macro_rules! impl_ring {
    ($($x: ty), *) => {
        $(impl Ring for $x {})*
    };
}

impl_ring!(Wu8, Wu16, Wu32, Wu64, Wu128, Wi8, Wi16, Wi32, Wi64, Wi128);

/// A trait for fields.
///
/// These are types implementing [`Ring`], [`ZeroGroup`], and [`ZeroNeOne`].
pub trait Field: Ring + ZeroGroup + ZeroNeOne {}
