//! Defines the type [`Nat`] of variable-sized natural numbers, and the basic
//! operations on them.

use crate::data::{trim, Poly};
use crate::traits::*;

/// Gets the `i`-th dword from a number `x`.
macro_rules! get_dword {
    ($x: expr, $i: expr) => {
        ($x >> 32 * $i & 0xFFFFFFFF) as u32
    };
}

/// A variable-sized natural number.
///
/// ## Internal representation
///
/// This type has a single `Poly<u32>` field. It represents the little-endian
/// base 2³² expansion of the number. It's subject to the same type invariant as
/// [`Poly`].
#[derive(Clone, Default, Debug, PartialEq, Eq)]
pub struct Nat(pub Poly<u32>);

impl From<Poly<u32>> for Nat {
    fn from(p: Poly<u32>) -> Self {
        Self(p)
    }
}

impl From<Vec<u32>> for Nat {
    fn from(v: Vec<u32>) -> Self {
        Self(v.into())
    }
}

impl From<u32> for Nat {
    fn from(x: u32) -> Self {
        Poly::c(x).into()
    }
}

impl From<u64> for Nat {
    fn from(x: u64) -> Self {
        vec![get_dword!(x, 0), get_dword!(x, 1)].into()
    }
}

impl From<u128> for Nat {
    fn from(x: u128) -> Self {
        vec![
            get_dword!(x, 0),
            get_dword!(x, 1),
            get_dword!(x, 2),
            get_dword!(x, 3),
        ]
        .into()
    }
}

impl AsRef<Poly<u32>> for Nat {
    fn as_ref(&self) -> &Poly<u32> {
        &self.0
    }
}

impl AsMut<Poly<u32>> for Nat {
    fn as_mut(&mut self) -> &mut Poly<u32> {
        &mut self.0
    }
}

impl Nat {
    /// Creates a new `Nat` from a `Poly<u32>`.
    pub fn new(p: Poly<u32>) -> Self {
        p.into()
    }

    /// Returns a reference to the inner slice.
    pub fn as_slice(&self) -> &[u32] {
        self.as_ref().as_ref()
    }

    /// The base 2³² digit count for the number.
    pub fn len(&self) -> usize {
        self.as_slice().len()
    }

    /// Returns whether the underlying slice is empty.
    pub fn is_empty(&self) -> bool {
        self.as_slice().is_empty()
    }

    /// Returns the `i`-th digit of the number.
    pub fn digit(&self, i: usize) -> u32 {
        *self.as_slice().get(i).unwrap_or(&0)
    }
}

impl Zero for Nat {
    fn zero() -> Self {
        Self::default()
    }

    fn is_zero(&self) -> bool {
        self.is_empty()
    }
}

impl One for Nat {
    fn one() -> Self {
        Poly::one().into()
    }

    fn is_one(&self) -> bool {
        self.len() == 1 && self.digit(0) == 1
    }
}

impl ZeroNeOne for Nat {}

/// A convenience macro for defining a natural from its underlying array.
///
/// ## Examples
///
/// ```
/// # use xmath::nat;
/// # use xmath::data::Nat;
/// # use xmath::traits::basic::Zero;
/// assert_eq!(nat!(0), Nat::zero());
/// assert_eq!(nat!(5), Nat::from(5u32));
/// assert_eq!(nat!(0, 1), Nat::from(vec![0, 1]));
/// ```
#[macro_export]
macro_rules! nat {
    (0) => {
        Nat::zero()
    };
    ($x: expr) => {
        Nat::from($x as u32)
    };
    ($($xs: expr),*) => {
        Nat::from(vec![$($xs),*])
    }
}

/// Orders two references by a specified comparison function.
pub fn sort_by<'a, T, U: PartialOrd, F: Fn(&T) -> U>(
    x: &'a T,
    y: &'a T,
    f: F,
) -> (&'a T, &'a T) {
    if f(x) < f(y) {
        (x, y)
    } else {
        (y, x)
    }
}

/// Performs the operations `lhs + rhs + carry` and returns the lower 32 bits
/// and whether there was a carry.
const fn carrying_add(lhs: u32, rhs: u32, carry: bool) -> (u32, bool) {
    // This operation is bounded by `2 × 2³² - 1`.
    let res = (lhs as u64) + (rhs as u64) + (carry as u64);
    (get_dword!(res, 0), res > u32::MAX as u64)
}

/// Performs the operation `lhs * rhs + carry1 + carry2` and returns the lower
/// and upper 32 bits of the result, in that order.
const fn carrying_mul(
    lhs: u32,
    rhs: u32,
    carry1: u32,
    carry2: u32,
) -> (u32, u32) {
    // This operation is bounded by `2⁶⁴ - 1`.
    let res = (lhs as u64) * (rhs as u64) + (carry1 as u64) + (carry2 as u64);
    (get_dword!(res, 0), get_dword!(res, 1))
}

/// An auxiliary method. If the carry flag is set, adds one to `s` and
/// writes the output in `v`. Otherwise just copies `s` to `v`.
fn carry_push_add(s: &[u32], v: &mut Vec<u32>, mut carry: bool) {
    let mut i = 0;

    // We push zeros unless there's no more carry. We optimize around the
    // fact that each carry is very unlikely.
    while carry {
        let x = *s.get(i).unwrap_or(&0);

        if x == u32::MAX {
            v.push(0);
        } else {
            v.push(x + 1);
            carry = false;
        }

        i += 1;
    }

    // Copy the rest of the array.
    while i < s.len() {
        v.push(s[i]);
    }
}

/// Performs the operations `lhs - rhs - carry` and returns the lower 32 bits
/// and whether there was a carry.
const fn carrying_sub(lhs: u32, rhs: u32, carry: bool) -> (u32, bool) {
    // This operation is bounded by `-2³²`.
    let res = (lhs as i64) - (rhs as i64) - (carry as i64);
    (get_dword!(res, 0), res.is_negative())
}

/// An auxiliary method. If the carry flag is set, subtracts one to `s` and
/// writes the output in `v`. Otherwise just copies `s` to `v`.
///
/// Returns whether the computation is successful, i.e. does not underflow.
fn carry_push_sub(s: &[u32], v: &mut Vec<u32>, mut carry: bool) -> bool {
    let mut i = 0;

    // We push zeros unless there's no more carry. We optimize around the
    // fact that each carry is very unlikely.
    while carry {
        match s.get(i) {
            Some(0) => v.push(u32::MAX),
            Some(x) => {
                v.push(x - 1);
                carry = false;
            }
            None => {
                return false;
            }
        }

        i += 1;
    }

    // Copy the rest of the array.
    while i < s.len() {
        v.push(s[i]);
    }

    true
}

impl Add for Nat {
    fn add(&self, z: &Self) -> Self {
        // The shorter and longer number.
        let (x, y) = sort_by(self, z, Nat::len);

        let mut res = Vec::with_capacity(y.len() + 1);
        let mut carry = false;

        // Adds digits one by one, until the first number runs out.
        for i in 0..x.len() {
            let (d, b) = carrying_add(x.digit(i), y.digit(i), carry);
            res.push(d);
            carry = b;
        }

        // Continues until the second number runs out.
        carry_push_add(&y.as_slice()[x.len()..], &mut res, carry);

        // Safety: `carry_push` can't end in a `0`.
        unsafe { Poly::new_unchecked(res) }.into()
    }

    fn add_mut(&mut self, x: &Self) {
        unsafe {
            let mut carry = false;

            // Modify digits one by one, until our space runs out.
            for i in 0..self.len().min(x.len()) {
                let (d, b) = carrying_add(self.digit(i), x.digit(i), carry);
                self.0.as_slice_mut()[i] = d;
                carry = b;
            }

            // If the first number is at most as long as the second.
            if self.len() <= x.len() {
                // Then keep pushing the next digits.
                carry_push_add(
                    &x.0.as_slice()[self.len()..],
                    self.0.as_vec_mut(),
                    carry,
                );
            } else {
                // Otherwise, we keep overwriting the number.
                self.0.as_vec_mut().push(0);
                let mut i = x.len();

                while carry {
                    if self.digit(i) == u32::MAX {
                        self.0.as_slice_mut()[i] = 0;
                    } else {
                        self.0.as_slice_mut()[i] += 1;
                        carry = false;
                    }

                    i += 1;
                }

                self.0.as_vec_mut().truncate(i + 1);
            }
        }
    }

    fn add_rhs_mut(&self, rhs: &mut Self) {
        rhs.add_mut(self);
    }

    fn double(&self) -> Self {
        self.shl(1)
    }

    fn double_mut(&mut self) {
        self.shl_mut(1);
    }
}

impl AddMonoid for Nat {}
impl CommAdd for Nat {}

impl Mul for Nat {
    fn mul(&self, x: &Self) -> Self {
        if self.is_zero() || x.is_zero() {
            return Self::zero();
        }

        // Buffer for the result.
        let mut res = vec![0; self.len() + x.len()];

        // We multiply every two digits together and keep track of the carry.
        for (j, &d2) in x.as_slice().iter().enumerate() {
            let mut carry = 0;

            for (i, &d1) in self.as_slice().iter().enumerate() {
                (res[i + j], carry) = carrying_mul(d1, d2, carry, res[i + j]);
            }

            res[self.len() + j] = carry;
        }

        res.into()
    }
}

impl MulMonoid for Nat {}
impl CommMul for Nat {}

impl Nat {
    /// Multiplies the number by a `u32` value.
    pub fn mul_u32(&self, value: u32) -> Self {
        if value == 0 {
            return Self::zero();
        }

        // Buffer for the result.
        let mut res = Vec::new();
        let mut carry = 0;

        // We multiply our value by every digit and add the carry.
        for &digit in self.as_slice() {
            let new_digit;
            (new_digit, carry) = carrying_mul(digit, value, carry, 0);
            res.push(new_digit);
        }

        if carry != 0 {
            res.push(carry);
        }

        // Safety: the leading digit is guaranteed to be nonzero.
        unsafe { Poly::new_unchecked(res) }.into()
    }

    /// Multiplies the number by a `u32` value in place.
    pub fn mul_u32_mut(&mut self, value: u32) {
        if value == 0 {
            *self = Self::zero();
            return;
        }

        let mut carry = 0;

        // Safety: the leading digit is guaranteed to be nonzero.
        unsafe {
            for digit in self.0.as_slice_mut() {
                (*digit, carry) = carrying_mul(*digit, value, carry, 0);
            }

            if carry != 0 {
                self.0.as_vec_mut().push(carry);
            }
        }
    }

    /// Shifts right by a certain amount of dwords.
    pub fn shr_32(&self, dwords: usize) -> Self {
        self.as_slice()[dwords..].to_owned().into()
    }

    /// Shifts right by a certain amount of dwords in place.
    pub fn shr_32_mut(&mut self, dwords: usize) {
        if dwords >= self.len() {
            *self = Self::zero();
        }

        // Safety: we don't change the leading coefficient.
        unsafe {
            let len = self.len();
            for i in dwords..len {
                self.0.as_slice_mut()[i - dwords] = self.as_slice()[i];
            }

            self.0.as_vec_mut().truncate(len - dwords);
        }
    }

    /// Shifts right by a number of bits.
    pub fn shr(&self, bits: usize) -> Self {
        let (q, r) = (bits / 32, bits % 32);

        if r == 0 {
            self.shr_32(q)
        } else {
            let mut res = Vec::new();

            for i in q..self.len() {
                res.push((self.digit(i) >> r) + (self.digit(i + 1) << (32 - r)))
            }

            res.into()
        }
    }

    /// Shifts right by a number of bits in place.
    pub fn shr_mut(&mut self, bits: usize) {
        let (q, r) = (bits / 32, bits % 32);

        if r == 0 {
            self.shr_32_mut(q);
        } else {
            // Safety: we trim the vector at the end.
            unsafe {
                for i in 0..self.len() {
                    self.0.as_slice_mut()[i] = (self.digit(q + i) >> r)
                        + (self.digit(q + i + 1) << (32 - r));
                }

                trim(self.0.as_vec_mut());
            }
        }
    }

    /// Shifts left by a certain amount of dwords.
    pub fn shl_32(&self, dwords: usize) -> Self {
        let mut res = vec![0; dwords];
        res.extend_from_slice(self.as_slice());
        res.into()
    }

    /// Shifts left by a certain amount of dwords in place.
    pub fn shl_32_mut(&mut self, dwords: usize) {
        // Safety: we don't change the leading coefficient.
        unsafe {
            let len = self.len();
            self.0.as_vec_mut().resize(len + dwords, 0);

            for i in (0..len).rev() {
                self.0.as_slice_mut()[i + dwords] = self.as_slice()[i];
            }

            for i in 0..dwords {
                self.0.as_slice_mut()[i] = 0;
            }
        }
    }

    /// Shifts left by a number of bits.
    pub fn shl(&self, bits: usize) -> Self {
        let (q, r) = (bits / 32, bits % 32);

        if r == 0 {
            self.shl_32(q)
        } else {
            let mut res = vec![0; q];
            res.push(self.digit(0) << r);

            for i in 0..self.len() {
                res.push((self.digit(i + 1) << r) + (self.digit(i) >> (32 - r)))
            }

            res.into()
        }
    }

    /// Shifts left by a number of bits in place.
    pub fn shl_mut(&mut self, bits: usize) {
        let (q, r) = (bits / 32, bits % 32);

        if r == 0 {
            self.shl_32_mut(q);
        } else {
            // Safety: we trim the vector at the end.
            unsafe {
                let len = self.len();
                self.0.as_vec_mut().resize(len + q + 1, 0);

                for i in (0..len).rev() {
                    self.0.as_slice_mut()[q + i + 1] =
                        (self.digit(i + 1) << r) + (self.digit(i) >> (32 - r));
                }
                self.0.as_slice_mut()[q] = self.digit(0) << r;

                for i in 0..q {
                    self.0.as_slice_mut()[i] = 0;
                }

                trim(self.0.as_vec_mut());
            }
        }
    }

    /// Bitwise and.
    pub fn and(&self, rhs: &Self) -> Self {
        self.as_ref().pairwise(rhs.as_ref(), |x, y| x & y).into()
    }

    /// Bitwise and-assign.
    pub fn and_mut(&mut self, rhs: &Self) {
        self.as_mut().pairwise_mut(rhs.as_ref(), |x, y| *x &= y);
    }

    /// Bitwise or.
    pub fn or(&self, rhs: &Self) -> Self {
        self.as_ref().pairwise(rhs.as_ref(), |x, y| x | y).into()
    }

    /// Bitwise or-assign.
    pub fn or_mut(&mut self, rhs: &Self) {
        self.as_mut().pairwise_mut(rhs.as_ref(), |x, y| *x |= y);
    }

    /// Bitwise xor.
    pub fn xor(&self, rhs: &Self) -> Self {
        self.as_ref().pairwise(rhs.as_ref(), |x, y| x ^ y).into()
    }

    /// Bitwise xor-assign.
    pub fn xor_mut(&mut self, rhs: &Self) {
        self.as_mut().pairwise_mut(rhs.as_ref(), |x, y| *x ^= y);
    }
}

impl Ord for Nat {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Compares by length.
        let res = self.len().cmp(&other.len());
        if res != std::cmp::Ordering::Equal {
            return res;
        }

        // Compares entries in inverse order.
        for (i, j) in self.as_slice().iter().zip(other.as_slice()).rev() {
            let res = i.cmp(j);
            if res != std::cmp::Ordering::Equal {
                return res;
            }
        }

        std::cmp::Ordering::Equal
    }
}

impl PartialOrd for Nat {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Nat {
    /// The monus or truncated subtraction of two naturals. Returns `0` if
    /// `rhs > self`.
    pub fn monus(&self, rhs: &Self) -> Self {
        if self.len() < rhs.len() {
            return Self::zero();
        }

        let mut res = Vec::with_capacity(self.len());
        let mut carry = false;

        // Subtracts digits one by one, until the first number runs out.
        for i in 0..rhs.len() {
            let (d, b) = carrying_sub(self.digit(i), rhs.digit(i), carry);
            res.push(d);
            carry = b;
        }

        // Continues until the second number runs out.
        if carry_push_sub(&self.as_slice()[rhs.len()..], &mut res, carry) {
            res.into()
        } else {
            Self::zero()
        }
    }

    /// The monus or truncated subtraction of two naturals, performed in place.
    /// Assigns `0` if `rhs > self`.
    pub fn monus_mut(&mut self, rhs: &Self) {
        if self.len() < rhs.len() {
            *self = Self::zero();
            return;
        }

        unsafe {
            let mut carry = false;

            // Modify digits one by one, until our space runs out.
            for i in 0..rhs.len() {
                let (d, b) = carrying_sub(self.digit(i), rhs.digit(i), carry);
                self.0.as_slice_mut()[i] = d;
                carry = b;
            }

            // We keep overwriting the number.
            let mut i = rhs.len();

            while carry {
                match self.as_slice().get(i) {
                    Some(0) => {
                        self.0.as_slice_mut()[i] = u32::MAX;
                    }
                    Some(_) => {
                        self.0.as_slice_mut()[i] -= 1;
                        carry = false;
                    }
                    None => {
                        *self = Self::zero();
                        return;
                    }
                }

                i += 1;
            }

            trim(self.0.as_vec_mut())
        }
    }

    /// The monus or truncated subtraction of two naturals, performed in place.
    /// Assigns `0` if `rhs > self`.
    pub fn monus_rhs_mut(&self, rhs: &mut Self) {
        if self.len() < rhs.len() {
            *rhs = Self::zero();
            return;
        }

        unsafe {
            let mut carry = false;

            // Modify digits one by one, until our space runs out.
            for i in 0..rhs.len() {
                let (d, b) = carrying_sub(self.digit(i), rhs.digit(i), carry);
                rhs.0.as_slice_mut()[i] = d;
                carry = b;
            }

            // We keep overwriting the number.
            let mut i = rhs.len();

            while carry {
                match self.as_slice().get(i) {
                    Some(0) => {
                        rhs.0.as_slice_mut()[i] = u32::MAX;
                    }
                    Some(_) => {
                        rhs.0.as_slice_mut()[i] -= 1;
                        carry = false;
                    }
                    None => {
                        *rhs = Self::zero();
                        return;
                    }
                }

                i += 1;
            }

            trim(rhs.0.as_vec_mut())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Auxiliary method for testing addition.
    fn add_test(mut x: Nat, y: Nat, z: Nat) {
        // Immutable addition.
        assert_eq!(x.add(&y), z);

        // Mutable addition.
        x.add_mut(&y);
        assert_eq!(x, z);
    }

    /// Performs various test additions.
    #[test]
    fn add() {
        add_test(nat!(1), nat!(2), nat!(3));
        add_test(nat!(u32::MAX), nat!(1), nat!(0, 1));
        add_test(nat!(u32::MAX, u32::MAX), nat!(1), nat!(0, 0, 1));

        add_test(
            nat!(1585670998, 4230785404),
            nat!(1068143055, 774776642, 3494593297),
            nat!(2653814053, 710594750, 3494593298),
        )
    }

    /// Auxiliary method for testing multiplication by a `u32`.
    fn mul_u32_test(mut x: Nat, y: u32, z: Nat) {
        // Immutable multiplication.
        assert_eq!(x.mul_u32(y), z);

        // Mutable multiplication.
        x.mul_u32_mut(y);
        assert_eq!(x, z);
    }

    /// Performs various test multiplications by a `u32`.
    #[test]
    fn mul_u32() {
        mul_u32_test(nat!(3), 4, nat!(12));
        mul_u32_test(nat!(0), 5, nat!(0));
        mul_u32_test(nat!(6), 0, nat!(0));
        mul_u32_test(nat!(0x80000000), 2, nat!(0, 1));

        mul_u32_test(
            nat!(4099396007, 1653631812, 2225176924),
            1365017072,
            nat!(2180887440, 2341071018, 1912336926, 707200842),
        );
    }

    /// Auxiliary method for testing a right bit shift.
    fn shr_test(mut x: Nat, y: usize, z: Nat) {
        // Immutable shift.
        assert_eq!(x.shr(y), z);

        // Mutable shift.
        x.shr_mut(y);
        assert_eq!(x, z);
    }

    /// Performs various test right shifts.
    #[test]
    fn shr() {
        shr_test(nat!(55), 3, nat!(6));
        shr_test(nat!(2, 1), 0, nat!(2, 1));
        shr_test(nat!(2, 1), 1, nat!(0x80000001));
        shr_test(nat!(79, 1), 32, nat!(1));
        shr_test(nat!(79, 2), 33, nat!(1));

        shr_test(
            nat!(3759743582, 161857823, 2806339866),
            47,
            nat!(2989757259, 85642),
        )
    }

    /// Auxiliary method for testing a left bit shift.
    fn shl_test(mut x: Nat, y: usize, z: Nat) {
        // Immutable shift.
        assert_eq!(x.shl(y), z);

        // Mutable shift.
        x.shl_mut(y);
        assert_eq!(x, z);
    }

    /// Performs various test right shifts.
    #[test]
    fn shl() {
        shl_test(nat!(55), 3, nat!(440));
        shl_test(nat!(2, 1), 0, nat!(2, 1));
        shl_test(nat!(0x80000000), 1, nat!(0, 1));
        shl_test(nat!(2, 1), 32, nat!(0, 2, 1));
        shl_test(nat!(2, 1), 33, nat!(0, 4, 2));

        shl_test(
            nat!(1075296225, 677532807, 3880960401),
            39,
            nat!(0, 198963328, 824853408, 2841692308, 115),
        )
    }

    /// Auxiliary method for testing a multiplication.
    fn mul_test(x: Nat, y: Nat, z: Nat) {
        // Immutable shift.
        assert_eq!(x.mul(&y), z);
    }

    /// Performs various test multiplications.
    #[test]
    fn mul() {
        mul_test(nat!(4), nat!(5), nat!(20));
        mul_test(nat!(55), nat!(0), nat!(0));
        mul_test(nat!(1, 1), nat!(1, 2, 1), nat!(1, 3, 3, 1));

        mul_test(
            nat!(2264321224, 2023871691, 4021409222),
            nat!(2119383293, 3578280933, 2420753119),
            nat!(
                864821672, 1696420961, 2865231083, 16771142, 4044334621,
                2266568811
            ),
        );
    }

    /// Performs various test comparisons.
    #[test]
    fn cmp() {
        use std::cmp::Ordering::*;

        assert_eq!(nat!(15).cmp(&nat!(17)), Less);
        assert_eq!(nat!(0, 1).cmp(&nat!(u32::MAX)), Greater);
        assert_eq!(nat!(4, 4).cmp(&nat!(3, 4)), Greater);
        assert_eq!(nat!(12, 14).cmp(&nat!(12, 14)), Equal);
    }

    /// Auxiliary method for testing a subtraction.
    fn monus_test(mut x: Nat, y: Nat, z: Nat) {
        // Immutable shift.
        assert_eq!(x.monus(&y), z);

        // Mutable shift.
        x.monus_mut(&y);
        assert_eq!(x, z);
    }

    /// Performs various test subtractions.
    #[test]
    fn monus() {
        monus_test(nat!(3), nat!(2), nat!(1));
        monus_test(nat!(5, 0, 1), nat!(6), nat!(u32::MAX, u32::MAX));
        monus_test(nat!(14, 16), nat!(0, 22), nat!(0));
        monus_test(nat!(3, 1), nat!(3, 2), nat!(0));

        monus_test(
            nat!(1541468456, 839813775, 2949060051),
            nat!(2189725333, 2257444005, 2601135520),
            nat!(3646710419, 2877337065, 347924530),
        )
    }
}
