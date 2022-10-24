use crate::traits::{basic::*, matrix::List};

use super::poly::Poly;

/// The most basic representation for a variable-sized natural number.
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
        vec![(x % u32::MAX as u64) as u32, (x >> 32) as u32].into()
    }
}

impl From<u128> for Nat {
    fn from(x: u128) -> Self {
        vec![
            (x % u32::MAX as u128) as u32,
            ((x >> 32) % u32::MAX as u128) as u32,
            ((x >> 64) % u32::MAX as u128) as u32,
            (x >> 96) as u32,
        ]
        .into()
    }
}

impl Nat {
    /// Creates a new `Nat` from a `Poly<u32>`.
    pub fn new(p: Poly<u32>) -> Self {
        p.into()
    }

    /// The base 2³² digit count for the number.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns whether the underlying slice is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Returns the `i`-th digit of the number.
    pub fn digit(&self, i: usize) -> u32 {
        self.0.coeff(i)
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

impl Nat {
    /// An auxiliary method that adds two `u32`s and a carry together. Returns
    /// the wrapped result and whether there was a carry.
    fn add_digits(x: u32, y: u32, carry: bool) -> (u32, bool) {
        let (d, b) = x.overflowing_add(y);

        if carry {
            // This is the only case in which adding `1` to `d` would overflow.
            if d == u32::MAX {
                (0, true)
            } else {
                (d + 1, b)
            }
        } else {
            (d, b)
        }
    }

    /// An auxiliary method. If the carry flag is set, adds one to `s` and
    /// writes the output in `v`. Otherwise just copies `s` to `v`.
    fn carry_push(s: &[u32], v: &mut Vec<u32>, mut carry: bool) {
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
}

impl Add for Nat {
    fn add(&self, z: &Self) -> Self {
        // The shorter and longer number.
        let (x, y) = sort_by(self, z, Nat::len);

        let mut res = Vec::with_capacity(y.len() + 1);
        let mut carry = false;

        // Adds digits one by one, until the first number runs out.
        for i in 0..x.len() {
            let (d, b) = Nat::add_digits(x.digit(i), y.digit(i), carry);
            res.push(d);
            carry = b;
        }

        // Continues until the second number runs out.
        Nat::carry_push(&y.0.as_slice()[x.len()..], &mut res, carry);

        // Safety: `carry_push` can't end in a `0`.
        unsafe { Poly::new_unchecked(res).into() }
    }

    fn add_mut(&mut self, x: &Self) {
        unsafe {
            let mut carry = false;

            // Modify digits one by one, until our space runs out.
            for i in 0..self.len().min(x.len()) {
                let (d, b) = Nat::add_digits(self.digit(i), x.digit(i), carry);
                self.0.as_slice_mut()[i] = d;
                carry = b;
            }

            // If the first number is at most as long as the second.
            if self.len() <= x.len() {
                // Then keep pushing the next digits.
                Nat::carry_push(
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

    /* fn double_mut(&mut self) {} */
}

impl AddMonoid for Nat {}

#[cfg(test)]
mod tests {
    use super::*;

    fn add_test(mut x: Nat, y: Nat, z: Nat) {
        // Immutable addition.
        assert_eq!(x.add(&y), z);

        // Mutable addition.
        x.add_mut(&y);
        assert_eq!(x, z);
    }

    #[test]
    fn add() {
        add_test(Nat::from(1u32), Nat::from(2u32), Nat::from(3u32));
        add_test(Nat::from(u32::MAX), Nat::one(), Poly::x().into());
        add_test(
            Nat::from(vec![u32::MAX; 2]),
            Nat::one(),
            Nat::from(vec![0, 0, 1]),
        );
        add_test(
            Nat::from(vec![u32::MAX; 2]),
            Nat::from(vec![u32::MAX; 2]),
            Nat::from(vec![u32::MAX - 1, u32::MAX, 1]),
        )
    }
}
