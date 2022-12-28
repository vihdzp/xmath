pub mod matrix;

use crate::{data::aliases::Additive, traits::*};

/// Multiplication by doubling.
///
/// Its multiplicative analog is [`upow`].
pub fn umul<T: AddMonoid + Clone>(mut x: T, mut e: u32) -> T {
    let mut y = T::zero();

    if e == 0 {
        return y;
    }

    while e != 1 {
        if e % 2 == 1 {
            y.add_mut(&x);
        }

        x.double_mut();
        e /= 2;
    }

    y.add_mut(&x);
    y
}

/// Exponentiation by squaring.
///
/// Its additive analog is [`umul`].
pub fn upow<T: MulMonoid + Clone>(x: T, e: u32) -> T {
    umul(Additive(x), e).0
}
