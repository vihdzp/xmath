use crate::traits::*;

/// A trait for a type resulting from a repeated
/// [Cayley–Dickson construction](https://en.wikipedia.org/wiki/Cayley_Dickson_construction).
pub trait CayleyTuple: ArrayLike {
    /// Takes the conjugate of a number.
    fn conj(&self) -> Self;

    /// Mutably takes the conjugate of a number.
    fn conj_mut(&mut self);

    /// Returns the squared norm of a number.
    fn sq_norm(&self) -> Self::Item;
}

/// A type endowed with a trivial implementation of [`CayleyTuple`].
#[repr(C)]
#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Transparent,
    ArrayFromIter,
)]
pub struct Cayley1<T>(pub T);

impl<T: Add> Add for Cayley1<T> {
    fn add(&self, rhs: &Self) -> Self {
        Self(self.0.add(&rhs.0))
    }

    fn add_mut(&mut self, rhs: &Self) {
        self.0.add_mut(&rhs.0)
    }

    fn add_rhs_mut(&self, rhs: &mut Self) {
        self.0.add_rhs_mut(&mut rhs.0)
    }
}

impl<T: Neg> Neg for Cayley1<T> {
    fn neg(&self) -> Self {
        Self(self.0.neg())
    }

    fn neg_mut(&mut self) {
        self.0.neg_mut();
    }
}

impl<T: Sub> Sub for Cayley1<T> {
    fn sub(&self, rhs: &Self) -> Self {
        Self(self.0.sub(&rhs.0))
    }

    fn sub_mut(&mut self, rhs: &Self) {
        self.0.sub_mut(&rhs.0)
    }

    fn sub_rhs_mut(&self, rhs: &mut Self) {
        self.0.sub_rhs_mut(&mut rhs.0)
    }
}

impl<T: Mul> Mul for Cayley1<T> {
    fn mul(&self, rhs: &Self) -> Self {
        Self(self.0.mul(&rhs.0))
    }

    fn mul_mut(&mut self, rhs: &Self) {
        self.0.mul_mut(&rhs.0)
    }

    fn mul_rhs_mut(&self, rhs: &mut Self) {
        self.0.mul_rhs_mut(&mut rhs.0)
    }
}

impl<T: Mul> CayleyTuple for Cayley1<T> {
    fn conj(&self) -> Self {
        self.clone()
    }

    fn conj_mut(&mut self) {}

    fn sq_norm(&self) -> Self::Item {
        self.sq().to_single()
    }
}

/// A pair in the Cayley–Dickson construction.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CayleyPair<T>(pub T, pub T);

/// The type of complex numbers, formed as pairs of `T` in the
/// [Cayley–Dickson construction](https://en.wikipedia.org/wiki/Cayley_Dickson_construction).
pub type Complex<T> = CayleyPair<Cayley1<T>>;

/// The type of quaternions, formed as 4-tuples of `T` in the
/// [Cayley–Dickson construction](https://en.wikipedia.org/wiki/Cayley_Dickson_construction).
pub type Quaternion<T> = CayleyPair<Complex<T>>;

/// The type of octonions, formed as 8-tuples of `T` in the
/// [Cayley–Dickson construction](https://en.wikipedia.org/wiki/Cayley_Dickson_construction).
pub type Octonion<T> = CayleyPair<Quaternion<T>>;

/// The type of sedenions, formed as 16-tuples of `T` in the
/// [Cayley–Dickson construction](https://en.wikipedia.org/wiki/Cayley_Dickson_construction).
pub type Sedenion<T> = CayleyPair<Octonion<T>>;

impl<T> CayleyPair<T> {
    /// Initializes a new pair.
    pub fn new(x: T, y: T) -> Self {
        Self(x, y)
    }

    /// Returns a reference to the first entry.
    pub fn fst(&self) -> &T {
        &self.0
    }

    /// Returns a reference to the second entry.
    pub fn snd(&self) -> &T {
        &self.1
    }

    /// Returns a mutable reference to the first entry.
    pub fn fst_mut(&mut self) -> &mut T {
        &mut self.0
    }

    /// Returns a mutable reference to the second entry.
    pub fn snd_mut(&mut self) -> &mut T {
        &mut self.1
    }
}

impl<T: CayleyTuple> SliceLike for CayleyPair<T> {
    type Item = T::Item;

    fn as_slice(&self) -> &[T::Item] {
        ArrayLike::as_ref(self)
    }

    fn as_mut_slice(&mut self) -> &mut [T::Item] {
        ArrayLike::as_mut(self)
    }
}

impl<T: CayleyTuple> ArrayLike for CayleyPair<T> {
    const LEN: usize = T::LEN * 2;

    fn from_iter_mut<I: Iterator<Item = Self::Item>>(iter: &mut I) -> Self {
        Self::new(T::from_iter_mut(iter), T::from_iter_mut(iter))
    }
}

impl<T: CayleyTuple> FromIterator<T::Item> for CayleyPair<T> {
    fn from_iter<I: IntoIterator<Item = T::Item>>(iter: I) -> Self {
        ArrayLike::from_iter(iter)
    }
}

impl<T: Add> Add for CayleyPair<T> {
    fn add(&self, rhs: &Self) -> Self {
        Self::new(self.fst().add(rhs.fst()), self.snd().add(rhs.snd()))
    }

    fn add_mut(&mut self, rhs: &Self) {
        self.fst_mut().add_mut(&rhs.fst());
        self.snd_mut().add_mut(&rhs.snd());
    }

    fn add_rhs_mut(&self, rhs: &mut Self) {
        self.fst().add_rhs_mut(rhs.fst_mut());
        self.snd().add_rhs_mut(rhs.snd_mut());
    }
}

impl<T: Neg> Neg for CayleyPair<T> {
    fn neg(&self) -> Self {
        Self::new(self.fst().neg(), self.snd().neg())
    }

    fn neg_mut(&mut self) {
        self.fst_mut().neg_mut();
        self.snd_mut().neg_mut();
    }
}

impl<T: Sub> Sub for CayleyPair<T> {
    fn sub(&self, rhs: &Self) -> Self {
        Self::new(self.fst().sub(rhs.fst()), self.snd().sub(rhs.snd()))
    }

    fn sub_mut(&mut self, rhs: &Self) {
        self.fst_mut().sub_mut(rhs.fst());
        self.snd_mut().sub_mut(rhs.snd());
    }

    fn sub_rhs_mut(&self, rhs: &mut Self) {
        self.fst().sub_rhs_mut(rhs.fst_mut());
        self.snd().sub_rhs_mut(rhs.snd_mut());
    }
}

impl<T: Add + Sub + Mul + CayleyTuple> Mul for CayleyPair<T> {
    fn mul(&self, rhs: &Self) -> Self {
        // (ac - d*b, da + bc*)
        Self::new(
            self.fst()
                .mul(rhs.fst())
                .sub(&rhs.snd().conj().mul(self.snd())),
            rhs.snd()
                .mul(self.fst())
                .add(&self.snd().mul(&rhs.fst().conj())),
        )
    }
}

impl<T: Neg + CayleyTuple> CayleyTuple for CayleyPair<T>
where
    T::Item: Add,
{
    fn conj(&self) -> Self {
        Self::new(self.fst().conj(), self.snd().neg())
    }

    fn conj_mut(&mut self) {
        self.fst_mut().conj_mut();
        self.snd_mut().neg_mut();
    }

    fn sq_norm(&self) -> Self::Item {
        self.fst().sq_norm().add(&self.snd().sq_norm())
    }
}

impl<T> Complex<T> {
    pub fn new2(x: T, y: T) -> Self {
        Self::new(Cayley1(x), Cayley1(y))
    }

    pub fn re(x: T) -> Self
    where
        T: Zero,
    {
        Self::new2(x, T::zero())
    }

    pub fn im(y: T) -> Self
    where
        T: Zero,
    {
        Self::new2(T::zero(), y)
    }

    pub fn i() -> Self
    where
        T: Zero + One,
    {
        Self::im(T::one())
    }
}

impl<T: Zero> From<T> for Complex<T> {
    fn from(value: T) -> Self {
        Self::re(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::Wrapping;

    #[test]
    fn complex() {
        assert_eq!(Complex::i().sq(), Complex::from(Wrapping(-1i8)))
    }
}
