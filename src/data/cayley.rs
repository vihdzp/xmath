//pub trait Algebra<F:Field>:LinearModule<F>+Mul{}

use xmath_macro::Transparent;

use crate::traits::*;

/// A trait for a type resulting from a repeated
/// [Cayley–Dickson construction](https://en.wikipedia.org/wiki/Cayley_Dickson_construction).
pub trait CayleyTuple: ArrayLike {
    /// Takes the conjugate of a number.
    fn conj(&self) -> Self;

    /// Mutably takes the conjugate of a number.
    fn conj_mut(&mut self);

    /// Takes the norm of a number.
    fn norm(&self) -> Self::Item;
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

impl<T: Clone> CayleyTuple for Cayley1<T> {
    fn conj(&self) -> Self {
        self.clone()
    }

    fn conj_mut(&mut self) {}

    fn norm(&self) -> Self::Item {
        self.clone().to_single()
    }
}

/// A pair in the Cayley–Dickson construction.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CayleyPair<T>(pub T, pub T);

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
        Self::new(self.0.add(&rhs.0), self.1.add(&rhs.1))
    }

    fn add_mut(&mut self, rhs: &Self) {
        self.0.add_mut(&rhs.0);
        self.1.add_mut(&rhs.1);
    }

    fn add_rhs_mut(&self, rhs: &mut Self) {
        self.0.add_rhs_mut(&mut rhs.0);
        self.1.add_rhs_mut(&mut rhs.1);
    }
}

impl<T: Neg> Neg for CayleyPair<T> {
    fn neg(&self) -> Self {
        Self::new(self.0.neg(), self.1.neg())
    }

    fn neg_mut(&mut self) {
        self.0.neg_mut();
        self.1.neg_mut();
    }
}

impl<T: Sub> Sub for CayleyPair<T> {
    fn sub(&self, rhs: &Self) -> Self {
        Self::new(self.0.sub(&rhs.0), self.1.sub(&rhs.1))
    }

    fn sub_mut(&mut self, rhs: &Self) {
        self.0.sub_mut(&rhs.0);
        self.1.sub_mut(&rhs.1);
    }

    fn sub_rhs_mut(&self, rhs: &mut Self) {
        self.0.sub_rhs_mut(&mut rhs.0);
        self.1.sub_rhs_mut(&mut rhs.1);
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

impl<T: CayleyTuple> CayleyTuple for CayleyPair<T> {
    fn conj(&self) -> Self {
        todo!()
    }

    fn conj_mut(&mut self) {
        todo!()
    }

    fn norm(&self) -> Self::Item {
        todo!()
    }
}
