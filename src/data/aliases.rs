//! Declares various type aliases, which change the structure a type is endowed
//! with.

use crate::traits::basic::*;
use crate::traits::dim::{C2, U2};
use crate::traits::matrix::*;

/// A type alias that endows a type with additive operations instead of
/// multiplicative ones.
///
/// This allows us to implement algorithms for addition, and have them
/// immediately in multiplicative contexts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
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

/// A type alias which guarantees that the enclosed value is nonzero.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct NonZero<T>(T);

/// Returns a reference to the underlying value.
impl<T> AsRef<T> for NonZero<T> {
    fn as_ref(&self) -> &T {
        &self.0
    }
}

impl<T: Zero> NonZero<T> {
    /// Returns a mutable reference to the inner value.
    ///
    /// ## Safety
    ///
    /// The caller must ensure this value remains nonzero.
    pub unsafe fn as_mut(&mut self) -> &mut T {
        &mut self.0
    }

    /// Gets a `NonZero<T>` reference from a nonzero `T` reference.
    ///
    /// ## Safety
    ///
    /// The caller must ensure `x` is indeed nonzero.
    pub unsafe fn to_ref(x: &T) -> &Self {
        &*(x as *const T).cast()
    }

    /// Gets a mutable `NonZero<T>` reference from a nonzero `T` reference.
    ///
    /// ## Safety
    ///
    /// The caller must ensure `x` is indeed nonzero.
    pub unsafe fn to_mut(x: &mut T) -> &mut Self {
        &mut *(x as *mut T).cast()
    }

    /// Initializes a new `NonZero` from a given nonzero value.
    ///
    /// ## Safety
    ///
    /// The caller must ensure `x` is indeed nonzero.
    pub const unsafe fn new_unchecked(x: T) -> Self {
        Self(x)
    }

    /// Initializes a new `NonZero` from a given value. Returns `None` if the
    /// input is zero.
    pub fn new(x: T) -> Option<NonZero<T>> {
        if x.is_zero() {
            None
        } else {
            // Safety: we just checked that `x` is nonzero.
            unsafe { Some(Self::new_unchecked(x)) }
        }
    }
}

impl<T: ZeroNeOne> One for NonZero<T> {
    fn one() -> Self {
        // Safety: we assume `0 != 1`.
        unsafe { Self::new_unchecked(T::one()) }
    }
}

impl<T: IntegralDomain> Mul for NonZero<T> {
    fn mul(&self, x: &Self) -> Self {
        // Safety: we assume the product of nonzero elements is nonzero.
        unsafe { Self::new_unchecked(self.as_ref().mul(x.as_ref())) }
    }

    fn mul_mut(&mut self, x: &Self) {
        // Safety: we assume the product of nonzero elements is nonzero.
        unsafe { self.as_mut().mul_mut(x.as_ref()) };
    }

    fn sq(&self) -> Self {
        // Safety: we assume the product of nonzero elements is nonzero.
        unsafe { Self::new_unchecked(self.as_ref().sq()) }
    }

    fn sq_mut(&mut self) {
        // Safety: we assume the product of nonzero elements is nonzero.
        unsafe { self.as_mut().sq_mut() };
    }
}

impl<T: IntegralDomain + MulMonoid + ZeroNeOne> MulMonoid for NonZero<T> {}

/// A wrapper for a matrix that is to be interpreted with the rows and columns
/// swapped.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(transparent)]
pub struct Transpose<T>(pub T);

/// Returns a reference to the underlying matrix.
impl<T> AsRef<T> for Transpose<T> {
    fn as_ref(&self) -> &T {
        &self.0
    }
}

/// Returns a mutable reference to the underlying matrix.
impl<T> AsMut<T> for Transpose<T> {
    fn as_mut(&mut self) -> &mut T {
        &mut self.0
    }
}

impl<T> Transpose<T> {
    /// Gets a `Transpose<T>` reference from a `T` reference.
    pub fn to_ref(x: &T) -> &Self {
        // Safety: the type is `repr(transparent)`.
        unsafe { &*(x as *const T).cast() }
    }

    /// Gets a mutable `Transpose<T>` reference from a mutable `T` reference.
    pub fn to_mut(x: &mut T) -> &mut Self {
        // Safety: the type is `repr(transparent)`.
        unsafe { &mut *(x as *mut T).cast() }
    }
}

impl<M: List<U2>> List<U2> for Transpose<M> {
    type Item = M::Item;
    const SIZE: C2<Dim> = M::SIZE.swap();

    fn coeff_ref_gen(&self, index: &C2<usize>) -> Option<&Self::Item> {
        self.0.coeff_ref_gen(&index.swap())
    }

    unsafe fn coeff_set_unchecked_gen(
        &mut self,
        index: &C2<usize>,
        value: Self::Item,
    ) {
        self.0.coeff_set_unchecked_gen(&index.swap(), value)
    }

    fn map<F: Fn(&Self::Item) -> Self::Item>(&self, f: F) -> Self {
        Self(self.0.map(f))
    }

    fn map_mut<F: Fn(&mut Self::Item)>(&mut self, f: F) {
        self.0.map_mut(f);
    }
}

/// The zero value is just the transpose of 0.
impl<T: Zero> Zero for Transpose<T> {
    fn zero() -> Self {
        Self(T::zero())
    }

    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

/// The one value is just the transpose of 1.
impl<T: One> One for Transpose<T> {
    fn one() -> Self {
        Self(T::one())
    }

    fn is_one(&self) -> bool {
        self.0.is_one()
    }
}

/// Negates the inner value.
impl<T: Neg> Neg for Transpose<T> {
    fn neg(&self) -> Self {
        Self(self.0.neg())
    }
}

/// Adds the inner values.
impl<T: Add> Add for Transpose<T> {
    fn add(&self, x: &Self) -> Self {
        Self(self.0.add(&x.0))
    }

    fn add_mut(&mut self, x: &Self) {
        self.0.add_mut(&x.0);
    }

    fn double(&self) -> Self {
        Self(self.0.double())
    }

    fn double_mut(&mut self) {
        self.0.double_mut();
    }
}

/// Subtracts the inner values.
impl<T: Sub> Sub for Transpose<T> {
    fn sub(&self, x: &Self) -> Self {
        Self(self.0.sub(&x.0))
    }

    fn sub_mut(&mut self, x: &Self) {
        self.0.sub_mut(&x.0);
    }
}

impl<T: AddMonoid> AddMonoid for Transpose<T> {}
impl<T: AddGroup> AddGroup for Transpose<T> {}

impl<M: Module<U2>> Module<U2> for Transpose<M>
where
    Self::Item: Ring,
{
    fn smul(&self, x: &M::Item) -> Self {
        Self(self.0.smul(x))
    }

    fn smul_mut(&mut self, x: &M::Item) {
        self.0.smul_mut(x);
    }

    fn dot(&self, x: &Self) -> Self::Item {
        self.0.dot(&x.0)
    }
}

impl<M: Matrix> Matrix for Transpose<M>
where
    M::Item: Ring,
{
    const DIR: Direction = M::DIR.transpose();

    fn col_support(&self, j: usize) -> usize {
        self.0.row_support(j)
    }

    fn row_support(&self, i: usize) -> usize {
        self.0.col_support(i)
    }

    fn height(&self) -> usize {
        self.0.width()
    }

    fn width(&self) -> usize {
        self.0.height()
    }

    fn collect_row<I: Iterator<Item = M::Item>, J: Iterator<Item = I>>(
        iter: J,
    ) -> Self {
        Self(M::collect_col(iter))
    }

    fn collect_col<I: Iterator<Item = M::Item>, J: Iterator<Item = I>>(
        iter: J,
    ) -> Self {
        Self(M::collect_row(iter))
    }
}
