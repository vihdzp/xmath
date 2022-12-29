//! Implements the type [`Array`] of statically-sized arrays with element-wise
//! operations.

use crate::{
    algs::{madd_gen, mmul_gen},
    traits::*, transpose::Transpose,
};
use xmath_macro::SliceIndex;
use xmath_traits::*;

/// A statically-sized array of elements of a single type.
///
/// ## Internal representation
///
/// This stores a single `[T; N]` field. The layout is guaranteed to be the
/// same, and the allowed values are equal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, SliceIndex, ArrayFromIter)]
#[repr(transparent)]
pub struct Array<T, const N: usize>(pub [T; N]);

impl<T, const N: usize> SliceLike for Array<T, N> {
    type Item = T;

    fn as_slice(&self) -> &[Self::Item] {
        &self.0
    }

    fn as_mut_slice(&mut self) -> &mut [Self::Item] {
        &mut self.0
    }
}

impl<T, const N: usize> ArrayLike for Array<T, N> {
    const LEN: usize = N;

    fn from_iter_mut<I: Iterator<Item = Self::Item>>(iter: &mut I) -> Self {
        Self::from_fn(|_| iter.next().unwrap())
    }
}

/// Iterates over the entries of the array.
impl<T, const N: usize> IntoIterator for Array<T, N> {
    type Item = T;
    type IntoIter = std::array::IntoIter<T, N>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<T, const N: usize> Array<T, N> {
    /// Initializes a new array from a `[T; N]`.
    pub fn new(x: [T; N]) -> Self {
        Self(x)
    }

    /// Returns an iterator over the array.
    pub fn iter(&self) -> std::slice::Iter<T> {
        self.as_ref().iter()
    }

    /// Returns a mutable iterator over the array.
    pub fn iter_mut(&mut self) -> std::slice::IterMut<T> {
        self.as_mut().iter_mut()
    }

    /// Returns a reference to the specified entry, or `None` if it doesn't
    /// exist.
    pub fn get(&self, index: usize) -> Option<&T> {
        self.as_ref().get(index)
    }

    /// Sets a value without doing bounds checking.
    ///
    /// ## Safety
    ///
    /// The index must be within bounds.
    pub unsafe fn set_unchecked(&mut self, index: usize, value: T) {
        *self.as_mut().get_unchecked_mut(index) = value;
    }

    /// Sets a value, doing bounds checking.
    ///
    /// ## Panics
    ///
    /// Panics if the index is out of bounds.
    pub fn set(&mut self, index: usize, value: T) {
        self.as_mut()[index] = value;
    }

    /// Initializes the array `[f(0), f(1), ..., f(N - 1)]`.
    pub fn from_fn<F: FnMut(usize) -> T>(f: F) -> Self {
        Self::new(std::array::from_fn(f))
    }

    pub fn from_iter_zero<I: IntoIterator<Item = T>>(iter: I) -> Self
    where
        T: Zero,
    {
        let mut iter = iter.into_iter();
        Self(std::array::from_fn(|_| iter.next().unwrap_or_else(T::zero)))
    }

    /// Performs a pairwise operation on two arrays.
    pub fn pairwise<F: FnMut(&T, &T) -> T>(&self, x: &Self, mut f: F) -> Self {
        Self::from_fn(|i| f(&self[i], &x[i]))
    }

    /// Performs a mutable pairwise operation on two arrays.
    pub fn pairwise_mut<F: FnMut(&mut T, &T)>(&mut self, x: &Self, mut f: F) {
        for i in 0..N {
            f(&mut self[i], &x[i]);
        }
    }
}

/// The default value is the array `[T::default(); N]`.
impl<T: Default, const N: usize> Default for Array<T, N> {
    fn default() -> Self {
        Self::from_fn(|_| T::default())
    }
}

/// The zero value is the array `[0; N]`.
impl<T: Zero, const N: usize> Zero for Array<T, N> {
    fn zero() -> Self {
        Self::from_fn(|_| T::zero())
    }

    fn is_zero(&self) -> bool {
        self.iter().all(T::is_zero)
    }
}

/// The one value is the array `[1; N]`.
impl<T: One, const N: usize> One for Array<T, N> {
    fn one() -> Self {
        Self::from_fn(|_| T::one())
    }

    fn is_one(&self) -> bool {
        self.iter().all(T::is_one)
    }
}

/// Element-wise negation.
impl<T: Neg, const N: usize> Neg for Array<T, N> {
    fn neg(&self) -> Self {
        self.map(T::neg)
    }

    fn neg_mut(&mut self) {
        self.map_mut(T::neg_mut);
    }
}

/// Element-wise inversion.
impl<T: Inv, const N: usize> Inv for Array<T, N> {
    fn inv(&self) -> Self {
        self.map(T::inv)
    }

    fn inv_mut(&mut self) {
        self.map_mut(T::inv_mut)
    }
}

/// Element-wise addition.
impl<T: Add, const N: usize> Add for Array<T, N> {
    fn add(&self, x: &Self) -> Self {
        self.pairwise(x, T::add)
    }

    fn add_mut(&mut self, x: &Self) {
        self.pairwise_mut(x, T::add_mut);
    }

    fn double(&self) -> Self {
        self.map(T::double)
    }

    fn double_mut(&mut self) {
        self.map_mut(T::double_mut);
    }
}

/// Element-wise multiplication.
impl<T: Mul, const N: usize> Mul for Array<T, N> {
    fn mul(&self, x: &Self) -> Self {
        self.pairwise(x, T::mul)
    }

    fn mul_mut(&mut self, x: &Self) {
        self.pairwise_mut(x, T::mul_mut);
    }

    fn sq(&self) -> Self {
        self.map(T::sq)
    }

    fn sq_mut(&mut self) {
        self.map_mut(T::sq_mut);
    }
}

impl<T: CommAdd, const N: usize> CommAdd for Array<T, N> {}
impl<T: CommMul, const N: usize> CommMul for Array<T, N> {}

/// Element-wise subtraction.
impl<T: Sub, const N: usize> Sub for Array<T, N> {
    fn sub(&self, x: &Self) -> Self {
        self.pairwise(x, T::sub)
    }

    fn sub_mut(&mut self, x: &Self) {
        self.pairwise_mut(x, T::sub_mut);
    }
}

/// Element-wise division.
impl<T: Div, const N: usize> Div for Array<T, N> {
    fn div(&self, x: &Self) -> Self {
        self.pairwise(x, T::div)
    }

    fn div_mut(&mut self, x: &Self) {
        self.pairwise_mut(x, T::div_mut);
    }
}

impl<T: AddMonoid, const N: usize> AddMonoid for Array<T, N> {}
impl<T: MulMonoid, const N: usize> MulMonoid for Array<T, N> {}
impl<T: AddGroup, const N: usize> AddGroup for Array<T, N> {}
impl<T: MulGroup, const N: usize> MulGroup for Array<T, N> {}
impl<T: Ring, const N: usize> Ring for Array<T, N> {}

/// Arrays are a one-dimensional list.
impl<T, const N: usize> List<U1> for Array<T, N> {
    type Item = T;
    const SIZE: Array1<Dim> = Array1(Dim::Fin(N));

    fn coeff_ref_gen(&self, index: &Array1<usize>) -> Option<&Self::Item> {
        self.get(index.0)
    }

    unsafe fn coeff_set_unchecked_gen(&mut self, index: &Array1<usize>, value: Self::Item) {
        *self.as_mut().get_unchecked_mut(index.0) = value;
    }

    fn map<F: Fn(&Self::Item) -> Self::Item>(&self, f: F) -> Self {
        Self(std::array::from_fn(|i| f(&self[i])))
    }

    fn map_mut<F: Fn(&mut Self::Item)>(&mut self, f: F) {
        for i in 0..N {
            f(&mut self[i]);
        }
    }
}

impl<T: Ring, const N: usize> Module<U1> for Array<T, N> {
    fn dot(&self, x: &Self) -> Self::Item {
        let mut res = Self::Item::zero();

        for i in 0..N {
            res.add_mut(&self[i].mul(&x[i]));
        }

        res
    }
}

impl<T, const N: usize> LinearModule for Array<T, N>
where
    T: Ring,
{
    fn support(&self) -> usize {
        N
    }
}

impl<C: TypeNum, V: List<C>, const N: usize> List<Succ<C>> for Array<V, N> {
    type Item = V::Item;
    const SIZE: ArrayPair<Dim, C::Array<Dim>> = ArrayPair(Dim::Fin(N), V::SIZE);

    fn coeff_ref_gen(
        &self,
        index: &ArrayPair<usize, <C as TypeNum>::Array<usize>>,
    ) -> Option<&Self::Item> {
        self[index.0].coeff_ref_gen(&index.1)
    }

    unsafe fn coeff_set_unchecked_gen(
        &mut self,
        index: &<Succ<C> as TypeNum>::Array<usize>,
        value: Self::Item,
    ) {
        self[index.0].coeff_set_unchecked_gen(&index.1, value);
    }

    fn map<F: Fn(&Self::Item) -> Self::Item>(&self, f: F) -> Self {
        Self::from_fn(|i| self[i].map(&f))
    }

    fn map_mut<F: Fn(&mut Self::Item)>(&mut self, f: F) {
        for x in self.iter_mut() {
            x.map_mut(&f);
        }
    }
}

impl<C: TypeNum, V: Module<C>, const N: usize> Module<Succ<C>> for Array<V, N>
where
    V::Item: Ring,
{
    fn dot(&self, x: &Self) -> Self::Item {
        let mut res = V::Item::zero();

        for i in 0..N {
            res.add_mut(&self[i].dot(&x[i]));
        }

        res
    }
}

impl<V: LinearModule, const N: usize> Matrix for Array<V, N>
where
    V::Item: Ring,
{
    const DIR: Direction = Direction::Row;

    fn col_support(&self, _: usize) -> usize {
        N
    }

    fn row_support(&self, index: usize) -> usize {
        self.get(index).map_or(0, V::support)
    }

    fn height(&self) -> usize {
        N
    }

    fn width(&self) -> usize {
        (0..N)
            .map(|i| self.row_support(i))
            .max()
            .unwrap_or_default()
    }

    fn collect_row<I: Iterator<Item = Self::Item>, J: Iterator<Item = I>>(mut iter: J) -> Self {
        let mut res = Self::zero();

        for i in 0..N {
            if let Some(iter) = iter.next() {
                res[i] = iter.collect();
            } else {
                break;
            }
        }

        res
    }

    fn collect_col<I: Iterator<Item = Self::Item>, J: Iterator<Item = I>>(iter: J) -> Self {
        let mut res = Self::zero();

        for (j, mut iter) in iter.enumerate() {
            for i in 0..N {
                if let Some(x) = iter.next() {
                    res[i].coeff_set(j, x);
                }
            }
        }

        res
    }
}

/// An alias for an M × N statically sized matrix.
pub type MatrixMN<T, const M: usize, const N: usize> = Array<Array<T, N>, M>;

/// Checks that we can transmute between two [`MatrixMN`].
macro_rules! transmute_check {
    ($m: expr, $n: expr, $k: expr, $l: expr) => {
        assert_eq!(
            $m * $n,
            $k * $l,
            "incompatible matrix sizes for transmutation (M: {}, N: {}, K: {}, L: {})",
            $m,
            $n,
            $k,
            $l
        );
    };
}

impl<T: Ring, const M: usize, const N: usize> MatrixMN<T, M, N> {
    /// Adds two statically sized matrices.
    pub fn madd(&self, m: &Self) -> Self {
        madd_gen(self, m)
    }

    /// Multiplies two statically sized matrices.
    pub fn mmul<const K: usize>(&self, m: &MatrixMN<T, N, K>) -> MatrixMN<T, M, K> {
        mmul_gen(self, m)
    }

    pub fn from_fn_matrix<F: Copy + FnMut(usize, usize) -> T>(f: F) -> Self {
        Matrix::from_fn(M, N, f)
    }

    /// Transmutes a matrix as another matrix of the same size. The size check
    /// is performed at compile time.
    ///
    /// ## Panics
    ///
    /// Panics if both matrices don't have the same size.
    pub fn transmute<const K: usize, const L: usize>(self) -> MatrixMN<T, K, L> {
        transmute_check!(M, N, K, L);

        // Safety: matrices of the same size have the same layout.
        unsafe { xmath_core::transmute_gen(self) }
    }

    /// Transmutes a reference to a matrix as a refrence to another matrix of
    /// the same size. The size check is performed at compile time.
    ///
    /// ## Panics
    ///
    /// Panics if both matrices don't have the same size.
    pub fn transmute_ref<const K: usize, const L: usize>(&self) -> &MatrixMN<T, K, L> {
        transmute_check!(M, N, K, L);

        // Safety: we've performed the size check.
        unsafe { xmath_core::transmute_ref(self) }
    }

    /// Transmutes a mutable reference to a matrix as a mutable refrence to
    /// another matrix of the same size. The size check is performed at
    /// compile time.
    ///
    /// ## Panics
    ///
    /// Panics if both matrices don't have the same size.
    pub fn transmute_mut<const K: usize, const L: usize>(&mut self) -> &mut MatrixMN<T, K, L> {
        transmute_check!(M, N, K, L);

        // Safety: we've performed the size check.
        unsafe { xmath_core::transmute_mut(self) }
    }
}

/// Interprets a vector as a row vector.
pub type RowVec<T> = Array<T, 1>;

impl<T> RowVec<T> {
    /// Creates a new row vector.
    pub fn new_row(x: T) -> Self {
        Self([x])
    }

    /// Gets the inner vector.
    pub fn inner(self) -> T {
        xmath_core::from_array(self.0)
    }
}

/// Interprets a vector as a column vector.
pub type ColVec<T> = Transpose<RowVec<T>>;

impl<T> ColVec<T> {
    /// Creates a new column vector.
    pub fn new_col(x: T) -> Self {
        Self(RowVec::new_row(x))
    }

    /// Gets the inner vector.
    pub fn inner(self) -> T {
        self.0.inner()
    }
}

/// A macro to simplify writing down a [`MatrixMN`].
///
/// See also [`matrix_dyn`](crate::matrix_dyn).
///
/// ## Example
///
/// ```
/// # use std::num::Wrapping;
/// # use xmath::data::MatrixMN;
/// # use xmath::matrix_mn;
/// # use xmath::traits::Wu8;///
/// let m: MatrixMN<Wu8, 2, 2> = matrix_mn!(
///     Wrapping(1), Wrapping(2);
///     Wrapping(3), Wrapping(4)
/// );
///
/// // Array([Array([1, 2]), Array([3, 4])])
/// println!("{:?}", m);
/// ```
#[macro_export]
macro_rules! matrix_mn {
    ($($($x: expr),*);*) => {
        xmath::data::Array([$(xmath::data::Array([$($x),*])),*])
    };
}
