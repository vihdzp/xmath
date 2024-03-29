//! Implements the type of univariate polynomials [`Poly`] and the type of
//! bivariate polynomials [`Poly2`].

use crate::algs::{madd_gen, mmul_gen};
use crate::traits::*;
use std::cmp::Ordering::*;
use std::fmt::Write;
use std::ops::Index;
use std::slice::SliceIndex;
use xmath_traits::*;
use xmath_traits::{array, array_type};

/// A polynomial whose entries belong to a type.
///
/// ## Internal representation.
///
/// This type has a single `Vec<T>` field. It represents the coefficients in
/// order of increasing degree. The **only restriction** on this vector is that
/// its last entry not be a `0`. This ensures every `Poly<T>` has a unique
/// representation.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Poly<T: Zero>(Vec<T>);

/// Removes leading zeros from a vector.
///
/// Calling this function will make the vector safe to use as the backing vector
/// of a polynomial.
pub fn trim<T: Zero>(v: &mut Vec<T>) {
    while let Some(x) = v.last() {
        if x.is_zero() {
            v.pop();
        } else {
            break;
        }
    }
}

/// Returns whether the slice is valid for a polynomial.
pub fn is_valid<T: Zero>(slice: &[T]) -> bool {
    slice.last().map(T::is_zero) != Some(true)
}

/// When debug assertions are enabled, asserts that the slice is valid for a
/// polynomial.
macro_rules! debug_verify {
    ($slice: expr) => {
        debug_assert!(
            is_valid($slice),
            "the type invariant for polynomials has been broken"
        )
    };
}

/// Returns a reference to the inner slice.
impl<T: Zero> AsRef<[T]> for Poly<T> {
    fn as_ref(&self) -> &[T] {
        &self.0
    }
}

impl<T: Zero> Poly<T> {
    /// Returns a reference to the inner slice.
    pub fn as_slice(&self) -> &[T] {
        self.as_ref()
    }

    /// Returns a mutable reference to the inner slice.
    ///
    /// ## Safety
    ///
    /// The last entry must remain nonzero after modifying the inner slice.
    pub unsafe fn as_slice_mut(&mut self) -> &mut [T] {
        &mut self.0
    }

    /// Returns a reference to the inner vector.
    pub fn as_vec(&self) -> &Vec<T> {
        &self.0
    }

    /// Returns a mutable reference to the inner vector.
    ///
    /// ## Safety
    ///
    /// The last entry must remain nonzero after modifying the inner vector.
    pub unsafe fn as_vec_mut(&mut self) -> &mut Vec<T> {
        &mut self.0
    }

    /// Calls a function on the inner vector. Does not trim any leading zeros.
    ///
    /// This is useful when complex logic must be executed, as it guarantees
    /// that no other methods from `Poly` will be called while the vector is in
    /// an invalid (intermediate) state.
    ///
    /// When ran with debug assertions, this will perform the zero check
    /// regardless.
    ///
    /// ## Safety
    ///
    /// The caller must guarantee that the inner vector will not end up with a
    /// leading zero when the function finishes executing.
    pub unsafe fn and_trim_unchecked<U, F: FnOnce(&mut Vec<T>) -> U>(&mut self, f: F) -> U {
        let res = f(self.as_vec_mut());
        debug_verify!(self.as_slice());
        res
    }

    /// Calls a function on the inner vector, then trims it. This is a safe way
    /// to modify the inner vector.
    pub fn and_trim<U, F: FnOnce(&mut Vec<T>) -> U>(&mut self, f: F) -> U {
        // Safety: we trim the vector.
        unsafe {
            self.and_trim_unchecked(|vec| {
                let res = f(vec);
                trim(vec);
                res
            })
        }
    }

    /// Returns an iterator over the inner slice.
    pub fn iter(&self) -> std::slice::Iter<T> {
        self.as_ref().iter()
    }

    /// Creates a polynomial with the specified backing vector.
    ///
    /// ## Safety
    ///
    /// The vector must have a non-zero last entry.
    pub unsafe fn new_unchecked(v: Vec<T>) -> Self {
        debug_verify!(v.as_slice());
        Self(v)
    }

    /// Creates a polynomial with the specified backing vector. Automatically
    /// trims any leading zeros.
    pub fn new(mut v: Vec<T>) -> Self {
        trim(&mut v);

        // Safety: we just trimmed the vector.
        unsafe { Self::new_unchecked(v) }
    }

    /// The degree of the polynomial plus one. Returns `0` for the zero
    /// polynomial.
    pub fn len(&self) -> usize {
        self.as_slice().len()
    }

    /// Returns the capacity of the inner vector.
    pub fn capacity(&self) -> usize {
        self.as_vec().capacity()
    }

    /// Gets a reference to the entry in a given index.
    pub fn get(&self, index: usize) -> Option<&T> {
        self.as_slice().get(index)
    }

    pub fn modify<U, F: FnOnce(&mut T) -> U>(&mut self, index: usize, f: F) -> U {
        match self.len().cmp(&(index + 1)) {
            // Safety: the leading coefficient isn't modified.
            Greater => unsafe { f(&mut self.as_slice_mut()[index]) },

            Equal => self.and_trim(|vec| f(&mut vec[index])),

            Less => {
                let mut value = T::zero();
                let res = f(&mut value);

                if !value.is_zero() {
                    // Safety: the leading coefficient is not zero.
                    unsafe {
                        self.and_trim_unchecked(|vec| {
                            vec.resize_with(index, T::zero);
                            vec.push(value);
                        });
                    }
                }

                res
            }
        }
    }

    /// Sets a given index with a given value.
    pub fn set(&mut self, index: usize, value: T) {
        self.modify(index, |x| *x = value);
    }

    /// Pushes a value at the end of the underlying vector. A no-op in the case
    /// of zero.
    pub fn push(&mut self, value: T) {
        if !value.is_zero() {
            // Safety: we're not pushing 0.
            unsafe {
                self.as_vec_mut().push(value);
            }
        }
    }

    /// Return and remove the leading coefficient. We return `0` in the case of
    /// the zero polynomial.
    pub fn pop(&mut self) -> T {
        self.and_trim(|vec| vec.pop()).unwrap_or_else(T::zero)
    }

    /// Clears the underyling vector.
    pub fn clear(&mut self) {
        self.0.clear();
    }

    /// Returns a reference to the constant coefficient.
    pub fn first(&self) -> Option<&T> {
        self.0.first()
    }

    /// Returns a reference to the leading coefficient.
    pub fn last(&self) -> Option<&T> {
        self.0.last()
    }

    /// The degree of the polynomial. Returns `None` for the zero polynomial.
    pub fn degree(&self) -> Option<usize> {
        match self.len() {
            0 => None,
            x => Some(x - 1),
        }
    }

    /// Returns whether the underlying slice is empty.
    pub fn is_empty(&self) -> bool {
        self.as_slice().is_empty()
    }

    pub fn pairwise<U: Zero, V: Zero, F: FnMut(&T, &U) -> V>(
        &self,
        rhs: &Poly<U>,
        mut f: F,
    ) -> Poly<V> {
        let mut res = Vec::new();

        if self.len() < rhs.len() {
            for i in 0..self.len() {
                res.push(f(&self.as_ref()[i], &rhs.as_ref()[i]));
            }

            let z = T::zero();

            for i in self.len()..rhs.len() {
                res.push(f(&z, &rhs.as_ref()[i]));
            }
        } else {
            for i in 0..rhs.len() {
                res.push(f(&self.as_ref()[i], &rhs.as_ref()[i]));
            }

            let z = U::zero();

            for i in rhs.len()..self.len() {
                res.push(f(&self.as_ref()[i], &z));
            }
        }

        res.into()
    }

    pub fn pairwise_mut<U: Zero, F: FnMut(&mut T, &U)>(&mut self, rhs: &Poly<U>, mut f: F) {
        self.and_trim(|vec| {
            if vec.len() < rhs.len() {
                vec.resize_with(rhs.len(), T::zero);

                for i in 0..rhs.len() {
                    f(&mut vec[i], &rhs.as_ref()[i]);
                }
            } else {
                for i in 0..rhs.len() {
                    f(&mut vec[i], &rhs.as_ref()[i]);
                }

                let z = U::zero();

                for i in rhs.len()..vec.len() {
                    f(&mut vec[i], &z);
                }
            }
        });
    }

    /// Formats a polynomial, with a specified variable name. Defaults to `x`.
    pub fn fmt_with(&self, f: &mut std::fmt::Formatter<'_>, c: char) -> std::fmt::Result
    where
        T: std::fmt::Display,
    {
        // Are we writing down the `0` polynomial?
        let mut zero = true;
        f.write_char('(')?;

        // Write nonzero coefficients one by one.
        for (n, x) in self.as_slice().iter().enumerate() {
            if !x.is_zero() {
                if !zero {
                    write!(f, " + ")?;
                }

                zero = false;
                write!(f, "{x} {c}^{n}")?;
            }
        }

        // We write the zero polynomial as "(0)".
        if zero {
            write!(f, "0)")
        } else {
            f.write_char(')')
        }
    }
}

impl<T: Zero> Extend<T> for Poly<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        self.and_trim(|vec| vec.extend(iter));
    }
}

impl<T: Zero> From<Vec<T>> for Poly<T> {
    fn from(v: Vec<T>) -> Self {
        Self::new(v)
    }
}

impl<T: Zero> From<Poly<T>> for Vec<T> {
    fn from(p: Poly<T>) -> Self {
        p.0
    }
}

/// Iterates over the coefficients of the polynomial.
impl<T: Zero> IntoIterator for Poly<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a, T: Zero> IntoIterator for &'a Poly<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

/// Calls [`Poly::fmt_with`] with the default variable name `x`.
impl<T: Zero + std::fmt::Display> std::fmt::Display for Poly<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.fmt_with(f, 'x')
    }
}

impl<T: Zero, I> Index<I> for Poly<T>
where
    I: SliceIndex<[T]>,
{
    type Output = I::Output;

    fn index(&self, index: I) -> &Self::Output {
        &self.as_slice()[index]
    }
}

impl<T: Zero> Default for Poly<T> {
    fn default() -> Self {
        // Safety: this is a valid vector.
        unsafe { Self::new_unchecked(Vec::new()) }
    }
}

impl<T: Zero + PartialEq> Zero for Poly<T> {
    fn zero() -> Self {
        Self::default()
    }

    fn is_zero(&self) -> bool {
        self.is_empty()
    }
}

impl<T: Zero> Poly<T> {
    /// Creates the polynomial c * xⁿ, for nonzero `c`.
    ///
    /// ## Safety
    ///
    /// The caller must verify `c` is indeed nonzero.
    pub unsafe fn cxn_unchecked(c: T, n: usize) -> Self {
        debug_assert!(!c.is_zero(), "c equals zero");
        let mut res = Vec::new();
        res.resize_with(n, T::zero);
        res.push(c);
        Self::new_unchecked(res)
    }

    /// Creates the polynomial c * xⁿ.
    pub fn cxn(c: T, n: usize) -> Self {
        if c.is_zero() {
            Self::zero()
        } else {
            // Safety: we just verified `c` is nonzero.
            unsafe { Self::cxn_unchecked(c, n) }
        }
    }

    /// Creates a constant polynomial from a nonzero value.
    ///
    /// ## Safety
    ///
    /// The caller must verify `c` is indeed nonzero.
    pub unsafe fn c_unchecked(c: T) -> Self {
        Self::cxn_unchecked(c, 0)
    }

    /// Creates a constant polynomial.
    pub fn c(c: T) -> Self {
        Self::cxn(c, 0)
    }

    /// Creates the polynomial xⁿ.
    pub fn xn(n: usize) -> Self
    where
        T: ZeroNeOne,
    {
        // Safety: we assume `0 != 1`.
        unsafe { Self::cxn_unchecked(T::one(), n) }
    }

    /// Creates the `x` polynomial.
    pub fn x() -> Self
    where
        T: ZeroNeOne,
    {
        Self::xn(1)
    }

    /// Evaluates the polynomial at a point.
    pub fn eval(&self, x: &T) -> T
    where
        T: Add + Mul + One,
    {
        let mut res = T::zero();
        let mut xn = T::one();

        for c in self.as_slice() {
            res.add_mut(&c.mul(&xn));
            xn.mul_mut(x);
        }

        res
    }
}

impl<T: ZeroNeOne> One for Poly<T> {
    fn one() -> Self {
        Self::xn(0)
    }
}

impl<T: ZeroNeOne> ZeroNeOne for Poly<T> {}

impl<T: AddGroup> Neg for Poly<T> {
    fn neg(&self) -> Self {
        let res: Vec<T> = self.as_slice().iter().map(Neg::neg).collect();

        // Safety: nonzero elements of groups have nonzero negatives.
        unsafe { Self::new_unchecked(res) }
    }
}

impl<T: Zero + Clone> Poly<T> {
    /// An auxiliary method that generalizes addition and subtraction. The
    /// function `f` is the one to be performed coordinate by coordinate, while
    /// `g` must match `|x| f(0, x)`.
    ///
    /// ## Safety
    ///
    /// For this to give a valid polynomial we must have `g(x) == 0` only when
    /// `x == 0`.
    unsafe fn add_sub<F: Fn(&T, &T) -> T, G: Fn(&T) -> T>(&self, x: &Self, f: F, g: G) -> Self {
        match self.len().cmp(&x.len()) {
            Less => {
                let mut res = Vec::with_capacity(x.len());

                for i in 0..self.len() {
                    res.push(f(&self[i], &x[i]));
                }
                for i in self.len()..x.len() {
                    res.push(g(&x[i]));
                }

                Self::new_unchecked(res)
            }
            Equal => {
                let mut res = Vec::with_capacity(x.len());

                for i in 0..self.len() {
                    res.push(f(&self[i], &x[i]));
                }

                res.into()
            }
            Greater => {
                let mut res = Vec::with_capacity(self.len());

                for i in 0..x.len() {
                    res.push(f(&self[i], &x[i]));
                }
                for i in x.len()..self.len() {
                    res.push(self[i].clone());
                }

                Self::new_unchecked(res)
            }
        }
    }

    /// An auxiliary method that generalizes addition and subtraction. The
    /// function `f` is the one to be performed coordinate by coordinate, while
    /// `g` must match `|x| f(0, x)`.
    ///
    /// ## Safety
    ///
    /// For this to give a valid polynomial, we must have `g(x) == 0` only when
    /// `x == 0`.
    unsafe fn add_sub_mut<F: Fn(&mut T, &T), G: Fn(&T) -> T>(&mut self, other: &Self, f: F, g: G) {
        match self.len().cmp(&other.len()) {
            // Safety: the leading coefficient of the result is g of the leading
            // coefficient of `other`, which we assume nonzero.
            Less => {
                for i in 0..self.len() {
                    f(&mut self.as_vec_mut()[i], &other[i]);
                }
                for i in self.len()..other.len() {
                    self.as_vec_mut().push(g(&other[i]));
                }
            }

            Equal => {
                self.and_trim(|vec| {
                    for i in 0..vec.len() {
                        f(&mut vec[i], &other[i]);
                    }
                });
            }

            // Safety: the leading coefficient is not modified.
            Greater => {
                for i in 0..other.len() {
                    f(&mut self.as_vec_mut()[i], &other[i]);
                }
            }
        }
    }
}

impl<T: AddMonoid> Add for Poly<T> {
    fn add(&self, rhs: &Self) -> Self {
        // Safety: trivial.
        unsafe { self.add_sub(rhs, T::add, Clone::clone) }
    }

    fn add_mut(&mut self, rhs: &Self) {
        // Safety: trivial.
        unsafe { self.add_sub_mut(rhs, T::add_mut, Clone::clone) }
    }

    fn double(&self) -> Self {
        self.as_slice()
            .iter()
            .map(Add::double)
            .collect::<Vec<_>>()
            .into()
    }

    fn double_mut(&mut self) {
        // `x != 0` does not imply `x + x != 0`.
        self.and_trim(|vec| {
            for x in vec {
                x.double_mut();
            }
        });
    }
}

impl<T: AddMonoid> AddMonoid for Poly<T> {}
impl<T: AddMonoid + CommAdd> CommAdd for Poly<T> {}

impl<T: AddGroup> Sub for Poly<T> {
    fn sub(&self, x: &Self) -> Self {
        // Safety: in groups, nonzero elements have nonzero negatives.
        unsafe { self.add_sub(x, T::sub, T::neg) }
    }

    fn sub_mut(&mut self, x: &Self) {
        // Safety: in groups, nonzero elements have nonzero negatives.
        unsafe { self.add_sub_mut(x, T::sub_mut, T::neg) }
    }
}

impl<T: AddGroup> AddGroup for Poly<T> {}

impl<T: Zero + Add + Mul> Mul for Poly<T> {
    fn mul(&self, x: &Self) -> Self {
        let mut res = Vec::new();

        // Generally one longer than needed.
        res.resize_with(self.len() + x.len(), T::zero);

        for i in 0..self.len() {
            for j in 0..x.len() {
                res[i + j].add_mut(&self[i].mul(&x[j]));
            }
        }

        // We can omit trimming only in an integral domain.
        res.into()
    }
}

impl<T: Zero + CommAdd + CommMul> CommMul for Poly<T> {}
impl<T: Ring + ZeroNeOne> MulMonoid for Poly<T> {}
impl<T: Ring + ZeroNeOne> Ring for Poly<T> {}

impl<T: Zero> List<U1> for Poly<T> {
    const SIZE: array_type!(Dim; 1) = array!(Dim::Inf);
    type Item = T;

    fn coeff_ref_gen(&self, i: &Array1<usize>) -> Option<&Self::Item> {
        self.as_slice().get(i.0)
    }

    unsafe fn coeff_set_unchecked_gen(&mut self, index: &Array1<usize>, value: Self::Item) {
        self.set(index.0, value);
    }

    fn map<F: Fn(&Self::Item) -> Self::Item>(&self, f: F) -> Self {
        self.iter().map(f).collect()
    }

    fn map_mut<F: Fn(&mut Self::Item)>(&mut self, f: F) {
        self.and_trim(|vec| {
            for x in vec {
                f(x);
            }
        });
    }
}

impl<T: Ring> Module<U1> for Poly<T> {
    fn smul(&self, x: &T) -> Self {
        if x.is_zero() {
            Self::zero()
        } else {
            self.map(|y| y.mul(x))
        }
    }

    fn smul_mut(&mut self, x: &T) {
        if x.is_zero() {
            self.set_zero();
        } else {
            self.map_mut(|y| y.mul_mut(x));
        }
    }

    fn dot(&self, x: &Self) -> Self::Item {
        let mut res = Self::Item::zero();

        for i in 0..self.len().min(x.len()) {
            res.add_mut(&self[i].mul(&x[i]));
        }

        res
    }
}

impl<T: Zero> FromIterator<T> for Poly<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        iter.into_iter().collect::<Vec<_>>().into()
    }
}

impl<T: Ring> LinearModule for Poly<T> {
    fn support(&self) -> usize {
        self.len()
    }
}

impl<C: TypeNum, V: List<C> + Zero> List<Succ<C>> for Poly<V> {
    type Item = V::Item;
    const SIZE: ArrayPair<Dim, C::Array<Dim>> = ArrayPair(Dim::Inf, V::SIZE);

    fn coeff_ref_gen(&self, index: &ArrayPair<usize, C::Array<usize>>) -> Option<&Self::Item> {
        self.get(index.0)?.coeff_ref_gen(&index.1)
    }

    unsafe fn coeff_set_unchecked_gen(
        &mut self,
        index: &ArrayPair<usize, C::Array<usize>>,
        value: Self::Item,
    ) {
        self.modify(index.0, |inner| {
            inner.coeff_set_unchecked_gen(&index.1, value)
        });
    }

    fn map<F: Fn(&Self::Item) -> Self::Item>(&self, f: F) -> Self {
        self.iter().map(|x| x.map(&f)).collect()
    }

    fn map_mut<F: Fn(&mut Self::Item)>(&mut self, f: F) {
        self.and_trim(|vec| {
            for x in vec.iter_mut() {
                x.map_mut(&f);
            }
        });
    }
}

impl<C: TypeNum, V: Module<C>> Module<Succ<C>> for Poly<V>
where
    V::Item: Ring,
{
    fn dot(&self, x: &Self) -> Self::Item {
        let mut res = V::Item::zero();

        for (i, j) in self.iter().zip(x.iter()) {
            res.add_mut(&i.dot(j));
        }

        res
    }
}

impl<V: LinearModule> Matrix for Poly<V>
where
    V::Item: Ring,
{
    const DIR: Direction = Direction::Row;

    fn col_support(&self, _: usize) -> usize {
        self.len()
    }

    fn row_support(&self, index: usize) -> usize {
        self.get(index).map_or(0, V::support)
    }

    fn height(&self) -> usize {
        self.len()
    }

    fn width(&self) -> usize {
        (0..self.len())
            .map(|i| self.row_support(i))
            .max()
            .unwrap_or_default()
    }

    fn collect_row<I: Iterator<Item = Self::Item>, J: Iterator<Item = I>>(iter: J) -> Self {
        let mut res = Vec::new();

        for iter in iter {
            res.push(iter.collect());
        }

        res.into()
    }

    fn collect_col<I: Iterator<Item = Self::Item>, J: Iterator<Item = I>>(iter: J) -> Self {
        let mut res = Self::zero();

        for (col, iter) in iter.enumerate() {
            for (row, x) in iter.enumerate() {
                res.coeff_set(row, col, x);
            }
        }

        res
    }
}

/// An alias for a dynamically sized matrix.
pub type MatrixDyn<T> = Poly<Poly<T>>;

impl<T: Ring> MatrixDyn<T> {
    /// Adds two dynamically sized matrices.
    pub fn madd(&self, m: &Self) -> Self {
        madd_gen(self, m)
    }

    /// Multiplies two dynamically sized matrices.
    pub fn mmul<const K: usize>(&self, m: &Self) -> Self {
        mmul_gen(self, m)
    }
}

/// A macro to simplify writing down a [`MatrixDyn`].
///
/// See also [`matrix_mn`](crate::matrix_mn).
///
/// ## Example
///
/// ```
/// # use std::num::Wrapping;
/// # use xmath::data::MatrixDyn;
/// # use xmath::matrix_dyn;
/// # use xmath::traits::Wu8;///
/// let m: MatrixDyn<Wu8> = matrix_dyn!(
///     Wrapping(1), Wrapping(2);
///     Wrapping(3), Wrapping(4)
/// );
///
/// // Poly([Poly([1, 2]), Poly([3, 4])])
/// println!("{:?}", m);
/// ```
#[macro_export]
macro_rules! matrix_dyn {
    ($($($x: expr),*);*) => {
        xmath::data::Poly::new(vec![$(xmath::data::Poly::new(vec![$($x),*])),*])
    };
}

// TODO: Lagrange interpolation

/// A polynomial in two variables.
///
/// You can cast a polynomial `p: Poly<T>` to a two-variable polynomial using
/// either of [`Poly2::of_x`] or [`Poly2::of_y`].
///
/// ## Internal representation
///
/// This is just a wrapper around `Poly<Poly<T>>`, with some extra methods and
/// an alternate display.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Poly2<T: Zero>(pub Poly<Poly<T>>);

impl<T: Zero> Poly2<T> {
    /// Initializes a new polynomial from a polynomial of polynomials.
    pub fn new(v: Poly<Poly<T>>) -> Self {
        Self(v)
    }

    /// Returns whether the underlying slice is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Formats a polynomial, with two specified variable names. Defaults to `x`
    /// and `y`.
    fn fmt_with(&self, f: &mut std::fmt::Formatter<'_>, c1: char, c2: char) -> std::fmt::Result
    where
        T: std::fmt::Display,
    {
        // Are we writing down the `0` polynomial?
        let mut zero = true;
        f.write_char('(')?;

        // Write nonzero coefficients one by one.
        for (n, y) in self.0.as_slice().iter().enumerate() {
            for (m, x) in y.as_slice().iter().enumerate() {
                if !x.is_zero() {
                    if !zero {
                        write!(f, " + ")?;
                    }

                    zero = false;
                    write!(f, "{x} {c1}^{m} {c2}^{n}")?;
                }
            }
        }

        // We write the zero polynomial as "(0)".
        if zero {
            write!(f, "0)")
        } else {
            f.write_char(')')
        }
    }
}

impl<T: Zero> From<Poly<Poly<T>>> for Poly2<T> {
    fn from(p: Poly<Poly<T>>) -> Self {
        Self::new(p)
    }
}

impl<T: Zero> From<Poly<T>> for Poly2<T> {
    fn from(p: Poly<T>) -> Self {
        Self::new(Poly::c(p))
    }
}

impl<T: Zero> IntoIterator for Poly2<T> {
    type Item = Poly<T>;
    type IntoIter = std::vec::IntoIter<Poly<T>>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<T: Zero + std::fmt::Display> std::fmt::Display for Poly2<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.fmt_with(f, 'x', 'y')
    }
}

impl<T: Zero> std::ops::Index<usize> for Poly2<T> {
    type Output = Poly<T>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<T: Zero> std::ops::Index<(usize, usize)> for Poly2<T> {
    type Output = T;

    fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
        &self[i][j]
    }
}

impl<T: Zero> Default for Poly2<T> {
    fn default() -> Self {
        Self::new(Poly::zero())
    }
}

impl<T: Zero + PartialEq> Zero for Poly2<T> {
    fn zero() -> Self {
        Self::default()
    }

    fn is_zero(&self) -> bool {
        self.is_empty()
    }
}

impl<T: Zero> Poly2<T> {
    /// Casts a polynomial in `x` to a two variable polynomial.
    pub fn of_x(p: Poly<T>) -> Self {
        p.into()
    }

    /// Casts a polynomial in `y` to a two variable polynomial.
    pub fn of_y(p: Poly<T>) -> Self {
        let mut res = Vec::new();

        for c in p.into_iter() {
            res.push(Poly::c(c));
        }

        // Safety: trivial.
        unsafe { Self::new(Poly::new_unchecked(res)) }
    }

    /// Creates the polynomial c * xⁿ, for nonzero `c`.
    ///
    /// ## Safety
    ///
    /// The caller must verify `c` is indeed nonzero.
    pub unsafe fn cxn_unchecked(c: T, n: usize) -> Self {
        Poly::cxn_unchecked(c, n).into()
    }

    /// Creates the polynomial c * xⁿ.
    pub fn cxn(c: T, n: usize) -> Self {
        Poly::cxn(c, n).into()
    }

    /// Creates a constant polynomial from a nonzero value.
    ///
    /// ## Safety
    ///
    /// The caller must verify `c` is indeed nonzero.
    pub unsafe fn c_unchecked(c: T) -> Self {
        Poly::c_unchecked(c).into()
    }

    /// Creates a constant polynomial.
    pub fn c(c: T) -> Self {
        Self::cxn(c, 0)
    }

    /// Creates the polynomial xⁿ.
    pub fn xn(n: usize) -> Self
    where
        T: ZeroNeOne,
    {
        Poly::<T>::xn(n).into()
    }

    /// Creates the `x` polynomial.
    pub fn x() -> Self
    where
        T: ZeroNeOne,
    {
        Poly::<T>::x().into()
    }

    /// Creates the polynomial c * yⁿ, for nonzero `c`.
    ///
    /// ## Safety
    ///
    /// The caller must verify `c` is indeed nonzero.
    pub unsafe fn cyn_unchecked(c: T, n: usize) -> Self {
        Self::of_y(Poly::cxn_unchecked(c, n))
    }

    /// Creates the polynomial c * yⁿ.
    pub fn cyn(c: T, n: usize) -> Self {
        Self::of_y(Poly::cxn(c, n))
    }

    /// Creates the polynomial yⁿ.
    pub fn yn(n: usize) -> Self
    where
        T: ZeroNeOne,
    {
        Self::of_y(Poly::xn(n))
    }

    /// Creates the `y` polynomial.
    pub fn y() -> Self
    where
        T: ZeroNeOne,
    {
        Self::of_y(Poly::x())
    }

    /// Evaluates the polynomial at a point.
    pub fn eval(&self, x: &T, y: &T) -> T
    where
        T: AddMonoid + Mul + ZeroNeOne,
    {
        self.0.eval(&Poly::c(y.clone())).eval(x)
    }
}

impl<T: ZeroNeOne> One for Poly2<T> {
    fn one() -> Self {
        Self::xn(0)
    }
}

impl<T: ZeroNeOne> ZeroNeOne for Poly2<T> {}

impl<T: AddGroup> Neg for Poly2<T> {
    fn neg(&self) -> Self {
        Self::new(self.0.neg())
    }

    fn neg_mut(&mut self) {
        self.0.neg_mut()
    }
}

impl<T: AddMonoid> Add for Poly2<T> {
    fn add(&self, x: &Self) -> Self {
        Self::new(self.0.add(&x.0))
    }

    fn add_mut(&mut self, x: &Self) {
        self.0.add_mut(&x.0)
    }
}

impl<T: AddMonoid> AddMonoid for Poly2<T> {}
impl<T: AddMonoid + CommAdd> CommAdd for Poly2<T> {}

impl<T: AddGroup> Sub for Poly2<T> {
    fn sub(&self, x: &Self) -> Self {
        Self::new(self.0.sub(&x.0))
    }

    fn sub_mut(&mut self, x: &Self) {
        self.0.sub_mut(&x.0)
    }
}

impl<T: AddGroup> AddGroup for Poly2<T> {}

impl<T: AddMonoid + Mul> Mul for Poly2<T> {
    fn mul(&self, x: &Self) -> Self {
        Self::new(self.0.mul(&x.0))
    }
}

impl<T: AddMonoid + CommAdd + CommMul> CommMul for Poly2<T> {}
impl<T: Ring + ZeroNeOne> MulMonoid for Poly2<T> {}
impl<T: Ring + ZeroNeOne> Ring for Poly2<T> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algs::upow;
    use std::num::Wrapping;
    use xmath_traits::Wu8;

    #[test]
    fn add() {
        let p: Poly2<Wu8> = Poly2::x().add(&Poly2::cyn(Wrapping(2), 1));
        let q = upow(p, 5);
        println!("{}, {}", q, q.eval(&Wrapping(0), &Wrapping(1)));
    }
}
