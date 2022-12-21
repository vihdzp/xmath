//! Defines traits which provide basic functionality for lists and matrices.
//!
//! We consider two basic kinds of lists:
//!
//! - Statically sized lists such as [`Array`](crate::data::Array), with a
//!   backing `[T; N]`.
//! - Dynamically sized lists such as [`Poly`](crate::data::Poly), with a
//!   backing `Vec`.
//!
//! Other lists can then be built by nesting these, or via wrapper types such as
//! [`Transpose`](crate::data::aliases::Transpose) on these.
//!
//! Correctly managing these two kinds of lists under the same interface is a
//! subtle matter. See [`List`] for more details.

use crate::tuple;

use super::{
    basic::*,
    dim::{CSingle, TypeNum, U1, U2},
};

/// An enum for one of two values, a "finite" `usize` value, or an infinite
/// value. This is used to measure the size of a [`List`].

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dim {
    /// A finite value, stored as a `usize`.
    Fin(usize),

    /// An infinite value.
    Inf,
}

impl Dim {
    /// Returns whether `x < self`.
    pub fn cmp_usize(self, x: usize) -> bool {
        match self {
            Dim::Fin(y) => x < y,
            Dim::Inf => true,
        }
    }
}

/// Represents a structure with a notion of coordinates or coefficients.
///
/// The argument `C` in `List<C>` is a type-level integer that represents the
/// number of dimensions of the list. For instance, a [`LinearModule`]
/// implements `List<U1>` while a [`Matrix`] implements `List<U2>`. Note that a
/// type can implement `List<C>` for more than one integer. For instance,
/// [`MatrixMN`](crate::data::matrix::MatrixMN) implements both `List<U1>` and
/// `List<U2>`, meaning that it can either be indexed by a single coordinate
/// (row-wise) or by two (entry-wise).
///
/// ## Valid indices
///
/// The type of indices doesn't necessarily need to exhaust the list. As such,
/// we consider a set of *valid indices*, being those that can be written to.
///
/// For `i` a valid index, we distinguish two kinds of entries depending on the
/// output of [`List::coeff_ref_gen`] at `i`:
///
/// - `Some(&x)`: *present entries*, have a physical location in memory.
/// - `None`: *ghost entries*, are a stand-in for some other value which depends
///   on the type.
///
/// As an example, consider the [`Poly`](crate::data::Poly) that represents
/// `6 + 8x`. Its inner storage will look something like `vec![6, 8]`. For this
/// `Poly`, the coefficient `8` of x is a present entry, while the coefficient
/// `0` of x² is a ghost entry, as it's not stored in memory. Nevertheless,
/// since `Poly` is dynamically-sized, it's still possible to write to the x²
/// coefficient, meaning `2` is a valid index.
///
/// To make things simpler, we further restrict lists so that their valid
/// entries define an n-dimensional rectangle. This rectangle is the *size* of
/// the list.
pub trait List<C: TypeNum> {
    /// The type of entries in the list.
    type Item;

    /// The size of the list. See the documentation for [`List`] for more info.
    const SIZE: C::Array<Dim>;

    /// Returns a reference to the entry with a certain coefficient. Returns
    /// `None` if said reference doesn't exist.
    ///
    /// This is the more general version of this function, contrast with
    /// [`LinearModule::coeff_ref`] and [`Matrix::coeff_ref`].
    fn coeff_ref_gen(&self, index: &C::Array<usize>) -> Option<&Self::Item>;

    /// Returns the value of the entry with a certain coefficient, defaulting to
    /// 0.
    ///
    /// This is the more general version of this function, contrast with
    /// [`LinearModule::coeff_or_zero`].
    fn coeff_or_zero_gen(&self, index: &C::Array<usize>) -> Self::Item
    where
        Self::Item: Clone + Zero,
    {
        self.coeff_ref_gen(index)
            .map(Clone::clone)
            .unwrap_or_else(Self::Item::zero)
    }

    /// Sets the entry with a certain index with a certain value.
    ///
    /// This is the more general version of this function, contrast with
    /// [`LinearModule::coeff_set_unchecked`] and
    /// [`Matrix::coeff_set_unchecked`].
    ///
    /// ## Safety
    ///
    /// The index must be valid.
    unsafe fn coeff_set_unchecked_gen(
        &mut self,
        index: &C::Array<usize>,
        value: Self::Item,
    );

    /// Sets the entry with a certain index with a certain value.
    ///
    /// This is the more general version of this function, contrast with
    /// [`LinearModule::coeff_set`] and [`Matrix::coeff_set`].
    ///
    /// ## Panics
    ///
    /// This function must panic if and only if it is given an invalid
    /// coefficient.
    fn coeff_set_gen(&mut self, index: &C::Array<usize>, value: Self::Item) {
        if is_valid_index_gen::<C, Self>(index) {
            unsafe { self.coeff_set_unchecked_gen(index, value) }
        } else {
            panic!("the index {:?} is invalid", index.as_ref())
        }
    }

    /// Creates a new list by mapping each present entry by the given function.
    fn map<F: Fn(&Self::Item) -> Self::Item>(&self, f: F) -> Self;

    /// Changes a list by mapping each present entry by the given function.
    fn map_mut<F: Fn(&mut Self::Item)>(&mut self, f: F);
}

/// Returns whether an index is valid for the list.
///
/// This is the more general version of this function, contrast with
/// [`is_valid_index_lin`] and [`is_valid_index`].
pub fn is_valid_index_gen<C: TypeNum, L: List<C> + ?Sized>(
    index: &C::Array<usize>,
) -> bool {
    super::dim::ctuple::pairwise::<C, _, _, _>(
        &L::SIZE,
        index,
        |x: &Dim, y: &usize| x.cmp_usize(*y),
    )
}

/// A [`List`] that is also an [`AddGroup`], such that addition in this
/// structure matches coordinatewise addition, and such that scalar
/// multiplication can be defined.
///
/// Any ghost entries in a module are to be interpreted as zeros.
pub trait Module<C: TypeNum>: AddGroup + List<C>
where
    Self::Item: Ring,
{
    /// Scalar multiplication.
    fn smul(&self, x: &Self::Item) -> Self {
        self.map(|y| y.mul(x))
    }

    /// Mutable scalar multiplication.
    fn smul_mut(&mut self, x: &Self::Item) {
        self.map_mut(|y| y.mul_mut(x));
    }

    // Dot product.
    fn dot(&self, x: &Self) -> Self::Item;
}

/// A trait for a one-dimensional [`Module`].
///
/// The implementation of [`FromIterator`] must write the coefficients in order,
/// interpreting `None` as zeros. It must stop reading from the iterator once
/// no more entries can be written. We provide a default implementation
/// [`LinearModule::from_iter`].
pub trait LinearModule: Module<U1> + FromIterator<Self::Item>
where
    Self::Item: Ring,
{
    /// Returns a reference to the entry with a certain coefficient. Returns
    /// `None` if said reference doesn't exist.
    ///
    /// This is the more specific version of this function, contrast with
    /// [`List::coeff_ref_gen`].
    fn coeff_ref(&self, index: usize) -> Option<&Self::Item> {
        self.coeff_ref_gen(&CSingle(index))
    }

    /// Returns the value of the entry with a certain coefficient, defaulting to
    /// 0.
    ///
    /// This is the more specific version of this function, contrast with
    /// [`List::coeff_or_zero_gen`].
    fn coeff_or_zero(&self, index: usize) -> Self::Item {
        self.coeff_or_zero_gen(&CSingle(index))
    }

    /// Sets the entry with a certain index with a certain value.
    ///
    /// This is the more specific version of this function, contrast with
    /// [`List::coeff_set_unchecked_gen`].
    ///
    /// ## Safety
    ///
    /// The index must be valid.
    unsafe fn coeff_set_unchecked(&mut self, index: usize, value: Self::Item) {
        self.coeff_set_unchecked_gen(&CSingle(index), value);
    }

    /// Sets the entry with a certain index with a certain value.
    ///
    /// This is the more specific version of this function, contrast with
    /// [`List::coeff_set_gen`].
    ///
    /// ## Panics
    ///
    /// This function must panic if and only if it is given an invalid
    /// coefficient.
    fn coeff_set(&mut self, index: usize, value: Self::Item) {
        self.coeff_set_gen(&CSingle(index), value);
    }

    /// Returns some number `i`, such that coefficients from `i` onwards are
    /// either zero or nonexistent. Any coefficient less than the support must
    /// be valid.
    fn support(&self) -> usize;

    /// A default `FromIterator` implementation for linear modules.
    fn from_iter<I: IntoIterator<Item = Self::Item>>(iter: I) -> Self {
        let mut res = Self::zero();

        for (i, x) in iter.into_iter().enumerate() {
            if is_valid_index_lin::<Self>(i) {
                res.coeff_set(i, x);
            } else {
                break;
            }
        }

        res
    }
}

/// Returns whether an index is valid for the linear module.
///
/// This is a more specific version of this function, contrast with
/// [`is_valid_index_gen`].
pub fn is_valid_index_lin<V: LinearModule + ?Sized>(index: usize) -> bool
where
    V::Item: Ring,
{
    is_valid_index_gen::<U1, V>(&CSingle(index))
}

/// The preferred order to write the entries of a matrix in. Sometimes allows
/// for some slight optimization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Direction {
    /// Write things row by row.
    Row,

    /// Write things column by column.
    Column,

    /// Doesn't matter.
    Either,
}

impl Direction {
    /// A `const fn` version of `eq`. Will be deprecated when Rust allows this
    /// automatically.
    pub const fn eq_const(self, p: Self) -> bool {
        self as u8 == p as u8
    }

    /// Returns whether the direction `p` is contained in `self`.
    pub const fn contains(self, p: Self) -> bool {
        self.eq_const(Self::Either) || self.eq_const(p)
    }

    /// Swaps the `Row` direction with the `Column` direction.
    pub const fn transpose(self) -> Self {
        match self {
            Self::Row => Self::Column,
            Self::Column => Self::Row,
            Self::Either => Self::Either,
        }
    }
}

/// A trait for matrices.
///
/// Matrices are just [`Module`]s that are indexed by two coordinates, with some
/// extra methods that allow us to do computations on them.
pub trait Matrix: Module<U2>
where
    Self::Item: Ring,
{
    /// The preferred direction to write the entries of the matrix.
    const DIR: Direction;

    /// Returns a reference to the entry with a certain coefficient. Returns
    /// `None` if said reference doesn't exist.
    ///
    /// This is the more specific version of this function, contrast with
    /// [`List::coeff_ref_gen`].
    fn coeff_ref(&self, row: usize, col: usize) -> Option<&Self::Item> {
        self.coeff_ref_gen(&tuple!(row, col))
    }

    /// Sets the entry with a certain index with a certain value.
    ///
    /// This is the more specific version of this function, contrast with
    /// [`List::coeff_set_unchecked_gen`].
    ///
    /// ## Safety
    ///
    /// The index must be valid.
    unsafe fn coeff_set_unchecked(
        &mut self,
        row: usize,
        col: usize,
        value: Self::Item,
    ) {
        self.coeff_set_unchecked_gen(&tuple!(row, col), value);
    }

    /// Sets the entry with a certain index with a certain value.
    ///
    /// This is the more specific version of this function, contrast with
    /// [`List::coeff_set_gen`].
    ///
    /// ## Panics
    ///
    /// This function must panic if and only if it is given an invalid
    /// coefficient.
    fn coeff_set(&mut self, row: usize, col: usize, value: Self::Item) {
        self.coeff_set_gen(&tuple!(row, col), value);
    }

    /// The support of a certain column, in the sense of
    /// [`LinearModule::support`].
    fn col_support(&self, index: usize) -> usize;

    /// The support of a certain row, in the sense of
    /// [`LinearModule::support`].
    fn row_support(&self, index: usize) -> usize;

    /// The height of the matrix, in the sense of [`LinearModule::support`].
    fn height(&self) -> usize;

    /// The width of the matrix, in the sense of [`LinearModule::support`].
    fn width(&self) -> usize;

    /// Writes a matrix in row order, interpreting `None` as zeros. It must stop
    /// reading from the iterator once no more entries can be written.
    ///
    /// The default implementation initializes a zero matrix, then writes
    /// entries one by one via [`Matrix::coeff_set`].
    ///
    /// You can use [`Matrix::DIR`] to figure out whether
    /// [`Matrix::collect_row`] or [`Matrix::collect_col`] should be used.
    fn collect_row<I: Iterator<Item = Self::Item>, J: Iterator<Item = I>>(
        iter: J,
    ) -> Self {
        let mut res = Self::zero();

        for (row, iter) in iter.into_iter().enumerate() {
            for (col, value) in iter.into_iter().enumerate() {
                res.coeff_set(row, col, value)
            }
        }

        res
    }

    /// Writes a matrix in column order, interpreting `None` as zeros. It must
    /// stop reading from the iterator once no more entries can be written.
    ///
    /// The default implementation initializes a zero matrix, then writes
    /// entries one by one via [`Matrix::coeff_set`].
    ///
    /// You can use [`Matrix::DIR`] to figure out whether
    /// [`Matrix::collect_row`] or [`Matrix::collect_col`] should be used.
    fn collect_col<I: Iterator<Item = Self::Item>, J: Iterator<Item = I>>(
        iter: J,
    ) -> Self {
        let mut res = Self::zero();

        for (c, iter) in iter.into_iter().enumerate() {
            for (r, x) in iter.into_iter().enumerate() {
                res.coeff_set(r, c, x)
            }
        }

        res
    }

    /// Creates a new matrix, filling up to the first `h` rows and `w` columns
    /// with the outputs of `f`, in the preferred direction of the matrix type.
    ///
    /// Any values not filled default to `0`.
    fn from_fn<F: Copy + FnMut(usize, usize) -> Self::Item>(
        h: usize,
        w: usize,
        mut f: F,
    ) -> Self {
        if Self::DIR.contains(Direction::Row) {
            Self::collect_row((0..h).map(|i| (0..w).map(move |j| f(i, j))))
        } else {
            Self::collect_col((0..w).map(|j| (0..h).map(move |i| f(i, j))))
        }
    }
}

/// Returns whether `(i, j)` is a valid index for the matrix.
///
/// This is a more specific version of this function, contrast with
/// [`is_valid_index_gen`].
pub fn is_valid_index<M: Matrix>(i: usize, j: usize) -> bool
where
    M::Item: Ring,
{
    is_valid_index_gen::<U2, M>(&tuple!(i, j))
}

/// The height of the size of the matrix.
pub const fn size_height<M: Matrix>() -> Dim
where
    M::Item: Ring,
{
    M::SIZE.0
}

/// The width of the size of the matrix.
pub const fn size_width<M: Matrix>() -> Dim
where
    M::Item: Ring,
{
    M::SIZE.1 .0
}
