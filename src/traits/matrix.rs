//! Defines traits which provide functionality for lists and matrices.
//!
//! We consider two basic kinds of lists:
//!
//! - Statically sized lists such as [`Array`](crate::data::array::Array), with
//!   a backing `[T; N]`.
//! - Dynamically sized lists such as [`Poly`](crate::data::poly::Poly), with a
//!   backing `Vec`.
//!
//! Other lists can then be built by nesting these, or via wrapper types such as
//! [`Transpose`](crate::data::matrix::Transpose) on these.
//!
//! Correctly managing these two kinds of lists under the same interface is a
//! subtle matter. See [`List`] for more details.

use super::basic::*;

/// Represents a structure with a notion of coordinates or coefficients.
///
/// Note that a list can be indexed by more than one kind of coefficient. For
/// instance, [`MatrixMN`](crate::data::matrix::MatrixMN) can be indexed by
/// either `usize` or `(usize, usize)`.
///
/// ## Valid coefficients
///
/// The type of coefficients doesn't necessarily need to exhaust the list. As
/// such, we consider a set of *valid coefficients*, being those that can be
/// written to.
///
/// For `i` a valid coefficient, we consider two kinds of entries depending on
/// the output of [`List::coeff_ref`] at `i`:
///
/// - `Some(&x)`: *present entries*, have a physical location in memory.
/// - `None`: *ghost entries*, are a stand-in for some other value which depends
///   on the type.
///
/// As an example, in the [`Poly`](crate::data::poly::Poly) that represents
/// `2 + 3x`, the coefficient `3` of x is a present entry, while the coefficient
/// `0` of x² is a ghost entry, as it's not stored in memory. Nevertheless,
/// since `Poly` is dynamically-sized, it's still possible to write to the x²
/// coefficient.
pub trait List<C> {
    /// The type of entries in the list.
    type Item;

    /// Returns whether the coefficient is valid.
    fn is_valid_coeff(i: C) -> bool;

    /// Returns a reference to the entry with a certain coefficient. Returns
    /// `None` if said reference doesn't exist.
    fn coeff_ref(&self, i: C) -> Option<&Self::Item>;

    /// Returns the value of the entry with a certain coefficient, defaulting to
    /// 0.
    fn coeff(&self, i: C) -> Self::Item
    where
        Self::Item: Clone + Zero,
    {
        self.coeff_ref(i)
            .map(Clone::clone)
            .unwrap_or_else(Self::Item::zero)
    }

    /// Sets the entry with a certain coefficient with a certain value. The
    /// allowed coefficients depend on the type.
    ///
    /// ## Panics
    ///
    /// This function must panic if and only if it is given an invalid
    /// coefficient.
    fn coeff_set(&mut self, i: C, x: Self::Item);
}

/// A dynamically typed iterator.
///
/// This is used in the type signature of [`ListIter::iter`]. Ideally this
/// wouldn't be necessary, and we could just require an [`IntoIterator`]
/// implementation on some dummy type `Iter<'a, &'a Self, C>`. Unfortunately,
/// implementing this for wrapper types such as [`Transpose`] invokes an old and
/// unfixed [compiler bug](https://github.com/rust-lang/rust/issues/37748).
pub struct BoxIter<'a, T>(Box<dyn Iterator<Item = T> + 'a>);

impl<'a, T> BoxIter<'a, T> {
    pub fn new<I: Iterator<Item = T> + 'a>(i: I) -> Self {
        Self(Box::new(i))
    }
}

impl<'a, T> Iterator for BoxIter<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

/// A [`List`] with iterator capabilities.
pub trait ListIter<C>: List<C> {
    /// Iterates over all entries of the list in some order.
    fn iter(&self) -> BoxIter<&Self::Item>;

    /// Iterates over all pairs of entries of two lists, such that at least one
    /// of them is present, in some order.
    ///
    /// If either entry is a ghost entry, a reference to it handled by the
    /// iterator should be returned.
    fn iter_pair<'a>(
        &'a self,
        x: &'a Self,
    ) -> BoxIter<(&'a Self::Item, &'a Self::Item)>;

    /// Maps all present entries through a function.
    fn map<F: FnMut(&Self::Item) -> Self::Item>(&self, f: F) -> Self;

    /// Mutably maps all present entries through a function `f`.
    fn map_mut<F: FnMut(&mut Self::Item)>(&mut self, f: F);

    /*
    /// Performs a pairwise operation.
    ///
    /// The operation is performed on every pair of entries with the same
    /// coefficient where at least one of them is present.
    fn pairwise<F: FnMut(&Self::Item, &Self::Item) -> Self::Item>(
        &self,
        x: &Self,
        f: F,
    ) -> Self;

    /// Performs a mutable pairwise operation.
    ///
    /// The operation is performed on every pair of entries with the same
    /// coefficient where at least one of them is present.
    fn pairwise_mut<F: FnMut(&mut Self::Item, &Self::Item)>(
        &mut self,
        x: &Self,
        f: F,
    ); */
}

/// A [`ListIter`] that is also an [`AddGroup`], such that addition in this
/// structure matches coordinatewise addition, and such that scalar
/// multiplication can be defined.
///
/// Any ghost entries in a module are to be interpreted as zeros.
pub trait Module<C>: AddGroup + ListIter<C>
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
    //fn dot(&self, x: &Self) -> Self::Item ;
}

/// A trait for a value representing the dimensions of a [`LinearModule`] or a
/// [`Matrix`](crate::data::matrix::Matrix).
///
/// This trait is only implemented for two types, `usize` and [`Inf`]. A `usize`
/// dimension represents a statically sized object, while an `Inf` dimension
/// represents a dynamically sized object. This trait should not be implemented
/// for anything else.
pub trait Dim: Copy + Eq {}

impl Dim for usize {}

/// Represents an infinite dimension. In practice, this means that the object in
/// question is dynamically sized.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Inf;

impl Dim for Inf {}

/// A [`Module`] whose coefficients are indexed by a `usize`.
///
/// The implementation of [`FromIterator`] must write the coefficients in order,
/// interpreting `None` as zeros. It must stop reading from the iterator once
/// no more entries can be written.
pub trait LinearModule: Module<usize> + FromIterator<Self::Item>
where
    Self::Item: Ring,
{
    /// Whether the module is statically or dynamically sized.
    type DimType: Dim;

    /// The dimension of the module.
    const DIM: Self::DimType;

    /// Returns some number `i`, such that coefficients from `i` onwards are
    /// either zero or nonexistent.
    fn support(&self) -> usize;
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
pub trait Matrix: Module<(usize, usize)>
where
    Self::Item: Ring,
{
    /// Whether the height is statically or dynamically sized.
    type HeightType: Dim;

    /// Whether the width is statically or dynamically sized.
    type WidthType: Dim;

    /// The height of the matrix.
    const HEIGHT: Self::HeightType;

    /// The width of the matrix.
    const WIDTH: Self::WidthType;

    /// The preferred direction to write the entries of the matrix.
    const DIR: Direction;

    /// The support of the `j`-th column, in the sense of
    /// [`LinearModule::support`].
    fn col_support(&self, j: usize) -> usize;

    /// The support of the `i`-th row, in the sense of
    /// [`LinearModule::support`].
    fn row_support(&self, i: usize) -> usize;

    /// The height of the matrix, in the sense of [`LinearModule::support`].
    fn height(&self) -> usize;

    /// The width of the matrix, in the sense of [`LinearModule::support`].
    fn width(&self) -> usize;

    /// Writes a matrix in row order, interpreting `None` as zeros. It must stop
    /// reading from the iterator once no more entries can be written.
    ///
    /// When either [`Matrix::collect_row`] or [`Matrix::collect_col`] can be
    /// used, the former should be the default, as it reflects the layout of the
    /// most common kinds of matrices.
    fn collect_row<I: Iterator<Item = Self::Item>, J: Iterator<Item = I>>(
        iter: J,
    ) -> Self;

    /// Writes a matrix in column order, interpreting `None` as zeros. It must
    /// stop reading from the iterator once no more entries can be written.
    ///
    /// When either [`Matrix::collect_row`] or [`Matrix::collect_col`] can be
    /// used, the former should be the default, as it reflects the layout of the
    /// most common kinds of matrices.
    fn collect_col<I: Iterator<Item = Self::Item>, J: Iterator<Item = I>>(
        iter: J,
    ) -> Self;

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
