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

use super::{
    basic::*,
    dim::{CSingle, TypeNum, U1, U2},
};

/// (UPDATE THIS!!!!)
///
/// A trait for a value representing the dimension of a [`LinearModule`], the
/// width and height of a [`Matrix`], among other things.
///
/// This trait is only implemented for two types, `usize` and [`Inf`]. A `usize`
/// dimension represents a statically sized object, while an `Inf` dimension
/// represents a dynamically sized object. This trait should not be implemented
/// for anything else.
///
/// We implement this at the type level instead of using something like an enum,
/// as it gives us more freedom to use this information at compile time.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dim {
    Fin(usize),
    Inf,
}

impl Dim {
    pub fn cmp_usize(self, x: usize) -> bool {
        match self {
            Dim::Fin(y) => x < y,
            Dim::Inf => true,
        }
    }
}

/// Represents a structure with a notion of coordinates or coefficients.
///
/// The argument `C` in `List<C>` is a tuple that whether each dimension of the
/// list (width, height, etc.) is statically or dynamically sized. Note that a
/// type can implement `List<C>` for more than one kind of tuple. For instance,
/// [`MatrixMN`](crate::data::matrix::MatrixMN) implements both `List<usize>`
/// and `List<(usize, usize)>`, meaning that it can either be indexed by a
/// single coordinate (row-wise) or by two (entry-wise), and is statically sized
/// in both cases.
///
/// ## Valid indices
///
/// The type of indices doesn't necessarily need to exhaust the list. As such,
/// we consider a set of *valid indices*, being those that can be written to.
///
/// For `i` a valid index, we consider two kinds of entries depending on the
/// output of [`List::coeff_ref`] at `i`:
///
/// - `Some(&x)`: *present entries*, have a physical location in memory.
/// - `None`: *ghost entries*, are a stand-in for some other value which depends
///   on the type.
///
/// As an example, consider the [`Poly`](crate::data::poly::Poly) that
/// represents `6 + 8x`. Its inner storage will look something like
/// `vec![6, 8]`. For this `Poly`, the coefficient `8` of x is a present entry,
/// while the coefficient `0` of x² is a ghost entry, as it's not stored in
/// memory. Nevertheless, since `Poly` is dynamically-sized, it's still possible
/// to write to the x² coefficient, meaning `2` is a valid index.
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
    fn coeff_ref_gen(&self, i: &C::Array<usize>) -> Option<&Self::Item>;

    /// Sets the entry with a certain index with a certain value.
    ///
    /// ## Safety
    ///
    /// The index must be valid.
    unsafe fn coeff_set_unchecked_gen(
        &mut self,
        i:& C::Array<usize>,
        x: Self::Item,
    );

    /// Sets the entry with a certain index with a certain value.
    ///
    /// ## Panics
    ///
    /// This function must panic if and only if it is given an invalid
    /// coefficient.
    fn coeff_set_gen(&mut self, i: &C::Array<usize>, x: Self::Item) {
        if list::is_valid_index::<C, Self>(i) {
            unsafe { self.coeff_set_unchecked_gen(i, x) }
        } else {
            panic!("the index {:?} is invalid", i.as_ref())
        }
    }

    /// Creates a new list by mapping each present entry by the given function.
    fn map<F: Fn(&Self::Item) -> Self::Item>(&self, f: F) -> Self;

    /// Changes a list by mapping each present entry by the given function.
    fn map_mut<F: Fn(&mut Self::Item)>(&mut self, f: F);
}

pub mod list {
    use crate::traits::dim;

    use super::*;

    /// Returns whether `i` is a valid index for the list.
    pub fn is_valid_index<C: TypeNum, L: List<C> + ?Sized>(
        i: &C::Array<usize>,
    ) -> bool {
        dim::ctuple::pairwise::<C, _, _, _>(
            &L::SIZE,
            i,
            |x: &Dim, y: &usize| x.cmp_usize(*y),
        )
    }

    /// Returns the value of the entry with a certain coefficient, defaulting to
    /// 0.
    fn coeff_or_zero_gen<C: TypeNum, L: List<C>>(
        l: &L,
        i: &C::Array<usize>,
    ) -> L::Item
    where
        L::Item: Clone + Zero,
    {
        l.coeff_ref_gen(i)
            .map(Clone::clone)
            .unwrap_or_else(L::Item::zero)
    }
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

/// A [`Module`] whose coefficients are indexed by a `usize`.
///
/// The implementation of [`FromIterator`] must write the coefficients in order,
/// interpreting `None` as zeros. It must stop reading from the iterator once
/// no more entries can be written. We provide a default implementation
/// [`linear_module::from_iter`].
pub trait LinearModule: Module<U1> + FromIterator<Self::Item>
where
    Self::Item: Ring,
{
    fn coeff_ref(&self, i: usize) -> Option<&Self::Item> {
        self.coeff_ref_gen(&CSingle(i))
    }

    unsafe fn coeff_set_unchecked(&mut self, i: usize, x: Self::Item) {
        self.coeff_set_unchecked_gen(&CSingle(i), x);
    }

    fn coeff_set(&mut self, i: usize, x: Self::Item) {
        self.coeff_set_gen(&CSingle(i), x);
    }

    /// Returns some number `i`, such that coefficients from `i` onwards are
    /// either zero or nonexistent. Any coefficient less than the support must
    /// be valid.
    fn support(&self) -> usize;
}

pub mod linear_module {
    use crate::traits::dim::CSingle;

    use super::*;

    pub fn is_valid_index<V: LinearModule>(i: usize) -> bool
    where
        V::Item: Ring,
    {
        list::is_valid_index::<U1, V>(&CSingle(i))
    }

    /// A default `FromIterator` implementation for linear modules.
    pub fn from_iter<V: LinearModule, I: IntoIterator<Item = V::Item>>(
        iter: I,
    ) -> V
    where
        V::Item: Ring,
    {
        let mut res = V::zero();
        let size = V::SIZE;

        for (i, x) in iter.into_iter().enumerate() {
            if is_valid_index::<V>(i) {
                res.coeff_set(i, x);
            } else {
                break;
            }
        }

        res
    }
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
    /// The default implementation initializes a zero matrix, then writes
    /// entries one by one via [`List::coeff_set`].
    ///
    /// When either [`Matrix::collect_row`] or [`Matrix::collect_col`] can be
    /// used, the former should be the default, as it reflects the layout of the
    /// most common kinds of matrices.
    fn collect_row<I: Iterator<Item = Self::Item>, J: Iterator<Item = I>>(
        iter: J,
    ) -> Self {
        let mut res = Self::zero();

        for (r, iter) in iter.into_iter().enumerate() {
            for (c, x) in iter.into_iter().enumerate() {
                todo!()
                // res.coeff_set(i, x)
            }
        }

        todo!()
    }

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

pub mod matrix {
    use super::*;

    pub const fn size_height<M: Matrix>() -> Dim
    where
        M::Item: Ring,
    {
        M::SIZE.0
    }

    pub const fn size_width<M: Matrix>() -> Dim
    where
        M::Item: Ring,
    {
        M::SIZE.1 .0
    }
}
