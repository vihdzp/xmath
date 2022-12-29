//! Implements type-level integers, and tuple types associated to these.
//!
//! This allows us to work with lists with a compile-time number of coordinates
//! (although almost always we work with 1 or 2 coordinates).

use std::fmt::Write;

pub use xmath_core::{Succ, U1};
pub use xmath_macro::{ArrayFromIter, Transparent};
use xmath_traits::*;

/// An enum for one of two values, a "finite" `usize` value, or an infinite
/// value. This is used to measure the size of a [`List`].

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dim {
    /// A finite value, stored as a `usize`.
    Fin(usize),

    /// An infinite value.
    Inf,
}

impl std::fmt::Display for Dim {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Dim::Fin(x) => write!(f, "{}", x),
            Dim::Inf => f.write_char('∞'),
        }
    }
}

impl Dim {
    /// Returns whether `x < self`.
    pub const fn cmp_usize(self, x: usize) -> bool {
        match self {
            Dim::Fin(y) => x < y,
            Dim::Inf => true,
        }
    }

    /// The minimum of a `Dim` and a `usize` value.
    pub const fn min(self, x: usize) -> usize {
        match self {
            Dim::Fin(y) => {
                if x < y {
                    x
                } else {
                    y
                }
            }
            Dim::Inf => x,
        }
    }
}

/// The array type associated to the type-level integer [`U1`].
///
/// If you just want an array with a single entry, consider using
/// [`Array<1>`](crate::data::Array) instead.
#[repr(transparent)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Transparent, ArrayFromIter,
)]
pub struct Array1<T>(pub T);

/// Displays the single entry.
impl<T: std::fmt::Display> std::fmt::Display for Array1<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A pair of elements with `repr(C)`. Used to build the array types for
/// type-level integers, see also [`TypeNum`].
///
/// This doesn't implement any algebraic traits, see [`Pair`](crate::data::Pair)
/// for that usage.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ArrayPair<T, U>(pub T, pub U);

impl<T, U> ArrayPair<T, U> {
    /// Initializes a new pair.
    pub fn new(x: T, y: U) -> Self {
        Self(x, y)
    }
}

/// The array type associated to the type-level integer [`U2`].
///
/// If you just want an array with two entries, consider using
/// [`Array<2>`](crate::data::Array) instead.
pub type Array2<T> = ArrayPair<T, Array1<T>>;

/// The array type associated to the type-level integer [`U3`].
///
/// If you just want an array with three entries, consider using
/// [`Array<3>`](crate::data::Array) instead.
pub type C3<T> = ArrayPair<T, Array2<T>>;

impl<T> Array2<T> {
    /// Initializes a new `Array2`.
    pub const fn new2(x: T, y: T) -> Self {
        crate::array!(x, y)
    }

    /// The first entry of the `Array2`.
    pub const fn fst(&self) -> &T {
        &self.0
    }

    /// An alias for `[Self::fst]`.
    pub const fn height(&self) -> &T {
        self.fst()
    }

    /// The second entry of the `Array2`.
    pub const fn snd(&self) -> &T {
        &self.1 .0
    }

    /// An alias for `[Self::snd]`.
    pub const fn width(&self) -> &T {
        self.snd()
    }

    /// Given the tuple `(x, y)`, returns `(y, x)`.
    pub const fn swap(self) -> Self
    where
        T: Copy,
    {
        Self::new2(self.1 .0, self.0)
    }

    /// Converts the tuple into an array.
    pub const fn to_array(self) -> [T; 2]
    where
        T: Copy,
    {
        [self.0, self.1 .0]
    }
}

/// A trait for a type with a `const` value of zero. This differs from [`Zero`]
/// since that traits allows the zero value to be dynamically allocated (albeit
/// still constant).
pub trait ConstZero {
    /// The constant zero value of the type.
    const ZERO: Self;
}

impl ConstZero for usize {
    const ZERO: Self = 0;
}

impl ConstZero for Dim {
    const ZERO: Self = Self::Fin(0);
}

impl<T: ConstZero> ConstZero for Array1<T> {
    const ZERO: Self = Self(T::ZERO);
}

impl<T: ConstZero, U: ConstZero> ConstZero for ArrayPair<T, U> {
    const ZERO: Self = Self(T::ZERO, U::ZERO);
}

/// A trait for an array of elements associated to a type-level integer.
///
/// This trait is only implemented for `C1<T>`, `Array2<T> = CPair<T, C1<T>>`,
/// `C3<T> = CPair<T, Array2<T>>`, etc. and shouldn't be implemented for any other
/// types.
pub trait ArrayTuple: ConstZero + ArrayLike {}

impl<T: ConstZero> ArrayTuple for Array1<T> {}

impl<T: std::fmt::Display> std::fmt::Display for Array2<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}×{}", self.fst(), self.snd())
    }
}

impl<T: ArrayLike> SliceLike for ArrayPair<T::Item, T> {
    type Item = T::Item;

    fn as_slice(&self) -> &[Self::Item] {
        ArrayLike::as_ref(self)
    }

    fn as_mut_slice(&mut self) -> &mut [Self::Item] {
        ArrayLike::as_mut(self)
    }
}

impl<T: ArrayLike> ArrayLike for ArrayPair<T::Item, T> {
    const LEN: usize = T::LEN + 1;

    fn from_iter_mut<I: Iterator<Item = Self::Item>>(iter: &mut I) -> Self {
        Self::new(iter.next().unwrap(), T::from_iter_mut(iter))
    }
}

impl<T: ArrayLike> FromIterator<T::Item> for ArrayPair<T::Item, T> {
    fn from_iter<I: IntoIterator<Item = T::Item>>(iter: I) -> Self {
        ArrayLike::from_iter(iter)
    }
}

impl<T: ArrayTuple> ArrayTuple for ArrayPair<T::Item, T> where T::Item: ConstZero {}

/// The zero element is the tuple all of whose entries are 0.
impl<T: Zero, U: ArrayTuple<Item = T> + Eq> Zero for U {
    fn zero() -> Self {
        std::iter::repeat_with(T::zero).collect()
    }
}

/// The one element is the tuple all of whose entries are 1.
impl<T: One, U: ArrayTuple<Item = T> + Eq> One for U {
    fn one() -> Self {
        std::iter::repeat_with(T::one).collect()
    }
}

/// A type level integer, starting from 1.
///
/// Each type level integer `N` has an associated array type with guaranteed
/// same layout as `[T; N]`.
pub trait TypeNum {
    /// The constant associated with the type.
    const VAL: usize;

    /// An array of the corresponding length.
    type Array<T: ConstZero>: ArrayTuple<Item = T>;
}

impl TypeNum for U1 {
    const VAL: usize = 1;
    type Array<T: ConstZero> = Array1<T>;
}

impl<D: TypeNum> TypeNum for Succ<D> {
    const VAL: usize = D::VAL + 1;
    type Array<T: ConstZero> = ArrayPair<T, D::Array<T>>;
}

/// Returns the type-level integer associated with a literal.
///
/// ## Example
///
/// ```
/// # use xmath::traits::{U1, Succ};
/// # use xmath::dim;
/// let x: dim!(1) = U1;
/// let y: dim!(2) = Succ(x);
/// let z: dim!(3) = Succ(y);
/// ```
#[macro_export]
macro_rules! dim {
    (0) => {
        compile_error!("dimension 0 is invalid")
    };
    ($n: literal) => {
        xmath_macro::dim!($n)
    };
}

/// The type-level integer `2`.
pub type U2 = Succ<U1>;

/// The type-level integer `3`.
pub type U3 = Succ<U2>;

/// Returns an [`ArrayTuple`] from a list of values.
///
/// ## Example
///
/// ```
/// # use xmath::{array_type, array};
/// let x: array_type!(usize; 1) = array!(0);
/// let y: array_type!(usize; 2) = array!(0, 1);
/// let z: array_type!(usize; 3) = array!(0, 1, 2);
/// ```
#[macro_export]
macro_rules! array {
    ($t: expr) => {
        xmath_traits::Array1($t)
    };
    ($t: expr, $($ts: expr),*) => {
        xmath_traits::ArrayPair($t, xmath_traits::array!($($ts),*))
    };
}

/// `array_type!(T; N)` returns the [`ArrayTuple`] type of item `T` and length
/// `N`.
///
/// ## Example
///
/// ```
/// # use xmath::traits::{Array1, ArrayPair};
/// # use xmath::array_type;
/// let x: array_type!(usize; 1) = Array1(0);
/// let y: array_type!(usize; 2) = ArrayPair(0, Array1(1));
/// let z: array_type!(usize; 3) = ArrayPair(0, ArrayPair(1, Array1(2)));
/// ```
#[macro_export]
macro_rules! array_type {
    ($t: ty; $n: literal) => {
        <xmath_traits::dim!($n) as xmath_traits::TypeNum>::Array<$t>
    };
}

/// An iterator that goes over all tuples whose entries are elementwise lesser
/// than a limiting tuple.
///
/// Each step, the first possible entry of the output increments by 1, and all
/// previous ones reset to 0.
pub struct TupleIter<T> {
    /// The current tuple for the iterator.
    current: T,

    /// The maximum tuple for the iterator.
    limit: T,

    /// Whether the iterator has finished.
    finished: bool,
}

impl<T: Zero + SliceLike<Item = usize>> TupleIter<T> {
    /// An iterator that has exhausted all its values.
    pub fn exhausted() -> Self {
        Self {
            current: T::zero(),
            limit: T::zero(),
            finished: true,
        }
    }

    /// Declares a new tuple iterator with the specified limit.
    pub fn new(limit: T) -> Self {
        // If any entry of the limit is 0, the iterator is already done.
        for &x in limit.as_slice() {
            if x == 0 {
                return Self::exhausted();
            }
        }

        Self {
            current: T::zero(),
            limit,
            finished: false,
        }
    }
}

/// Increments the current tuple, returns whether the iterator is finished.
fn increment(current: &mut [usize], limit: &[usize]) -> bool {
    for (x, &y) in current.iter_mut().zip(limit) {
        if *x + 1 < y {
            *x += 1;
            return false;
        } else {
            *x = 0;
        }
    }

    true
}

impl<T: Clone + SliceLike<Item = usize>> Iterator for TupleIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            None
        } else {
            let next = self.current.clone();
            self.finished = increment(self.current.as_mut_slice(), &*self.limit.as_mut_slice());
            Some(next)
        }
    }
}

/// The element-wise minimum of two tuples.
pub fn min<C: TypeNum>(x: &C::Array<Dim>, y: &C::Array<usize>) -> C::Array<usize> {
    let mut res = C::Array::<usize>::ZERO;

    for i in 0..C::VAL {
        res.as_mut_slice()[i] = x.as_slice()[i].min(y.as_slice()[i]);
    }

    res
}

/// Returns whether `f` holds for all pairs of entries with the same index.
pub fn pairwise<C: TypeNum, T: ConstZero, U: ConstZero, F: Fn(&T, &U) -> bool>(
    x: &C::Array<T>,
    y: &C::Array<U>,
    f: F,
) -> bool {
    for i in 0..C::VAL {
        if !f(&x.as_slice()[i], &y.as_slice()[i]) {
            return false;
        }
    }

    true
}

#[cfg(test)]
mod test {
    use super::*;

    /// Checks that the macros work as expected.
    #[test]
    fn tuple() {
        let _: array_type!(usize; 3) = array!(1, 2, 3);
        let _: <dim!(3) as super::TypeNum>::Array<usize> = array!(1, 2, 3);
    }

    /// Checks that [`TupleIter`] outputs things in order.
    #[test]
    fn tuple_iter_1() {
        let res = [[0, 0], [1, 0], [0, 1], [1, 1], [0, 2], [1, 2]];
        for (i, x) in TupleIter::new(Array2::new2(2, 3)).enumerate() {
            assert_eq!(x.to_array(), res[i]);
        }
    }

    /// Checks that [`TupleIter`] handles the empty case accordingly.
    #[test]
    fn tuple_iter_2() {
        assert_eq!(TupleIter::new(array!(1, 2, 0)).next(), None)
    }
}
