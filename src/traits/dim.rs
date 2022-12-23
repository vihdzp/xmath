//! Implements type-level integers, and tuple types associated to these.
//!
//! This allows us to work with lists with a compile-time number of coordinates
//! (although almost always we work with 1 or 2 coordinates).

use super::{
    basic::{One, Zero},
    matrix::Dim,
};
pub use xmath_core::{Succ, U1};

/// A single element with `repr(transparent)`.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct C1<T>(pub T);

/// Reads a single element from the iterator and returns the resulting `C1`.
impl<T> FromIterator<T> for C1<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self(iter.into_iter().next().unwrap())
    }
}

/// Returns a reference to the single entry.
impl<T> AsRef<[T]> for C1<T> {
    fn as_ref(&self) -> &[T] {
        std::slice::from_ref(&self.0)
    }
}

/// Returns a mutable reference to the single entry.
impl<T> AsMut<[T]> for C1<T> {
    fn as_mut(&mut self) -> &mut [T] {
        std::slice::from_mut(&mut self.0)
    }
}

/// Displays the single entry.
impl<T: std::fmt::Display> std::fmt::Display for C1<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A pair of elements with `repr(C)`.
///
/// This doesn't implement any algebraic traits, see [`Pair`](crate::data::Pair)
/// for that usage.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CPair<T, U>(pub T, pub U);

/// The array type associated with [`U2`].
pub type C2<T> = CPair<T, C1<T>>;

/// The array type associated with [`U3`].
pub type C3<T> = CPair<T, C2<T>>;

impl<T> C2<T> {
    /// Initializes a new `C2`.
    pub const fn new(x: T, y: T) -> Self {
        crate::tuple!(x, y)
    }

    /// The first entry of the `C2`.
    pub const fn fst(&self) -> &T {
        &self.0
    }

    /// An alias for `[Self::fst]`.
    pub const fn height(&self) -> &T {
        self.fst()
    }

    /// The second entry of the `C2`.
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
        Self::new(self.1 .0, self.0)
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

impl<T: ConstZero> ConstZero for C1<T> {
    const ZERO: Self = Self(T::ZERO);
}

impl<T: ConstZero, U: ConstZero> ConstZero for CPair<T, U> {
    const ZERO: Self = Self(T::ZERO, U::ZERO);
}

/// Represents an array of elements associated to a type-level integer, with the
/// same layout and alignment as `[Self::Item; Self::Len]`.
///
/// This trait is only implemented for `C1<T>`, `CPair<T, C1<T>>`,
/// etc. and shouldn't be implemented for any other types.
pub trait CTuple:
    ConstZero + FromIterator<Self::Item> + AsRef<[Self::Item]> + AsMut<[Self::Item]>
{
    /// The length of the tuple.
    const LEN: usize;

    /// The item in the tuple.
    type Item: ConstZero;
}

impl<T: ConstZero> CTuple for C1<T> {
    const LEN: usize = 1;
    type Item = T;
}

/// Reads elements from the iterator until the tuple is filled.
impl<T: CTuple> FromIterator<T::Item> for CPair<T::Item, T> {
    fn from_iter<I: IntoIterator<Item = T::Item>>(iter: I) -> Self {
        let mut iter = iter.into_iter();
        Self(iter.next().unwrap(), iter.collect())
    }
}

/// Returns the underlying slice of elements for the tuple.
impl<T: CTuple> AsRef<[T::Item]> for CPair<T::Item, T> {
    fn as_ref(&self) -> &[T::Item] {
        unsafe {
            std::slice::from_raw_parts(
                self as *const Self as *const _,
                Self::LEN,
            )
        }
    }
}

/// Returns the underlying mutable slice of elements for the tuple.
impl<T: CTuple> AsMut<[T::Item]> for CPair<T::Item, T> {
    fn as_mut(&mut self) -> &mut [T::Item] {
        unsafe {
            std::slice::from_raw_parts_mut(
                self as *mut Self as *mut _,
                Self::LEN,
            )
        }
    }
}

impl<T: std::fmt::Display> std::fmt::Display for C2<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}×{}", self.fst(), self.snd())
    }
}

impl<T: CTuple> CTuple for CPair<T::Item, T> {
    const LEN: usize = T::LEN + 1;
    type Item = T::Item;
}

/// The zero element is the tuple all of whose entries are 0.
impl<T: Zero, U: CTuple<Item = T> + Eq> Zero for U {
    fn zero() -> Self {
        std::iter::repeat_with(T::zero).collect()
    }
}

/// The one element is the tuple all of whose entries are 1.
impl<T: One, U: CTuple<Item = T> + Eq> One for U {
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
    type Array<T: ConstZero>: CTuple<Item = T>;
}

impl TypeNum for U1 {
    const VAL: usize = 1;
    type Array<T: ConstZero> = C1<T>;
}

impl<D: TypeNum> TypeNum for Succ<D> {
    const VAL: usize = D::VAL + 1;
    type Array<T: ConstZero> = CPair<T, D::Array<T>>;
}

/// Returns the type-level integer associated with a literal.
///
/// ## Example
///
/// ```
/// # use xmath::traits::dim::{U1, Succ};
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

/// Returns a [`CTuple`] from a list of values.
///
/// ## Example
///
/// ```
/// # use xmath::traits::dim::{CPair, C1};
/// # use xmath::{ctuple, tuple};
/// let x: ctuple!(usize; 1) = tuple!(0);
/// let y: ctuple!(usize; 2) = tuple!(0, 1);
/// let z: ctuple!(usize; 3) = tuple!(0, 1, 2);
/// ```
#[macro_export]
macro_rules! tuple {
    ($t: expr) => {
        $crate::traits::dim::C1($t)
    };
    ($t: expr, $($ts: expr),*) => {
        $crate::traits::dim::CPair($t, $crate::tuple!($($ts),*))
    };
}

/// `ctuple!(T; N)` returns the [`CTuple`] type of item `T` and length `N`.
///
/// ## Example
///
/// ```
/// # use xmath::traits::dim::{CPair, C1};
/// # use xmath::ctuple;
/// let x: ctuple!(usize; 1) = C1(0);
/// let y: ctuple!(usize; 2) = CPair(0, C1(1));
/// let z: ctuple!(usize; 3) = CPair(0, CPair(1, C1(2)));
/// ```
#[macro_export]
macro_rules! ctuple {
    ($t: ty; $n: literal) => {
        <$crate::dim!($n) as $crate::traits::dim::TypeNum>::Array<$t>
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

impl<T: Zero + AsRef<[usize]>> TupleIter<T> {
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
        for &x in limit.as_ref() {
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

impl<T: Clone + AsMut<[usize]>> Iterator for TupleIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            None
        } else {
            let next = self.current.clone();
            self.finished =
                increment(self.current.as_mut(), &*self.limit.as_mut());
            Some(next)
        }
    }
}

/// The element-wise minimum of two tuples.
pub fn min<C: TypeNum>(
    x: &C::Array<Dim>,
    y: &C::Array<usize>,
) -> C::Array<usize> {
    let mut res = C::Array::<usize>::ZERO;

    for i in 0..C::VAL {
        res.as_mut()[i] = x.as_ref()[i].min(y.as_ref()[i]);
    }

    res
}

/// Returns whether `f` holds for all pairs of entries with the same index.
pub fn pairwise<
    C: TypeNum,
    T: ConstZero,
    U: ConstZero,
    F: Fn(&T, &U) -> bool,
>(
    x: &C::Array<T>,
    y: &C::Array<U>,
    f: F,
) -> bool {
    for i in 0..C::VAL {
        if !f(&x.as_ref()[i], &y.as_ref()[i]) {
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
        let _: ctuple!(usize; 3) = tuple!(1, 2, 3);
        let _: <dim!(3) as super::TypeNum>::Array<usize> = tuple!(1, 2, 3);
    }

    /// Checks that [`TupleIter`] outputs things in order.
    #[test]
    fn tuple_iter_1() {
        let res = [[0, 0], [1, 0], [0, 1], [1, 1], [0, 2], [1, 2]];
        for (i, x) in TupleIter::new(C2::new(2, 3)).enumerate() {
            assert_eq!(x.to_array(), res[i]);
        }
    }

    /// Checks that [`TupleIter`] handles the empty case accordingly.
    #[test]
    fn tuple_iter_2() {
        assert_eq!(TupleIter::new(tuple!(1, 2, 0)).next(), None)
    }
}
