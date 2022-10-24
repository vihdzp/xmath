//! Implements matrices.
//!
//! ## Important note
//!
//! Due to compiler bug
//! [#37748](https://github.com/rust-lang/rust/issues/37748), you'll need to
//! provide type arguments to most functions explicitly.

use super::array::Array;
use super::poly::Poly;
use crate::traits::matrix::*;
use crate::{data::aliases::Transpose, traits::basic::*};

impl<C, V: List<C>, const N: usize> List<(usize, C)> for Array<V, N> {
    type Item = V::Item;

    fn is_valid_coeff(i: (usize, C)) -> bool {
        i.0 < N && V::is_valid_coeff(i.1)
    }

    fn coeff_ref(&self, i: (usize, C)) -> Option<&V::Item> {
        self.coeff_ref(i.0)?.coeff_ref(i.1)
    }

    fn coeff_set(&mut self, i: (usize, C), x: V::Item) {
        self[i.0].coeff_set(i.1, x);
    }
}

impl<'a, C, V: ListIter<C>, const N: usize> IntoIterator
    for Iter<'a, &'a Array<V, N>, (usize, C)>
where
    for<'b> Iter<'b, &'b V, C>: IntoIterator<Item = &'b V::Item>,
{
    type Item = &'a V::Item;
    type IntoIter = std::iter::Empty<&'a V::Item>;

    fn into_iter(self) -> Self::IntoIter {
        todo!()
    }
}

impl<C, V: Module<C>, const N: usize> ListIter<(usize, C)> for Array<V, N>
where
    V::Item: Ring,
    for<'a> Iter<'a, &'a V, C>: IntoIterator<Item = &'a V::Item>,
{
    fn map<F: FnMut(&V::Item) -> V::Item>(&self, mut f: F) -> Self {
        Self(std::array::from_fn(|i| self[i].map(|x| f(x))))
    }

    fn map_mut<F: FnMut(&mut V::Item)>(&mut self, mut f: F) {
        for i in 0..N {
            self[i].map_mut(|x| f(x));
        }
    }

    /*fn pairwise<F: FnMut(&Self::Item, &Self::Item) -> Self::Item>(
        &self,
        x: &Self,
        mut f: F,
    ) -> Self {
        self.pairwise(x, |u, v| u.pairwise(v, &mut f))
    }

    fn pairwise_mut<F: FnMut(&mut Self::Item, &Self::Item)>(
        &mut self,
        x: &Self,
        mut f: F,
    ) {
        self.pairwise_mut(x, |u, v| u.pairwise_mut(v, &mut f));
    }*/
}

impl<C, V: Module<C>, const N: usize> Module<(usize, C)> for Array<V, N>
where
    V::Item: Ring,
    for<'a> Iter<'a, &'a V, C>: IntoIterator<Item = &'a V::Item>,
{
}

impl<V: LinearModule, const N: usize> Matrix for Array<V, N>
where
    V::Item: Ring,
    for<'a> Iter<'a, &'a V, usize>: IntoIterator<Item = &'a V::Item>,
    for<'a> Iter<'a, &'a V, (usize, usize)>: IntoIterator<Item = &'a V::Item>,
{
    type HeightType = usize;
    type WidthType = V::DimType;

    const HEIGHT: Self::HeightType = N;
    const WIDTH: Self::WidthType = V::DIM;
    const DIR: Direction = Direction::Row;

    fn col_support(&self, _: usize) -> usize {
        N
    }

    fn row_support(&self, i: usize) -> usize {
        self.coeff_ref(i).map_or(0, V::support)
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

    fn collect_row<I: Iterator<Item = Self::Item>, J: Iterator<Item = I>>(
        mut iter: J,
    ) -> Self {
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

    fn collect_col<I: Iterator<Item = Self::Item>, J: Iterator<Item = I>>(
        iter: J,
    ) -> Self {
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

impl<T: Ring, const M: usize, const N: usize> MatrixMN<T, M, N> {
    /// Adds two statically sized matrices.
    pub fn madd(&self, _m: &Self) -> Self {
        todo!() // madd_gen::<Self, Self, Self>(self, m)
    }

    /// Multiplies two statically sized matrices.
    pub fn mmul<const K: usize>(
        &self,
        _m: &MatrixMN<T, N, K>,
    ) -> MatrixMN<T, M, K> {
        todo!()
        //       mmul_gen(self, m)
    }

    /// Transmutes a matrix as another matrix of the same size. The size check
    /// is performed at compile time.
    ///
    /// ## Panics
    ///
    /// Panics if both matrices don't have the same size.
    pub fn transmute<const K: usize, const L: usize>(
        self,
    ) -> MatrixMN<T, K, L> {
        if M * N == K * L {
            // Safety: matrices of the same size have the same layout.
            unsafe { crate::transmute_gen(self) }
        } else {
            panic!("{}", crate::DIM_MISMATCH)
        }
    }

    /// Transmutes a reference to a matrix as a refrence to another matrix of
    /// the same size. The size check is performed at compile time.
    ///
    /// ## Panics
    ///
    /// Panics if both matrices don't have the same size.
    pub fn transmute_ref<const K: usize, const L: usize>(
        &self,
    ) -> &MatrixMN<T, K, L> {
        if M * N == K * L {
            // Safety: we've performed the size check.
            unsafe { &*(self as *const Self).cast() }
        } else {
            panic!("{}", crate::DIM_MISMATCH)
        }
    }

    /// Transmutes a mutable reference to a matrix as a mutable refrence to
    /// another matrix of the same size. The size check is performed at
    /// compile time.
    ///
    /// ## Panics
    ///
    /// Panics if both matrices don't have the same size.
    pub fn transmute_mut<const K: usize, const L: usize>(
        &mut self,
    ) -> &mut MatrixMN<T, K, L> {
        if M * N == K * L {
            // Safety: we've performed the size check.
            unsafe { &mut *(self as *mut Self).cast() }
        } else {
            panic!("{}", crate::DIM_MISMATCH)
        }
    }
}

impl<C, V: List<C> + Zero> List<(usize, C)> for Poly<V> {
    type Item = V::Item;

    fn is_valid_coeff(i: (usize, C)) -> bool {
        V::is_valid_coeff(i.1)
    }

    fn coeff_ref(&self, i: (usize, C)) -> Option<&V::Item> {
        self.coeff_ref(i.0)?.coeff_ref(i.1)
    }

    fn coeff_set(&mut self, i: (usize, C), x: V::Item) {
        // Safety: we trim at the end.
        unsafe {
            self.as_slice_mut()[i.0].coeff_set(i.1, x);
            crate::data::poly::trim(self.as_vec_mut());
        }
    }
}

impl<C, V: Module<C>> ListIter<(usize, C)> for Poly<V>
where
    V::Item: Ring,
    for<'a> Iter<'a, &'a V, C>: IntoIterator<Item = &'a V::Item>,
    for<'a> Iter<'a, &'a Poly<V>, (usize, C)>: IntoIterator<Item = &'a V::Item>,
{
    fn map<F: FnMut(&V::Item) -> V::Item>(&self, mut f: F) -> Self {
        self.as_slice().iter().map(|v| v.map(|x| f(x))).collect()
    }

    fn map_mut<F: FnMut(&mut V::Item)>(&mut self, mut f: F) {
        // Safety: we trim at the end.
        unsafe {
            for v in self.as_slice_mut() {
                v.map_mut(&mut f);
            }

            crate::data::poly::trim(self.as_vec_mut());
        }
    }

    /*fn pairwise<F: FnMut(&Self::Item, &Self::Item) -> Self::Item>(
        &self,
        x: &Self,
        mut f: F,
    ) -> Self {
        <Poly<V> as ListIter<usize>>::pairwise(&self, x, |u: &V, v: &V| {
            u.pairwise(v, &mut f)
        })
    }

    fn pairwise_mut<F: FnMut(&mut Self::Item, &Self::Item)>(
        &mut self,
        x: &Self,
        f: F,
    ) {
        todo!()
    }*/
}

impl<C, V: Module<C>> Module<(usize, C)> for Poly<V>
where
    V::Item: Ring,
    for<'a> Iter<'a, &'a V, C>: IntoIterator<Item = &'a V::Item>,
    for<'a> Iter<'a, &'a Poly<V>, usize>: IntoIterator<Item = &'a V::Item>,
    for<'a> Iter<'a, &'a Poly<V>, (usize, C)>: IntoIterator<Item = &'a V::Item>,
{
}

impl<V: LinearModule> Matrix for Poly<V>
where
    V::Item: Ring,
    for<'a> Iter<'a, &'a V, usize>: IntoIterator<Item = &'a V::Item>,
    for<'a> Iter<'a, &'a Poly<V>, usize>: IntoIterator<Item = &'a V::Item>,
    for<'a> Iter<'a, &'a Poly<V>, (usize, usize)>:
        IntoIterator<Item = &'a V::Item>,
{
    type HeightType = Inf;
    type WidthType = V::DimType;

    const HEIGHT: Self::HeightType = Inf;
    const WIDTH: Self::WidthType = V::DIM;
    const DIR: Direction = Direction::Row;

    fn col_support(&self, _: usize) -> usize {
        self.len()
    }

    fn row_support(&self, i: usize) -> usize {
        self.coeff_ref(i).map_or(0, V::support)
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

    fn collect_row<I: Iterator<Item = Self::Item>, J: Iterator<Item = I>>(
        iter: J,
    ) -> Self {
        let mut res = Vec::new();

        for iter in iter {
            res.push(iter.collect());
        }

        res.into()
    }

    fn collect_col<I: Iterator<Item = Self::Item>, J: Iterator<Item = I>>(
        iter: J,
    ) -> Self {
        let mut res = Self::zero();

        for (j, iter) in iter.enumerate() {
            for (i, x) in iter.enumerate() {
                res.coeff_set((i, j), x);
            }
        }

        res
    }
}

/// An alias for a dynamically sized matrix.
pub type MatrixDyn<T> = Poly<Poly<T>>;

impl<T: Ring> MatrixDyn<T> {
    /// Adds two dynamically sized matrices.
    pub fn madd(&self, _m: &Self) -> Self {
        todo!() //madd_gen(self, m)
    }

    /// Multiplies two dynamically sized matrices.
    pub fn mmul<const K: usize>(&self, _m: &Self) -> Self {
        todo!() //  mmul_gen(self, m)
    }
}

/// Interprets a vector as a row vector.
pub type RowVec<T> = Array<T, 1>;

impl<T> RowVec<T> {
    /// Creates a new row vector.
    pub fn new(x: T) -> Self {
        Self([x])
    }

    /// Gets the inner vector.
    pub fn inner(self) -> T {
        crate::from_array(self.0)
    }
}

/// Interprets a vector as a column vector.
pub type ColVec<T> = Transpose<RowVec<T>>;

impl<T> ColVec<T> {
    /// Creates a new column vector.
    pub fn new(x: T) -> Self {
        Self(RowVec::new(x))
    }

    /// Gets the inner vector.
    pub fn inner(self) -> T {
        self.0.inner()
    }
}

#[cfg(test)]
mod tests {
    use std::num::Wrapping;

    use super::*;

    #[test]
    fn mul() {
        let m1: MatrixMN<Wu8, 2, 3> = Array([
            Array([Wrapping(1), Wrapping(2), Wrapping(3)]),
            Array([Wrapping(4), Wrapping(5), Wrapping(6)]),
        ]);

        let m2: MatrixMN<Wu8, 3, 2> = Array([
            Array([Wrapping(1), Wrapping(4)]),
            Array([Wrapping(2), Wrapping(5)]),
            Array([Wrapping(3), Wrapping(6)]),
        ]);

        let _m3: MatrixMN<Wu8, 2, 3> = Array([
            Array([Wrapping(1), Wrapping(2), Wrapping(3)]),
            Array([Wrapping(4), Wrapping(5), Wrapping(6)]),
        ]);

        println!("{:?}", m1.mmul(&m2));
    }
}
