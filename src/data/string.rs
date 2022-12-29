use xmath_traits::*;

/// A string in an alphabet, whose characters are given by the type `T`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct String<T>(pub Vec<T>);

impl<T> From<Vec<T>> for String<T> {
    fn from(value: Vec<T>) -> Self {
        Self(value)
    }
}

impl<T> String<T> {
    pub fn new(value: Vec<T>) -> Self {
        value.into()
    }
}

impl<T> SliceLike for String<T> {
    type Item = T;

    fn as_slice(&self) -> &[Self::Item] {
        self.0.as_slice()
    }

    fn as_mut_slice(&mut self) -> &mut [Self::Item] {
        self.0.as_mut_slice()
    }
}

impl<T: PartialEq> Zero for String<T> {
    fn zero() -> Self {
        Self::new(Vec::new())
    }
}

impl<T: Clone> Add for String<T> {
    fn add(&self, rhs: &Self) -> Self {
        let mut res = self.0.clone();
        res.extend_from_slice(rhs.as_slice());
        res.into()
    }

    fn add_mut(&mut self, rhs: &Self) {
        self.0.extend_from_slice(rhs.as_slice());
    }
}

impl<T: Eq + Clone> AddMonoid for String<T> {}
