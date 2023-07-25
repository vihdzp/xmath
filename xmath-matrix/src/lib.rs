//! Contains the definitions for basic matrix types, including [`Array`] and
//! [`Poly`].

pub mod algs;
mod array;
mod empty;
mod poly;
pub mod traits;
mod transpose;

pub use array::*;
pub use empty::*;
pub use poly::*;
pub use transpose::*;
