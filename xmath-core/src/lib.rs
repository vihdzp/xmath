//! This auxiliary crate contains only the definitions of the type-level integer
//! types [`U1`] and [`Succ`]. It is required so that these can be used by the 
//! procedural macro `dim!`.

/// The type-level integer `1`.
pub struct U1;

/// The type-level integer `D + 1`.
pub struct Succ<D>(pub D);
