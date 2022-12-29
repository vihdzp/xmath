//! This auxiliary crate contains only the definitions of the type-level integer
//! types [`U1`] and [`Succ`]. It is required so that these can be used by the
//! procedural macro `dim!`.

/// The type-level integer `1`.
pub struct U1;

/// The type-level integer `D + 1`.
pub struct Succ<D>(pub D);

/// Transmutes a reference `&T` into a reference `&U`.
///
/// ## Safety
///
/// Both types must have the same layout.
pub unsafe fn transmute_ref<T: ?Sized, U>(x: &T) -> &U {
    &*(x as *const T).cast()
}

/// Transmutes a mutable reference `&mut T` into a mutable reference `&mut U`.
///
/// ## Safety
///
/// Both types must have the same layout.
pub unsafe fn transmute_mut<T: ?Sized, U>(x: &mut T) -> &mut U {
    &mut *(x as *mut T).cast()
}

/// A workaround for transmuting a generic type into another.
///
/// Borrowed from https://github.com/rust-lang/rust/issues/61956.
///
/// ## Safety
///
/// All the same safety considerations for [`std::mem::transmute`] still apply.
pub unsafe fn transmute_gen<T, U>(x: T) -> U {
    (*transmute_ref::<_, std::mem::MaybeUninit<U>>(&std::mem::MaybeUninit::new(x)))
        .assume_init_read()
}

/// Transmutes `[T; 1]` into `T`.
pub fn from_array<T>(x: [T; 1]) -> T {
    // Safety: both types have the same layout.
    unsafe { crate::transmute_gen(x) }
}
