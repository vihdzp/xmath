//! Defines various procedural macros for both internal and external use in
//! `xmath`.

mod derive;
mod dim;

use proc_macro::TokenStream;
use syn::DeriveInput;

/// Please call this as `xmath::dim`.
#[proc_macro]
pub fn dim(input: TokenStream) -> TokenStream {
    dim::dim_aux(input.to_string().parse().expect("invalid integer"))
}

fn parse_derive(input: TokenStream) -> DeriveInput {
    syn::parse(input).unwrap()
}

/// **This macro only works for structs like the following:**
/// ```rs
/// #[repr(transparent)]
/// struct Foo<T>(T);
/// ```
/// It defines the `SliceLike`, `ArrayLike`, `FromIterator`, and `Transparent`
/// traits in the obvious way, by treating `Foo<T>` as an array with a single
/// element of type `T`.
#[proc_macro_derive(Transparent)]
pub fn transparent(input: TokenStream) -> TokenStream {
    derive::transparent(parse_derive(input).ident)
}

/// Derives [`Index`](std::ops::Index) and [`IndexMut`](std::ops::IndexMut) from
/// `SliceLike`.
#[proc_macro_derive(SliceIndex)]
pub fn slice_index(input: TokenStream) -> TokenStream {
    derive::slice_index(parse_derive(input))
}

/// Derives [`FromIterator`] from `ArrayLike`.
#[proc_macro_derive(ArrayFromIter)]
pub fn array_from_iter(input: TokenStream) -> TokenStream {
    derive::array_from_iter(parse_derive(input))
}
