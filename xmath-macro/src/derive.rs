use proc_macro::TokenStream;
use quote::quote;
use syn::{DeriveInput, Ident};

/// Contains the code for the [`crate::Transparent`] derive macro.
pub fn transparent(name: Ident) -> TokenStream {
    quote! {
        impl<T> xmath_traits::SliceLike for #name<T> {
            type Item = T;

            fn as_slice(&self) -> &[Self::Item] {
                xmath_traits::ArrayLike::as_ref(self)
            }

            fn as_mut_slice(&mut self) -> &mut [Self::Item] {
                xmath_traits::ArrayLike::as_mut(self)
            }
        }

        impl<T> xmath_traits::ArrayLike for #name<T> {
            const LEN: usize = 1;

            fn from_iter_mut<I: Iterator<Item = T>>(iter: &mut I) -> Self {
                xmath_traits::Transparent::from_iter_mut_def(iter)
            }
        }

        impl<T> xmath_traits::Transparent for #name<T> {}
    }
    .into()
}

/// Contains the code for the [`SliceIndex`](crate::SliceIndex) derive macro.
/// 
/// TODO: use `SliceIndex`
pub fn slice_index(input: DeriveInput) -> TokenStream {
    let name = input.ident;
    let (impl_gens, ty_gens, where_clause) = input.generics.split_for_impl();

    quote! {
        impl #impl_gens std::ops::Index<usize> for #name #ty_gens #where_clause {
            type Output = <Self as xmath_traits::SliceLike>::Item;

            fn index(&self, index: usize) -> &Self::Output {
                xmath_traits::SliceLike::index(self, index)
            }
        }
        
        impl #impl_gens std::ops::IndexMut<usize> for #name #ty_gens #where_clause {
            fn index_mut(&mut self, index: usize) -> &mut Self::Output {
                xmath_traits::SliceLike::index_mut(self, index)
            }
        }

        
        impl #impl_gens std::ops::Index<std::ops::Range<usize>> for #name #ty_gens #where_clause {
            type Output = [<Self as xmath_traits::SliceLike>::Item];

            fn index(&self, index: std::ops::Range<usize>) -> &Self::Output {
                xmath_traits::SliceLike::index_range(self, index)
            }
        }
        
        impl #impl_gens std::ops::IndexMut<std::ops::Range<usize>> for #name #ty_gens #where_clause {
            fn index_mut(&mut self, index: std::ops::Range<usize>) -> &mut Self::Output {
                xmath_traits::SliceLike::index_range_mut(self, index)
            }
        }
    }
    .into()
}

/// Contains the code for the [`ArrayFromIter`](crate::ArrayFromIter) derive 
/// macro.
pub fn array_from_iter(input: DeriveInput) -> TokenStream {
    let name = input.ident;
    let (impl_gens, ty_gens, where_clause) = input.generics.split_for_impl();

    quote! {
        impl #impl_gens FromIterator<<Self as xmath_traits::SliceLike>::Item> for #name #ty_gens #where_clause {
            fn from_iter<I: IntoIterator<Item = <Self as xmath_traits::SliceLike>::Item>>(iter: I) -> Self {
                xmath_traits::ArrayLike::from_iter(iter)
            }
        }
    }
    .into()
}
