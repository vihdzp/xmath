use proc_macro::TokenStream;
use quote::quote;
use syn::{DeriveInput, Ident};

/// Contains the code for the [`crate::Transparent`] derive macro.
pub fn transparent(name: Ident) -> TokenStream {
    quote! {
        impl<T> xmath::traits::SliceLike for #name<T> {
            type Item = T;

            fn as_slice(&self) -> &[Self::Item] {
                xmath::traits::ArrayLike::as_ref(self)
            }

            fn as_mut_slice(&mut self) -> &mut [Self::Item] {
                xmath::traits::ArrayLike::as_mut(self)
            }
        }

        impl<T> xmath::traits::ArrayLike for #name<T> {
            const LEN: usize = 1;

            fn from_iter_mut<I: Iterator<Item = T>>(iter: &mut I) -> Self {
                xmath::traits::Transparent::from_iter_mut_def(iter)
            }
        }

        impl<T> xmath::traits::Transparent for #name<T> {}
    }
    .into()
}

/// Contains the code for the [`SliceIndex`](crate::SliceIndex) derive macro.
pub fn slice_index(input: DeriveInput) -> TokenStream {
    let name = input.ident;
    let (impl_gens, ty_gens, where_clause) = input.generics.split_for_impl();

    quote! {
        impl #impl_gens std::ops::Index<usize> for #name #ty_gens #where_clause {
            type Output = <Self as xmath::traits::SliceLike>::Item;

            fn index(&self, index: usize) -> &Self::Output {
                xmath::traits::SliceLike::index(self, index)
            }
        }
        
        impl #impl_gens std::ops::IndexMut<usize> for #name #ty_gens #where_clause {
            fn index_mut(&mut self, index: usize) -> &mut Self::Output {
                xmath::traits::SliceLike::index_mut(self, index)
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
        impl #impl_gens FromIterator<<Self as xmath::traits::SliceLike>::Item> for #name #ty_gens #where_clause {
            fn from_iter<I: IntoIterator<Item = <Self as xmath::traits::SliceLike>::Item>>(iter: I) -> Self {
                xmath::traits::ArrayLike::from_iter(iter)
            }
        }
    }
    .into()
}
