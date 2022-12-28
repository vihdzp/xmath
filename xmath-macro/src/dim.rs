//! Implements [`dim_aux`], which does all the heavy lifting for the
//! [`dim!`](crate::dim!) procedural macro.

use proc_macro::*;

/// The core crate name.
const XMATH_CORE: &str = "xmath_core";

/// The name for the 1 type.
const U1: &str = "U1";

/// The name for the successor type.
const SUCC: &str = "Succ";

/// An auxiliary type that simplifies building the necessary identifiers.
#[derive(Default)]
struct TokenTreeBuilder(Vec<TokenTree>);

impl From<TokenTreeBuilder> for TokenStream {
    fn from(value: TokenTreeBuilder) -> Self {
        value.0.into_iter().collect()
    }
}

impl TokenTreeBuilder {
    /// Creates a new token tree builder.
    fn new() -> Self {
        Self::default()
    }

    /// Pushes an identifier with the given `string`, expanded at the call site.
    fn push_ident(&mut self, string: &str) {
        self.0
            .push(TokenTree::Ident(Ident::new(string, Span::call_site())));
    }

    /// Pushes two colons `::`.
    fn push_colons(&mut self) {
        self.0
            .push(TokenTree::Punct(Punct::new(':', Spacing::Joint)));
        self.0
            .push(TokenTree::Punct(Punct::new(':', Spacing::Alone)));
    }

    /// Pushes `xmath_core::`, then calls `push_ident` with the given `string`.
    fn push_core_ident(&mut self, string: &str) {
        self.push_ident(XMATH_CORE);
        self.push_colons();
        self.push_ident(string);
    }

    /// Pushes `<`.
    fn push_lt(&mut self) {
        self.0
            .push(TokenTree::Punct(Punct::new('<', Spacing::Alone)));
    }

    /// Pushes `>` with the given spacing.
    fn push_gt(&mut self, spacing: Spacing) {
        self.0.push(TokenTree::Punct(Punct::new('>', spacing)));
    }

    /// Pushes the `>` character `n` times.
    fn push_gts(&mut self, n: u8) {
        if n != 0 {
            for _ in 1..n {
                self.push_gt(Spacing::Joint);
            }
            self.push_gt(Spacing::Alone);
        }
    }
}

/// Returns the type-level integer associated to `n != 0`.
pub fn dim_aux(n: u8) -> TokenStream {
    let mut builder = TokenTreeBuilder::new();

    // Succ<Succ<...<
    for _ in 1..n {
        builder.push_core_ident(SUCC);
        builder.push_lt();
    }

    // U1>>...>
    builder.push_core_ident(U1);
    builder.push_gts(n - 1);
    builder.into()
}
