#[macro_export]
macro_rules! make_entity_id {
    ($name:ident, $fmt:literal) => {
        #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(::std::num::NonZeroU64);

        impl $name {
            #[allow(unused)]
            pub const DUMMY: Self = Self(::std::num::NonZeroU64::new(u64::MAX).unwrap());

            pub fn new() -> Self {
                use std::sync::atomic;
                static NEXT: atomic::AtomicU64 = atomic::AtomicU64::new(1);
                let id = NEXT.fetch_add(1, atomic::Ordering::SeqCst);
                Self(::std::num::NonZeroU64::new(id).unwrap())
            }
        }

        impl ::std::fmt::Debug for $name {
            fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
                write!(f, $fmt, self.0.get())
            }
        }
    };
}
