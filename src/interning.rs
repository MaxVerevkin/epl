use std::collections::HashSet;
use std::hash::Hash;
use std::sync::Mutex;

use bumpalo::Bump;

pub struct Interner<'ctx, T: ?Sized>(&'ctx Bump, Mutex<HashSet<&'ctx T>>);

#[derive(Debug)]
pub struct Interned<'ctx, T: ?Sized>(&'ctx T);

impl<'ctx, T: ?Sized> Interner<'ctx, T> {
    pub fn new(arena: &'ctx Bump) -> Self {
        Self(arena, Mutex::new(HashSet::new()))
    }
}

impl<'ctx, T: Hash + Eq> Interner<'ctx, T> {
    pub fn intern<'this>(&'this self, value: T) -> Interned<'ctx, T> {
        let mut guard = self.1.lock().unwrap();
        Interned(if let Some(interned) = guard.get(&value).copied() {
            interned
        } else {
            let interned = self.0.alloc(value) as &'ctx T;
            guard.insert(interned);
            interned
        })
    }
}

impl<'ctx, T: Hash + Eq + Copy> Interner<'ctx, [T]> {
    pub fn intern_slice<'this>(&'this self, value: &[T]) -> Interned<'ctx, [T]> {
        let mut guard = self.1.lock().unwrap();
        Interned(if let Some(interned) = guard.get(value).copied() {
            interned
        } else {
            let interned = self.0.alloc_slice_copy(value) as &'ctx [T];
            guard.insert(interned);
            interned
        })
    }
}

impl<'ctx, T: ?Sized> Interned<'ctx, T> {
    pub fn new_unchecked(interned: &'ctx T) -> Self {
        Self(interned)
    }

    pub fn get(self) -> &'ctx T {
        self.0
    }
}

impl<T: ?Sized> Clone for Interned<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: ?Sized> Copy for Interned<'_, T> {}

impl<T: ?Sized> PartialEq for Interned<'_, T> {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::addr_eq(self.0, other.0)
    }
}

impl<T: ?Sized> Eq for Interned<'_, T> {}

impl<T: ?Sized> PartialOrd for Interned<'_, T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: ?Sized> Ord for Interned<'_, T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.0 as *const T).addr().cmp(&(other.0 as *const T).addr())
    }
}

impl<T: ?Sized> Hash for Interned<'_, T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::ptr::hash(self.0 as *const T, state);
    }
}

impl<'ctx, T: ?Sized> AsRef<Interner<'ctx, T>> for Interner<'ctx, T> {
    fn as_ref(&self) -> &Interner<'ctx, T> {
        self
    }
}
