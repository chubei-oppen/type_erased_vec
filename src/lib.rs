//! This crate provides a single struct [TypeErasedVec], which (as its name says) is a type erased [Vec].
//!
//! When you know what its type is, you can get a slice or [Vec] back using [TypeErasedVec::get] or [TypeErasedVec::get_mut].
//!
//! # Motivation
//!
//! When communicating with a world outside Rust (GPU for example), it often wants a raw buffer and some kind of type descriptor.
//!
//! There were two options for expressing this in Rust:
//!
//! - `Vec<u8>` + type descriptor.
//! - `Vec<T>`.
//!
//! The first option is not attractive because `Vec<u8>` cannot be safely used as `Vec<T>`, hence we lose the ability of modifying the buffer.
//!
//! The second option makes all types holding that buffer generic over `T`, which is not feasible when `T` must be determined at runtime.
//! For example, buffers can be loaded from a 3D model file on disk, where the file contains type information to be passed to the 3D renderer.
//!
//! # Leaking
//!
//! `TypeErasedVec` (and its companion struct [VecMut]), as other RAII types, relies on the destructor being called to correctly release resources.
//! Failing to do so can cause memory leak, for example, through the use of [std::mem::forget].
//!
//! What's more, the content of `TypeErasedVec` is only valid after `VecMut`'s destructor is called.
//!
//! A `TypeErasedVec` is said to be in `leaked` state if the destructor of the returned `VecMut` of a previous call to `get_mut` didn't run.
//!
//! Calling any method except for [TypeErasedVec::is_leaked] on a leaked `TypeErasedVec` results in panic.
//!
//! # Example
//!
//! ```
//! use type_erased_vec::TypeErasedVec;
//!
//! let mut vec = TypeErasedVec::new::<i32>();
//!
//! let mut vec_mut = unsafe { vec.get_mut() };
//! for i in 0..10 {
//!     vec_mut.push(i);
//! }
//!
//! assert_eq!(*vec_mut, (0..10).collect::<Vec<_>>());
//! ```

#![deny(
    missing_docs,
    missing_debug_implementations,
    trivial_casts,
    trivial_numeric_casts,
    unused_import_braces,
    unused_qualifications
)]
#![feature(allocator_api)]

use std::{
    alloc::{Allocator, Global},
    mem::{forget, ManuallyDrop},
    ops::{Deref, DerefMut},
};

mod raw {
    use super::{Allocator, Global, ManuallyDrop};

    #[derive(Debug)]
    /// The raw parts of a `Vec`.
    ///
    /// This struct is leaking. It doesn't call the destructor of the elements or deallocate memory.
    pub struct RawVec<A: Allocator> {
        ptr: *mut u8,
        len: usize,
        cap: usize,
        alloc: A,
    }

    impl<A: Allocator> RawVec<A> {
        pub fn from_vec<T>(vec: Vec<T, A>) -> Self {
            let (ptr, len, cap, alloc) = vec.into_raw_parts_with_alloc();
            RawVec {
                ptr: ptr.cast(),
                len,
                cap,
                alloc,
            }
        }

        pub fn allocator(&self) -> &A {
            &self.alloc
        }

        /// # Safety
        ///
        /// `T` must be the same as in `from_vec`.
        pub unsafe fn into_vec<T>(self) -> Vec<T, A> {
            Vec::from_raw_parts_in(self.ptr.cast(), self.len, self.cap, self.alloc)
        }

        /// # Safety
        ///
        /// `T` must be the same as in `from_vec`.
        pub unsafe fn as_slice<T>(&self) -> &[T] {
            std::slice::from_raw_parts(self.ptr.cast(), self.len)
        }
    }

    impl RawVec<Global> {
        /// # Safety
        /// - `T` must be the same as in `from_vec`.
        /// - Returned value must not outlive underlying memory.
        /// - Multiple return values of this method must not be dropped more than once.
        pub unsafe fn as_manually_drop_vec<T>(&self) -> ManuallyDrop<Vec<T, Global>> {
            ManuallyDrop::new(Vec::from_raw_parts_in(
                self.ptr.cast(),
                self.len,
                self.cap,
                self.alloc,
            ))
        }
    }

    /// # Safety
    ///
    /// `T` must be the same as in `from_vec`.
    pub unsafe fn drop_raw_vec<T, A: Allocator>(raw: RawVec<A>) {
        drop(raw.into_vec::<T>());
    }
}

use raw::{drop_raw_vec, RawVec};

#[derive(Debug)]
/// A type erased [Vec].
pub struct TypeErasedVec<A: Allocator = Global> {
    /// The raw form of the `Vec`. It's only None after [TypeErasedVec::get_mut] and restored to `Some` after [VecMut] destruction.
    raw: Option<RawVec<A>>,
    drop: unsafe fn(RawVec<A>),
}

impl<A: Allocator> TypeErasedVec<A> {
    /// Constructs a new, empty `TypeErasedVec`. See [Vec::new_in].
    pub fn new_in<T>(alloc: A) -> Self {
        Self::from_vec(Vec::<T, A>::new_in(alloc))
    }

    /// Constructs a new, empty `TypeErasedVec` with specified capacity. See [Vec::with_capacity_in].
    pub fn with_capacity_in<T>(capacity: usize, alloc: A) -> Self {
        Self::from_vec(Vec::<T, A>::with_capacity_in(capacity, alloc))
    }

    /// Erases the type of `vec`.
    pub fn from_vec<T>(vec: Vec<T, A>) -> Self {
        TypeErasedVec {
            raw: Some(RawVec::from_vec(vec)),
            drop: drop_raw_vec::<T, A>,
        }
    }

    /// Returns if `self` is leaked.
    pub fn is_leaked(&self) -> bool {
        self.raw.is_none()
    }

    /// Converts to `Vec<T>`.
    ///
    /// # Safety
    ///
    /// `T` must be the same type used constructing this `TypeErasedVec`.
    ///
    /// Constructors include `new`, `new_in`, `with_capacity`, `with_capacity_in`, `from_vec`.
    ///
    /// # Panics
    ///
    /// Panics if `self` is leaked.
    pub unsafe fn into_vec<T>(mut self) -> Vec<T, A> {
        let vec = self.raw.take().unwrap().into_vec();
        forget(self);
        vec
    }

    /// Gets a reference to \[T\].
    ///
    /// # Safety
    ///
    /// See [TypeErasedVec::into_vec].
    ///
    /// # Panics
    ///
    /// Panics if `self` is leaked.
    pub unsafe fn get<T>(&self) -> &[T] {
        self.raw.as_ref().unwrap().as_slice()
    }

    /// Gets a smart pointer to `mut Vec<T>`.
    ///
    /// # Safety
    ///
    /// See [TypeErasedVec::into_vec].
    ///
    /// # Panics
    ///
    /// Panics if `self` is leaked.
    pub unsafe fn get_mut<T>(&mut self) -> VecMut<T, A> {
        VecMut::new(self)
    }

    /// Gets a reference to the underlying allocator.
    ///
    /// # Panics
    ///
    /// Panics if `self` is leaked.
    pub fn allocator(&self) -> &A {
        self.raw.as_ref().unwrap().allocator()
    }
}

impl TypeErasedVec<Global> {
    /// Constructs a new, empty `TypeErasedVec`. See [Vec::new].
    pub fn new<T>() -> Self {
        Self::new_in::<T>(Global)
    }

    /// Constructs a new, empty `TypeErasedVec` with specified capacity. See [Vec::with_capacity].
    pub fn with_capacity<T>(capacity: usize) -> Self {
        Self::with_capacity_in::<T>(capacity, Global)
    }

    /// Gets a smart pointer to `Vec<T>`.
    ///
    /// This is usually not want you want. Check [TypeErasedVec::get] instead.
    ///
    /// This method is only implemented for `TypeErasedVec<Global>` because we can't get a `Vec` back without giving it a allocator.
    ///
    /// # Safety
    ///
    /// See [TypeErasedVec::into_vec].
    ///
    /// # Panics
    ///
    /// Panics if `self` is leaked.
    pub unsafe fn get_ref<T>(&self) -> VecRef<T> {
        VecRef::new(self)
    }
}

impl<A: Allocator> Drop for TypeErasedVec<A> {
    fn drop(&mut self) {
        if let Some(raw) = self.raw.take() {
            let drop = self.drop;
            unsafe {
                drop(raw);
            }
        }
    }
}

#[derive(Debug)]
/// `Deref`s to `Vec<T, Global>`.
pub struct VecRef<'a, T> {
    raw: &'a TypeErasedVec<Global>,
    vec: ManuallyDrop<Vec<T, Global>>,
}

impl<'a, T> VecRef<'a, T> {
    /// # Safety
    ///
    /// `T` must be what `raw` was constructred with.
    unsafe fn new(raw: &'a TypeErasedVec<Global>) -> Self {
        let vec = raw.raw.as_ref().unwrap().as_manually_drop_vec();
        VecRef { raw, vec }
    }
}

impl<'a, T> Deref for VecRef<'a, T> {
    type Target = Vec<T, Global>;

    fn deref(&self) -> &Vec<T, Global> {
        &self.vec
    }
}

impl<'a, T> Clone for VecRef<'a, T> {
    fn clone(&self) -> Self {
        unsafe { Self::new(self.raw) }
    }
}

#[derive(Debug)]
/// `DerefMut`s to `Vec<T, A>`.
pub struct VecMut<'a, T, A: Allocator> {
    raw: &'a mut TypeErasedVec<A>,
    vec: Option<ManuallyDrop<Vec<T, A>>>,
}

impl<'a, T, A: Allocator> VecMut<'a, T, A> {
    /// # Safety
    ///
    /// `T` must be what `raw` was constructred with.
    unsafe fn new(raw: &'a mut TypeErasedVec<A>) -> Self {
        let vec = Some(ManuallyDrop::new(raw.raw.take().unwrap().into_vec()));
        VecMut { raw, vec }
    }
}

impl<'a, T, A: Allocator> Deref for VecMut<'a, T, A> {
    type Target = Vec<T, A>;

    fn deref(&self) -> &Vec<T, A> {
        self.vec.as_ref().unwrap()
    }
}

impl<'a, T, A: Allocator> DerefMut for VecMut<'a, T, A> {
    fn deref_mut(&mut self) -> &mut Vec<T, A> {
        self.vec.as_mut().unwrap()
    }
}

impl<'a, T, A: Allocator> Drop for VecMut<'a, T, A> {
    fn drop(&mut self) {
        let vec = self.vec.take().unwrap();
        let vec = ManuallyDrop::into_inner(vec);
        *self.raw = TypeErasedVec::from_vec(vec);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let vec = TypeErasedVec::new::<i32>();
        let vec_ref = unsafe { vec.get_ref::<i32>() };
        assert_eq!(vec_ref.len(), 0);
        assert_eq!(vec_ref.capacity(), 0);
    }

    #[test]
    fn test_with_capacity() {
        let vec = TypeErasedVec::with_capacity::<i32>(42);
        assert_eq!(unsafe { vec.get_ref::<i32>().capacity() }, 42);
    }

    #[test]
    fn test_from_vec() {
        let origin: Vec<i32> = (0..10).collect();
        let vec = TypeErasedVec::from_vec(origin.clone());
        let vec_ref = unsafe { vec.get::<i32>() };
        assert_eq!(origin, *vec_ref);
    }

    #[test]
    fn test_into_vec() {
        let origin: Vec<i32> = (0..10).collect();
        let vec = TypeErasedVec::from_vec(origin.clone());
        assert_eq!(unsafe { vec.into_vec::<i32>() }, origin);
    }

    #[test]
    fn test_get() {
        let vec = if true {
            TypeErasedVec::new::<i32>()
        } else {
            TypeErasedVec::new::<f64>()
        };
        let _: &[u8] = if true {
            bytemuck::cast_slice(unsafe { vec.get::<i32>() })
        } else {
            bytemuck::cast_slice(unsafe { vec.get::<f64>() })
        };
    }

    #[test]
    fn test_get_mut() {
        let mut vec = TypeErasedVec::new::<i32>();
        let mut vec_mut = unsafe { vec.get_mut::<i32>() };
        for i in 0..10 {
            vec_mut.push(i);
        }
        drop(vec_mut);
        let vec_ref = unsafe { vec.get::<i32>() };
        assert_eq!((0..10).collect::<Vec<_>>(), *vec_ref);
    }
}
