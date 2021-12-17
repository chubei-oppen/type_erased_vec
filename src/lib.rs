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
#![feature(slice_ptr_get)]
#![feature(alloc_layout_extra)]

use std::{
    alloc::{handle_alloc_error, Allocator, Global, Layout},
    mem::{forget, ManuallyDrop},
    ops::{Deref, DerefMut},
    ptr::{copy_nonoverlapping, NonNull},
};

#[derive(Debug)]
/// A type erased [Vec].
pub struct TypeErasedVec<A: Allocator = Global> {
    /// The allocator, is only None after [TypeErasedVec::get_mut] and restored to `Some` after [VecMut] destruction.
    alloc: Option<A>,
    /// Pointer to memory, may be dangling.
    ptr: *mut u8,
    /// Length of the `Vec`.
    len: usize,
    /// Capacity of the `Vec`.
    cap: usize,
    /// Layout of `T`. Not the layout of `ptr`.
    layout: Layout,
}

impl<A: Allocator> TypeErasedVec<A> {
    /// Constructs a new, empty `TypeErasedVec`. See [Vec::new_in].
    pub fn new_in<T: Copy>(alloc: A) -> Self {
        Self::from_vec(Vec::<T, A>::new_in(alloc))
    }

    /// Constructs a new, empty `TypeErasedVec` with specified capacity. See [Vec::with_capacity_in].
    pub fn with_capacity_in<T: Copy>(capacity: usize, alloc: A) -> Self {
        Self::from_vec(Vec::<T, A>::with_capacity_in(capacity, alloc))
    }

    /// Erases the type of `vec`.
    pub fn from_vec<T: Copy>(vec: Vec<T, A>) -> Self {
        let (ptr, len, cap, alloc) = vec.into_raw_parts_with_alloc();
        TypeErasedVec {
            alloc: Some(alloc),
            ptr: ptr.cast(),
            len,
            cap,
            layout: Layout::new::<T>(),
        }
    }

    /// Converts to `Vec<T>`.
    ///
    /// # Safety
    ///
    /// `T` must be the same type used constructing this `TypeErasedVec`.
    ///
    /// Constructors include `new`, `new_in`, `with_capacity`, `with_capacity_in`, `from_vec`.
    pub unsafe fn into_vec<T: Copy>(mut self) -> Vec<T, A> {
        let vec = Vec::from_raw_parts_in(
            self.ptr.cast(),
            self.len,
            self.cap,
            self.alloc.take().unwrap(),
        );
        forget(self);
        vec
    }

    /// Gets a reference to \[T\].
    ///
    /// # Safety
    ///
    /// See [TypeErasedVec::into_vec].
    pub unsafe fn get<T: Copy>(&self) -> &[T] {
        std::slice::from_raw_parts(self.ptr.cast(), self.len)
    }

    /// Gets a smart pointer to `Vec<T>`.
    ///
    /// This is usually not want you want. Check [TypeErasedVec::get] instead.
    ///
    /// # Safety
    ///
    /// See [TypeErasedVec::into_vec].
    pub unsafe fn get_ref<T: Copy>(&self) -> VecRef<T, A> {
        VecRef::new(self)
    }

    /// Gets a smart pointer to `mut Vec<T>`.
    ///
    /// # Safety
    ///
    /// See [TypeErasedVec::into_vec].
    pub unsafe fn get_mut<T: Copy>(&mut self) -> VecMut<T, A> {
        VecMut::new(self)
    }

    /// Gets a reference to the underlying allocator.
    pub fn allocator(&self) -> &A {
        self.alloc.as_ref().unwrap()
    }

    fn memory(&self) -> Option<Layout> {
        if self.layout.size() > 0 && self.cap > 0 {
            Some(unsafe {
                Layout::from_size_align_unchecked(
                    self.layout.size() * self.cap,
                    self.layout.align(),
                )
            })
        } else {
            None
        }
    }
}

impl TypeErasedVec<Global> {
    /// Constructs a new, empty `TypeErasedVec`. See [Vec::new].
    pub fn new<T: Copy>() -> Self {
        Self::new_in::<T>(Global)
    }

    /// Constructs a new, empty `TypeErasedVec` with specified capacity. See [Vec::with_capacity].
    pub fn with_capacity<T: Copy>(capacity: usize) -> Self {
        Self::with_capacity_in::<T>(capacity, Global)
    }
}

impl<A: Allocator> Drop for TypeErasedVec<A> {
    fn drop(&mut self) {
        if let Some(layout) = self.memory() {
            unsafe {
                self.alloc
                    .as_ref()
                    .unwrap()
                    .deallocate(NonNull::new(self.ptr).unwrap(), layout)
            }
        };
    }
}

impl<A: Allocator + Clone> Clone for TypeErasedVec<A> {
    fn clone(&self) -> Self {
        let alloc = self.alloc.as_ref().unwrap().clone();
        let ptr = match self.memory() {
            Some(layout) => {
                let ptr = match alloc.allocate(layout) {
                    Ok(ptr) => ptr,
                    Err(_) => handle_alloc_error(layout),
                }
                .as_mut_ptr();
                unsafe {
                    copy_nonoverlapping(self.ptr, ptr, self.len * self.layout.size());
                }
                ptr
            }
            None => self.layout.dangling().as_ptr(),
        };
        TypeErasedVec {
            alloc: Some(alloc),
            ptr,
            len: self.len,
            cap: self.cap,
            layout: self.layout,
        }
    }
}

#[derive(Debug)]
/// `Deref`s to `Vec<T, Global>`.
///
/// The allocator will always be [Global]. It won't be used anyway.
pub struct VecRef<'a, T: Copy, A: Allocator> {
    raw: &'a TypeErasedVec<A>,
    vec: ManuallyDrop<Vec<T, Global>>,
}

impl<'a, T: Copy, A: Allocator> VecRef<'a, T, A> {
    fn new(raw: &'a TypeErasedVec<A>) -> Self {
        let vec =
            unsafe { ManuallyDrop::new(Vec::from_raw_parts(raw.ptr as *mut T, raw.len, raw.cap)) };
        VecRef { raw, vec }
    }
}

impl<'a, T: Copy, A: Allocator> Deref for VecRef<'a, T, A> {
    type Target = Vec<T, Global>;

    fn deref(&self) -> &Vec<T, Global> {
        &self.vec
    }
}

impl<'a, T: Copy, A: Allocator> Clone for VecRef<'a, T, A> {
    fn clone(&self) -> Self {
        Self::new(self.raw)
    }
}

#[derive(Debug)]
/// `DerefMut`s to `Vec<T, A>`.
pub struct VecMut<'a, T: Copy, A: Allocator> {
    raw: &'a mut TypeErasedVec<A>,
    vec: Option<ManuallyDrop<Vec<T, A>>>,
}

impl<'a, T: Copy, A: Allocator> VecMut<'a, T, A> {
    fn new(raw: &'a mut TypeErasedVec<A>) -> Self {
        let vec = Some(unsafe {
            ManuallyDrop::new(Vec::from_raw_parts_in(
                raw.ptr as *mut T,
                raw.len,
                raw.cap,
                raw.alloc.take().unwrap(),
            ))
        });
        VecMut { raw, vec }
    }
}

impl<'a, T: Copy, A: Allocator> Deref for VecMut<'a, T, A> {
    type Target = Vec<T, A>;

    fn deref(&self) -> &Vec<T, A> {
        self.vec.as_ref().unwrap()
    }
}

impl<'a, T: Copy, A: Allocator> DerefMut for VecMut<'a, T, A> {
    fn deref_mut(&mut self) -> &mut Vec<T, A> {
        self.vec.as_mut().unwrap()
    }
}

impl<'a, T: Copy, A: Allocator> Drop for VecMut<'a, T, A> {
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

    #[test]
    fn test_clone() {
        let vec = TypeErasedVec::from_vec((0..10).collect::<Vec<i32>>());
        let clone = vec.clone();
        unsafe {
            assert_eq!(vec.get::<i32>(), clone.get::<i32>());
        }
    }
}
