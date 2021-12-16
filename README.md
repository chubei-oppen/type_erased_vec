# type_erased_vec

This crate provides a single struct [TypeErasedVec], which (as its name says) is a type erased [Vec].

When you know what its type is, you can get a slice or [Vec] back using [TypeErasedVec::get] or [TypeErasedVec::get_mut].

# Motivation

When communicating with a world outside Rust (GPU for example), it often wants a raw buffer and some kind of type descriptor.

There were two options for expressing this in Rust:

- `Vec<u8>` + type descriptor.
- `Vec<T>`.

The first option is not attractive because `Vec<u8>` cannot be safely used as `Vec<T>`, hence we lose the ability of modifying the buffer.

The second option makes all types holding that buffer generic over `T`, which is not feasible when `T` must be determined at runtime.
For example, buffers can be loaded from a 3D model file on disk, where the file contains type information to be passed to the 3D renderer.

# Example

```rust
use type_erased_vec::TypeErasedVec;

let mut vec = TypeErasedVec::new::<i32>();

let mut vec_mut = unsafe { vec.get_mut() };
for i in 0..10 {
    vec_mut.push(i);
}

assert_eq!(*vec_mut, (0..10).collect::<Vec<_>>());
```

[TypeErasedVec]: https://docs.rs/type_erased_vec/latest/type_erased_vec/struct.TypeErasedVec.html
[TypeErasedVec::get]: https://docs.rs/type_erased_vec/latest/type_erased_vec/struct.TypeErasedVec.html#method.get
[TypeErasedVec::get_mut]: https://docs.rs/type_erased_vec/latest/type_erased_vec/struct.TypeErasedVec.html#method.get_mut
[Vec]: https://doc.rust-lang.org/nightly/alloc/vec/struct.Vec.html
