@_wraps(np.array, lax_description=_ARRAY_DOC)
def array(object, dtype=None, copy=True, order="K", ndmin=0):
    if order is not None and order != "K":
        raise NotImplementedError("Only implemented for order='K'")

    # check if the given dtype is compatible with JAX
    lax_internal._check_user_dtype_supported(dtype, "array")

    # Here we make a judgment call: we only return a weakly-typed array when the
    # input object itself is weakly typed. That ensures asarray(x) is a no-op
    # whenever x is weak, but avoids introducing weak types with something like
    # array([1, 2, 3])
    weak_type = dtype is None and dtypes.is_weakly_typed(object)

    # For Python scalar literals, call coerce_to_array to catch any overflow
    # errors. We don't use dtypes.is_python_scalar because we don't want this
    # triggering for traced values. We do this here because it matters whether or
    # not dtype is None. We don't assign the result because we want the raw object
    # to be used for type inference below.
    if isinstance(object, (bool, int, float, complex)):
        _ = dtypes.coerce_to_array(object, dtype)

    leaves = tree_leaves(object)
    if dtype is None:
        # Use lattice_result_type rather than result_type to avoid canonicalization.
        # Otherwise, weakly-typed inputs would have their dtypes canonicalized.
        try:
            dtype = dtypes._lattice_result_type(*leaves)[0] if leaves else dtypes.float_
        except TypeError:
            # This happens if, e.g. one of the entries is a memoryview object.
            # This is rare, so we only handle it if the normal path fails.
            leaves = [_convert_to_array_if_dtype_fails(leaf) for leaf in leaves]
            dtype = dtypes._lattice_result_type(*leaves)[0]

    if not weak_type:
        dtype = dtypes.canonicalize_dtype(dtype)

    # We can't use the ndarray class because we need to handle internal buffers
    # (See https://github.com/google/jax/issues/8950)
    ndarray_types = (device_array.DeviceArray, core.Tracer)

    if not _any(isinstance(leaf, ndarray_types) for leaf in leaves):
        # TODO(jakevdp): falling back to numpy here fails to overflow for lists
        # containing large integers; see discussion in
        # https://github.com/google/jax/pull/6047. More correct would be to call
        # coerce_to_array on each leaf, but this may have performance implications.
        out = np.array(object, dtype=dtype, ndmin=ndmin, copy=False)
    elif isinstance(object, ndarray_types):
        assert object.aval is not None
        out = _array_copy(object) if copy else object
    elif isinstance(object, (list, tuple)):
        if object:
            out = stack([asarray(elt, dtype=dtype) for elt in object])
        else:
            out = np.array([], dtype=dtype)
    else:
        try:
            view = memoryview(object)
        except TypeError:
            pass  # `object` does not support the buffer interface.
        else:
            return array(np.asarray(view), dtype, copy, ndmin=ndmin)

        raise TypeError(f"Unexpected input type for array: {type(object)}")

    out = lax_internal._convert_element_type(out, dtype, weak_type=weak_type)
    if ndmin > ndim(out):
        out = lax.expand_dims(out, range(ndmin - ndim(out)))
    return out
