# WHERE EVERYTHING LIVES

Grain's pipeline lives entirely on GPU: from `RandomAccessDataSource` through `MapDataset`, batching, shuffling, and `to_iter_dataset`, all runs in Python on the CPU. Grain never touches the GPU. The thread pool referenced by ReadOptions(num_threads=...) is a CPU thread pool for hiding I/O latency during disk reads. Prefetched batches are also held in CPU RAM.

`InMemoryHDF5Source` loads the entire dataset into CPU RAM once at construction via `f['data'][:]`, which produces a NumPy array (h5py always returns NumPy). `ToyDataSource` generates its data using JAX but under `jax.default_device(cpu)`, so the resulting `jax.Arrays` are backed by CPU memory.

The transfer from CPU RAM to GPU happens at the point your training step consumes the batch.

If your `train_step` is decorated with `@nnx.jit` (or `@jax.jit`), JAX will automatically transfer any NumPy arrays passed as inputs to the default device (GPU/TPU) when the function is first called. Using NumPy arrays directly as inputs to JIT-compiled functions results in a CPU-to-GPU transfer on every call.

Flax NNX model parameters live on GPU from the moment the model is constructed, because `jnp.*` operations default to the most capable device available. Once an array is placed on a device, JAX tries to keep computations involving that array on the same device, minimizing data transfers. APXML So the forward pass, gradients, and optimizer state all stay on GPU throughout training.

## MULTI-GPU SHARDING (IF NEEDED WHEN USING POLIMI CLUSTER)
By default with a single GPU, no sharding occurs — the full batch lands on gpu:0. For multi-GPU data parallelism you explicitly create a mesh and a sharding, then call jax.device_put(batch, sharding) before passing the batch to a jitted train step. The three main ways to create device-spanning arrays from external data are: putting the full array on all processes via jax.device_put(), loading only the local shard per process via jax.make_array_from_process_local_data(), or assembling pre-placed per-device arrays via jax.make_array_from_single_device_arrays() — the latter two are most common since materialising the full global data in every process is often too expensive. JAX Documentation Grain's grain.experimental.device_put helper wraps this pattern with a CPU staging buffer and a device buffer to pipeline transfers.

## WHY TO CAST JAX.ARRAYS TO NP.NDARRAYS IN TOYDAATSOURCE CONSTRUCTOR

In your current get_pipeline, you set num_threads=0 and prefetch_size=0 for in-memory sources, which means everything runs in the main Python process, single-threaded. In that case, the jax.Arrays stored in ToyDataSource are accessed directly, Grain slices them with __getitem__, and the resulting objects flow through to the iterator. JAX arrays support indexing and are valid pytrees, so Grain's batching step (which uses dm-tree to stack leaves) can handle them. It works.

When JAX arrays are created in a subprocess (e.g. via multiprocessing), they lose their original device placement upon pickling and unpickling: a CPU array serialised in a child process is deserialised onto the default device (usually the GPU) in the main process. GitHub Grain's multiprocessing support (grain.MultiprocessingOptions) works by spawning worker processes that call __getitem__ and send the results back via pickle. If your stored arrays are jax.Arrays, every single sample retrieved by a worker would silently land on the GPU during deserialisation — which is exactly what you are trying to avoid by keeping the data pipeline on CPU.
np.ndarrays have no device concept at all; they are plain CPU memory and survive pickling perfectly.

JAX cannot use 64-bit values by default: if a NumPy float64 array is passed to JAX, it is silently converted to float32. A jnp.asarray call will copy rather than share the buffer when a dtype cast is required. GitHub h5py's default integer indexing of a jax.Array returns another jax.Array, which is fine, but the slice it returns is a new JAX operation dispatch — slightly more overhead per sample than a NumPy slice.

Cast to np.ndarray after generation. It costs nothing (the data is already on CPU, and np.asarray(jax_array_on_cpu) is a zero-copy view when dtypes match). The fix is two lines at the end of ToyDataSource.__init__.

This makes the source safe with multiprocessing, consistent with every Grain example in the documentation (which all return NumPy arrays from __getitem__), and consistent with InMemoryHDF5Source (which returns NumPy arrays because h5py does). JAX will then do its single implicit CPU→GPU transfer when the batch first enters a jitted function — which is the right place for it.