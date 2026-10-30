import time
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

def main():
    # 1. Verify JAX Device Placement
    print("=" * 60)
    print("JAX Device Check:")
    devices = jax.devices()
    print(f"Available devices: {devices}")
    default_device = jax.default_backend()
    print(f"Default backend: {default_device}")
    
    if default_device != "gpu":
        print("\n[WARNING] JAX is running on CPU! Ensure ROCm is properly configured.")
    else:
        print(f"\n[SUCCESS] JAX is running natively on GPU: {devices[0].device_kind}")
    print("=" * 60)

    # 2. Load a Standard MuJoCo Model (Humanoid / Biped scale)
    xml_content = """
    <mujoco model="simple_biped">
      <option timestep="0.005"/>
      <worldbody>
        <light pos="0 0 2"/>
        <geom type="plane" size="10 10 0.1"/>
        <body name="torso" pos="0 0 1">
          <freejoint/>
          <geom type="sphere" size="0.2"/>
          <body name="leg" pos="0 0 -0.2">
            <joint type="hinge" axis="0 1 0"/>
            <geom type="capsule" size="0.05 0.2"/>
          </body>
        </body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml_content)
    
    # Put model and data on GPU via MJX
    mjx_model = mjx.put_model(model)
    mjx_data = mjx.put_data(model, mujoco.MjData(model))

    # 3. Define Vectorized Step Function
    num_envs = 4096
    print(f"\nVectorizing over {num_envs} parallel environments...")

    # Replicate model and data state across batch dimension
    batch_data = jax.vmap(lambda _: mjx_data)(jnp.arange(num_envs))

    # JIT-compile step across all environments
    @jax.jit
    def step_fn(data):
        return jax.vmap(lambda d: mjx.step(mjx_model, d))(data)

    # 4. Compilation Warm-up
    print("Compiling JAX step function (JIT)...")
    start_compile = time.perf_counter()
    compiled_data = step_fn(batch_data)
    compiled_data.qpos.block_until_ready()
    print(f"Compilation finished in {time.perf_counter() - start_compile:.2f} seconds.")

    # 5. Throughput Benchmark
    num_steps = 1000
    print(f"\nRunning benchmark: {num_steps} iterations over {num_envs} environments...")
    
    start_time = time.perf_counter()
    current_data = compiled_data
    
    for _ in range(num_steps):
        current_data = step_fn(current_data)
    
    # Force GPU execution completion before measuring time
    current_data.qpos.block_until_ready()
    elapsed_time = time.perf_counter() - start_time

    total_env_steps = num_envs * num_steps
    sps = total_env_steps / elapsed_time

    print("=" * 60)
    print(f"Benchmark Results:")
    print(f"Total Steps Taken: {total_env_steps:,}")
    print(f"Time Elapsed:      {elapsed_time:.3f} seconds")
    print(f"Throughput (SPS):  {sps:,.0f} steps/second")
    print("=" * 60)

if __name__ == "__main__":
    main()