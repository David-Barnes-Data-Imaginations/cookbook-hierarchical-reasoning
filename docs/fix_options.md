## Plan to Fix selective_scan_cuda Error in DGX Spark Environment

#### Problem Analysis
The error occurs when trying to import selective_scan_cuda from the mamba_ssm package during NeMo model initialization. This indicates the CUDA extension for MambaSSM wasn't built correctly or is missing in the NVIDIA PyTorch container environment on DGX Spark (Blackwell architecture).

#### Root Causes Identified
1. CUDA Extension Build Failure: The mamba-ssm package was installed via pip but the CUDA extension (selective_scan_cuda) failed to compile
2. Architecture Mismatch: Blackwell architecture (compute capability 12.0) may not be supported by the default build flags
3. Dependency Issues: Missing build tools or incompatible versions in the container environment

#### Solution Strategy
We'll systematically address the issue by:
1. Verifying current installation state
2. Rebuilding mamba-ssm with proper Blackwell support
3. Ensuring all build dependencies are available
4. Testing alternative installation methods if needed

#### Step-by-Step Plan

##### Phase 1: Diagnostic Checks (Inside Docker Container)
1. Verify current mamba-ssm installation
   - Check installation path: python -c "import mamba_ssm; print(mamba_ssm.__file__)"
   - List installed files: find /workspace/Automodel/.venv -name "*selective_scan*" -type f
2. Check for existing CUDA extensions
   - Search for .so files: find /workspace/Automodel/.venv -name "*.so" | grep selective
   - Verify if any mamba_ssm CUDA modules exist
3. Validate build environment
   - Check CUDA compiler: which nvcc
   - Verify PyTorch CUDA version: python -c "import torch; print(torch.version.cuda)"
   - Confirm GPU architecture: python -c "import torch; print([torch.cuda.get_device_properties(i).name for i in range(torch.cuda.device_count())])"

##### Phase 2: Rebuild mamba-ssm with Blackwell Support
4. Clean existing installation
   - uv pip uninstall mamba-ssm -y
   - rm -rf /workspace/Automodel/.venv/lib/python3.12/site-packages/mamba_ssm*
5. Install build dependencies
   - uv pip install ninja build (if missing)
   - Ensure system has: apt-get update && apt-get install -y build-essential (if needed)
6. Rebuild with explicit Blackwell flags
      export TORCH_CUDA_ARCH_LIST="12.0"  # Blackwell specific
   export MAX_JOBS=8
   uv pip install --no-build-isolation --verbose mamba-ssm
   
7. Verify successful build
   - Check for generated .so file in mamba_ssm ops directory
   - Test import: python -c "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; print('Import successful')"

#####   Phase 3: Alternative Solutions (If Build Fails)
8. Try pre-built wheel for newer mamba-ssm
   - Check latest compatible version: uv pip index versions mamba-ssm
   - Install specific version known to support Blackwell: uv pip install mamba-ssm==2.2.4 (example)
9. Force Triton implementation (if CUDA extension not critical)
   - Set environment variable: export MAMBA_SSM_FORCE_TRITON=1 before training
   - Check if model can operate with Triton-only selective scan
10. Use alternative container recommendation
    - Per fix_options.md: Try nvcr.io/nvidia/pytorch:25.09-py3 or later
    - Update README to use newer container if current one lacks necessary patches

##### Phase 4: Validation
11. Test import in isolation
    - Run: python -c "import mamba_ssm; from mamba_ssm.ops.triton.selective_state_update import selective_state_update; print('All imports ok')"
12. Run minimal training step
    - Execute a single training iteration to confirm fix works end-to-end
Risk Mitigation
- If building fails due to missing dependencies, we'll capture build logs to identify specific issues
- Will maintain ability to revert to original installation if new attempts break other functionality
- Will document successful configuration for future reference

#### Success Criteria
1. selective_scan_cuda module imports without error
2. NeMo model initializes successfully in QLoRA mode
3. Training script executes at least one step without CUDA extension errors
This plan provides a systematic approach to diagnose and fix the CUDA extension issue while considering the specific constraints of the DGX Spark Blackwell architecture environment.