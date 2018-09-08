def test_numba_cuda():
    import numba as nb
    import numba.cuda as cuda
    A = np.arange(3)
    try:
        A_gpu = cuda.to_device(A)
        @cuda.jit
        def double_gpu(A):
            tx = cuda.threadIdx.x
            A[tx] = 2*A[tx]
        double_gpu[1,3](A_gpu)
    except cuda.CudaSupportError:
        return False, "Are you sure you have a GPU?  If using Colab, Runtime->Change Runtime Type->Hardware accelerator = GPU"  # Add message to stop early
    except cuda.cudadrv.nvvm.NvvmSupportError:
        return False, None

    A *= 2
    if np.allclose(A, A_gpu.copy_to_host()):
        return True, "Numba Cuda IS INSTALLED AND WORKINNG!!  IGNORE ANY MESSAGES ABOVE THAT SAY DIFFERENTLY!!\n"
    else:
        return False, None

def install_none():
    pass

def install_conda():
    os.system('conda update conda')        
    os.system('conda install -c numba cudatoolkit')
    os.system('conda install -c numba numba')

def install_pip():
    os.system('apt-get update')
    os.system('apt install -y --no-install-recommends -q nvidia-cuda-toolkit')
    os.system('apt-get update')
    os.environ['NUMBAPRO_LIBDEVICE'] = "/usr/lib/nvidia-cuda-toolkit/libdevice"
    os.environ['NUMBAPRO_NVVM'] = "/usr/lib/x86_64-linux-gnu/libnvvm.so"
    os.system('pip install --upgrade numba')

# Loop over installation options
install_funcs = [install_none, install_conda, install_pip]
for install_func in install_funcs:
    print('-----------------------------------------------------------------------------------------------------')
    print(f"Trying to setup numba_cuda via {install_func.__name__.replace('_', ' ')}")
    try:
        install_func()
    except:
        pass
    else:
        is_working, message = test_numba_cuda()
        if message:
            print(message)
        if is_working:
            print("THAT WORKED!!")
            break
        else:
            print("That failed.")

if not is_working:
    raise Exception('Could not install numba_cuda')