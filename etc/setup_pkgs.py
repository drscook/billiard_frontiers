import os
import shutil
import numpy as np



def installer(install_funcs, test, pkg):
    for install_func in install_funcs:
        print('-----------------------------------------------------------------------------------------------------')
        print(f"Trying to setup {pkg} via {install_func.__name__}")
        try:
            install_func()
        except:
            pass
        else:
            is_working, message = test()
            if message:
                print(message)
            if is_working:
                print("THAT WORKED!!")
                break
            else:
                print("That failed.")
    if not is_working:
        raise Exception(f"Could not install {pkg}")
        
        
        
        
def setup_numba_cuda():
    pkg = 'numba.cuda'

    def test():
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
            return False, "Are you sure you have a GPU?  If using Colab, Runtime->Change Runtime Type->Hardware accelerator = GPU"
        except cuda.cudadrv.nvvm.NvvmSupportError:
            return False, None

        A *= 2
        if np.allclose(A, A_gpu.copy_to_host()):
            return True, None
        else:
            return False, None

    def install_none():
        pass

    def install_conda():        
        os.system('conda install -c numba cudatoolkit')        

    def install_apt_get():
        os.system('apt-get update')
        os.system('apt install -y --no-install-recommends -q nvidia-cuda-toolkit')
        os.system('apt-get update')
        os.environ['NUMBAPRO_LIBDEVICE'] = "/usr/lib/nvidia-cuda-toolkit/libdevice"
        os.environ['NUMBAPRO_NVVM'] = "/usr/lib/x86_64-linux-gnu/libnvvm.so"        

    install_funcs = [install_none, install_conda, install_apt_get]
    installer(install_funcs, test, pkg)
    
    
    
def setup_ffmpeg():
    pkg = 'ffmpeg'

    def test():
        is_working = shutil.which('ffmpeg') is not None
        return is_working, None 

    def install_none():
        pass

    def install_conda():
        os.system('conda install -c conda-forge ffmpeg')

    def install_apt_get():
        os.system('add-apt-repository ppa:mc3man/trusty-media ')
        os.system('apt-get update')
        os.system('apt-get install ffmpeg')
        os.system('apt-get install frei0r-plugins')

    install_funcs = [install_none, install_conda, install_apt_get]
    installer(install_funcs, test, pkg)
    
    
    
    
# def setup_google_drive():
#     try:    
#         os.chdir(path)
#     except:
#         !apt-get install -y -qq software-properties-common python-software-properties module-init-tools
#         !add-apt-repository -y ppa:alessandro-strada/ppa 2>&1 > /dev/null
#         !apt-get update -qq 2>&1 > /dev/null
#         !apt-get -y install -qq google-drive-ocamlfuse fuse

#         from google.colab import auth
#         auth.authenticate_user()
#         from oauth2client.client import GoogleCredentials
#         creds = GoogleCredentials.get_application_default()
#         import getpass

#         !google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret} < /dev/null 2>&1 | grep URL
#         vcode = getpass.getpass()
#         !echo {vcode} | google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret}


#         !mkdir -p drive
#         !google-drive-ocamlfuse drive

#         os.chdir(path)