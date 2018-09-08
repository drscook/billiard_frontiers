import os
import shutil

path = shutil.which('ffmpeg')
if path is None:
    print("ffmpeg not found.  Trying conda.")
    os.system('conda install -c conda-forge ffmpeg')
    path = shutil.which('ffmpeg')
    if path is None:
        print("That failed.  Trying apt-get.")
        os.system('add-apt-repository ppa:mc3man/trusty-media ')
        os.system('apt-get update')
        os.system('apt-get install ffmpeg')
        os.system('apt-get install frei0r-plugins')
        path = shutil.which('ffmpeg')
        if path is None:
            print("ffmpeg is not found.  I give up.")
if path is not None:
    print('ffmpeg WAS found!!')
    plt.rcParams['animation.ffmpeg_path'] = path
    
    
pkg = 'ffmpeg'

import os
import shutil

def test():
    path = shutil.which('ffmpeg')
    is_working = path is not None
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

# Loop over installation options
install_funcs = [install_none, install_conda, _apt_get]
for install_func in install_funcs:
    print('-----------------------------------------------------------------------------------------------------')
    print(f"Trying to setup numba_cuda via {install_func.__name__}")
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
    raise Exception('Could not install numba_cuda')