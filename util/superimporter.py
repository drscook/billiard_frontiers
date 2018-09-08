############################################################################################################# 
# Super-Importer created by Geir Arne Hjelle.  Will attempt to pip install module if not found by normal import.
############################################################################################################# 
from importlib.abc import MetaPathFinder
from importlib import util
import subprocess
import sys

class PipFinder(MetaPathFinder):
    
    def find_spec(self, fullname, path, target=None):
        cmd = f"{sys.executable} -m pip install {self}"
        try:
            subprocess.run(cmd.split(), check=True)
        except subprocess.CalledProcessError:
            return None
        return util.find_spec(self)
sys.meta_path.append(PipFinder)