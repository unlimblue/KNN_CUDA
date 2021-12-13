# KNN_CUDA

This is the knn_cuda branch compatible for Windows OS. The original code is from [unlimblue/KNN_CUDA](https://github.com/unlimblue/KNN_CUDA)

#### Prerequisites
+ Cuda versions of PyTorch and Nvidia match
+ Installed Visual Studtio with c++ compiler

#### Install

+ add msvc to the Path environment variable (System variable) by:
    1. go to Control Panel → System and Security → System → Advanced System Settings → New... (under System variables):
    2. add to Path: C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.26.28801\bin\Hostx64\x64
    → this will get you rid of the following error: Error checking compiler version for cl: [WinError 2] The system cannot find the file specified
+ uninstall original knn_cuda if you have installed it before

+ git clone *windows* branch
```bash
git clone --branch widnows https://github.com/blukaz/KNN_CUDA
```

+ install from source
++ in order to use *make* you can install [chocolatey](https://chocolatey.org/install). For more info you can check [here](https://stackoverflow.com/questions/32127524/how-to-install-and-use-make-in-windows)

++ run cmd (not PowerShell!) as admin

```bash
cd C:\\PATH_TO_KNN_CUDA
make
make install
```

#### Issues with original code
 Files which are not compatible with Windows platforms:

 + knn.cpp & knn.cu:
 ++ data type *long* on Linux 64-bit systems is 8 Bytes (64 bits), while on Windows 4 Bytes (32 bits): change *long* → *long long* 

 + makefile is suitable for Linux based shell commands and not Windows cmd commands
