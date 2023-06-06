@REM Poor's man makefile
@SETLOCAL ENABLEEXTENSIONS

@set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1
@set CUDA_PATH_V12_1=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1

@ECHO ---
@ECHO Add /Zi into compiler flasg if you need debug info
@ECHO ---

@set NVCC="c:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin\nvcc.exe"
@set NVCC_OPTIONS=--std=c++17 --define-macro=CUDA_ENABLED -arch=sm_80 -rdc=true
@set COMPILER_OPTIONS=-Xcompiler "/EHsc /W3 /nologo /O2 /MT /std:c++17"

@CALL :_COMPILE ed25519_ge.cu
@CALL :_COMPILE ed25519_fe.cu
@CALL :_COMPILE seed.cpp
@CALL :_COMPILE device_utils.cu

@REM kernels in order
@CALL :_COMPILE chacha_gpu_kernel.cu
@CALL :_COMPILE sha512_kernel.cu
@CALL :_COMPILE scalarmult_kernel.cu
@CALL :_COMPILE sha3_kernel.cu
@CALL :_COMPILE ripemd_kernel.cu
@CALL :_COMPILE matching_kernel.cu

%NVCC% -Xcompiler "/EHsc /W3 /nologo /O2 /MT /std:c++17" ^
  -Xlinker "bcrypt.lib" ^
  --std=c++17 --define-macro=CUDA_ENABLED -arch=sm_80 -rdc=true ^
  chacha_gpu_kernel.obj ^
  device_utils.obj ^
  ed25519_ge.obj ^
  ed25519_fe.obj ^
  seed.obj ^
  sha512_kernel.obj ^
  scalarmult_kernel.obj ^
  sha3_kernel.obj ^
  ripemd_kernel.obj ^
  matching_kernel.obj ^
  kernel.cu ^
  --link ^
  --output-file cuda-symbol-vanity-gen.exe

@if %errorlevel% neq 0 goto :ERROR

exit /B 0

:_COMPILE
@%NVCC% %COMPILER_OPTIONS% %NVCC_OPTIONS% --compile %1
@if %errorlevel% equ 0 @exit /b 0
@()>NUL