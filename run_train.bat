set venv=ml

call %USERPROFILE%\Anaconda3\Scripts\activate %USERPROFILE%\Anaconda3
call activate %venv%

cd D:\repos\ortho_classification

call %USERPROFILE%/Anaconda3/envs/%venv%/python.exe "%~dp0\train.py"
PAUSE