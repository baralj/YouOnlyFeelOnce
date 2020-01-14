ECHO OFF
setlocal EnableDelayedExpansion
for /F "tokens=*" %%A in (data/test.txt) do (
START /WAIT darknet.exe detector test data_ExpW\obj.data yolov3-tiny.cfg backup\yolov3-tiny_10000.weights -thresh 0.25 %%A
for /F %%i in ("%%A") do set FN=%%~nxi
rename predictions.jpg !FN:~0,-4%!_pred.jpg
move !FN:~0,-4%!_pred.jpg detections_ExpW\
)
PAUSE