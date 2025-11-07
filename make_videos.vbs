' ============================================================
' Run make_videos.py in a visible Command Prompt window
' ============================================================

Set shell = CreateObject("WScript.Shell")

' Get current folder (where this .vbs is located)
scriptDir = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)

' Build the command to execute
cmd = "cmd /k cd /d """ & scriptDir & """ && python make_videos.py"

' /k keeps the window open after execution (use /c if you want it to close)
shell.Run cmd, 1, True
