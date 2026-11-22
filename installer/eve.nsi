; Eve installer for Windows: per-user install, no admin rights needed.
; Built by installer/build_windows.py, which prepares build/python (a private Python
; with the app's packages) and build/app (Eve itself), then runs makensis on this file.

Target amd64-unicode     ; 64-bit only, like the bundled Python
SetCompressor /SOLID lzma
RequestExecutionLevel user

!define APP "Eve"
!ifndef VERSION
  !define VERSION "0.1.0"
!endif
!define UNINST_KEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\Eve"

Name "${APP}"
OutFile "../dist/Eve-Setup-${VERSION}.exe"
InstallDir "$LOCALAPPDATA\Programs\Eve"
InstallDirRegKey HKCU "Software\Eve" "InstallDir"
BrandingText "Eve ${VERSION}"

VIProductVersion "${VERSION}.0"
VIAddVersionKey "ProductName" "Eve"
VIAddVersionKey "FileDescription" "Eve installer"
VIAddVersionKey "FileVersion" "${VERSION}"
VIAddVersionKey "ProductVersion" "${VERSION}"
VIAddVersionKey "LegalCopyright" "AGPL-3.0"

!include "MUI2.nsh"

!define MUI_ICON "eve.ico"
!define MUI_UNICON "eve.ico"
!define MUI_HEADERIMAGE
!define MUI_HEADERIMAGE_RIGHT
!define MUI_HEADERIMAGE_BITMAP "header.bmp"
!define MUI_WELCOMEFINISHPAGE_BITMAP "wizard.bmp"
!define MUI_ABORTWARNING

!define MUI_WELCOMEPAGE_TITLE "$(WelcomeTitle)"
!define MUI_WELCOMEPAGE_TEXT "$(WelcomeText)"
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE "build/app/LICENSE"
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!define MUI_FINISHPAGE_TEXT "$(FinishText)"
!define MUI_FINISHPAGE_RUN
!define MUI_FINISHPAGE_RUN_FUNCTION LaunchEve
!define MUI_FINISHPAGE_RUN_TEXT "$(RunEve)"
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

!insertmacro MUI_LANGUAGE "English"
!insertmacro MUI_LANGUAGE "Arabic"

LangString WelcomeTitle ${LANG_ENGLISH} "Install Eve"
LangString WelcomeTitle ${LANG_ARABIC} "تنصيب Eve"
LangString WelcomeText ${LANG_ENGLISH} "Eve is an AI assistant that runs on your own computer.$\r$\n$\r$\nThe first time you open it, it downloads an AI model that fits your computer (about 1 to 3 GB). After that it works without the internet.$\r$\n$\r$\nNo admin rights are needed."
LangString WelcomeText ${LANG_ARABIC} "Eve مساعدة ذكاء اصطناعي بتشتغل على جهازك.$\r$\n$\r$\nأول مرة بتفتحها، بتنزّل نموذج على قد جهازك (تقريباً من 1 لـ 3 GB). بعدها بتشتغل بدون إنترنت.$\r$\n$\r$\nما بتحتاج صلاحيات مدير."
LangString FinishText ${LANG_ENGLISH} "Eve is installed. You will find it on your desktop and in the Start menu."
LangString FinishText ${LANG_ARABIC} "Eve تنصّبت. بتلاقيها على سطح المكتب وبقائمة ابدأ."
LangString RunEve ${LANG_ENGLISH} "Open Eve now"
LangString RunEve ${LANG_ARABIC} "افتح Eve هلأ"
LangString AskData ${LANG_ENGLISH} "Also delete your chats, memory and the downloaded AI model?$\r$\n$\r$\nChoose No to keep them for a later install."
LangString AskData ${LANG_ARABIC} "بدك كمان تمحي محادثاتك وذاكرتك والنموذج اللي انتزّل؟$\r$\n$\r$\nاختار لا إذا بدك تضل محفوظة لما ترجع تنصّبها."

Function LaunchEve
  Exec '"$INSTDIR\python\pythonw.exe" "$INSTDIR\app\eve.pyw"'
FunctionEnd

; stop a running Eve so its files can be replaced
!macro StopEve
  IfFileExists "$INSTDIR\python\python.exe" 0 +3
    nsExec::Exec '"$INSTDIR\python\python.exe" "$INSTDIR\app\eve.pyw" --quit'
    Pop $0
!macroend

Section "Eve" SecMain
  SectionIn RO
  !insertmacro StopEve
  SetOutPath "$INSTDIR"
  RMDir /r "$INSTDIR\app"
  File "eve.ico"
  File /r "build/python"
  File /r "build/app"

  CreateDirectory "$SMPROGRAMS\Eve"
  CreateShortcut "$SMPROGRAMS\Eve\Eve.lnk" "$INSTDIR\python\pythonw.exe" '"$INSTDIR\app\eve.pyw"' "$INSTDIR\eve.ico" 0 SW_SHOWNORMAL "" "Eve: AI on your computer"
  CreateShortcut "$SMPROGRAMS\Eve\Uninstall Eve.lnk" "$INSTDIR\Uninstall.exe"
  CreateShortcut "$DESKTOP\Eve.lnk" "$INSTDIR\python\pythonw.exe" '"$INSTDIR\app\eve.pyw"' "$INSTDIR\eve.ico" 0 SW_SHOWNORMAL "" "Eve: AI on your computer"

  WriteUninstaller "$INSTDIR\Uninstall.exe"
  WriteRegStr HKCU "Software\Eve" "InstallDir" "$INSTDIR"
  WriteRegStr HKCU "${UNINST_KEY}" "DisplayName" "Eve"
  WriteRegStr HKCU "${UNINST_KEY}" "DisplayVersion" "${VERSION}"
  WriteRegStr HKCU "${UNINST_KEY}" "Publisher" "basillt"
  WriteRegStr HKCU "${UNINST_KEY}" "URLInfoAbout" "https://github.com/basillt/eve"
  WriteRegStr HKCU "${UNINST_KEY}" "DisplayIcon" "$INSTDIR\eve.ico"
  WriteRegStr HKCU "${UNINST_KEY}" "InstallLocation" "$INSTDIR"
  WriteRegStr HKCU "${UNINST_KEY}" "UninstallString" '"$INSTDIR\Uninstall.exe"'
  WriteRegDWORD HKCU "${UNINST_KEY}" "NoModify" 1
  WriteRegDWORD HKCU "${UNINST_KEY}" "NoRepair" 1
  WriteRegDWORD HKCU "${UNINST_KEY}" "EstimatedSize" 60000
SectionEnd

Section "Uninstall"
  !insertmacro StopEve
  Delete "$DESKTOP\Eve.lnk"
  RMDir /r "$SMPROGRAMS\Eve"
  RMDir /r "$INSTDIR\python"
  RMDir /r "$INSTDIR\app"
  Delete "$INSTDIR\eve.ico"
  Delete "$INSTDIR\Uninstall.exe"
  RMDir "$INSTDIR"
  DeleteRegKey HKCU "${UNINST_KEY}"
  DeleteRegKey HKCU "Software\Eve"
  MessageBox MB_YESNO|MB_ICONQUESTION|MB_DEFBUTTON2 "$(AskData)" /SD IDNO IDNO keep
    RMDir /r "$LOCALAPPDATA\Eve"
  keep:
SectionEnd
