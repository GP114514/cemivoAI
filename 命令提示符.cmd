@echo off
chcp 65001 >nul
title cemivoAI - INPUT / OUTPUT
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0console.ps1"
