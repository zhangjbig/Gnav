# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(
    ['window_qt_2.py','_.py'],
    pathex=[r'C:\Users\Xing_Zi\anaconda3\envs\brats\lib\site-packages\tensorflow,matplotlib,torchaudio,torchvision,torch'],
    binaries=[],
    datas=[('nnunet', 'nnunet'), ('nnUNetFrame','nnUNetFrame'), ('tmp','tmp'),('UI','UI'), ('scripts', 'scripts'),('fonts','fonts')],
    hiddenimports=['railroad','tensorflow','pkg_resources.py2_warn','pkg_resources.markers','sip'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='window_qt_2',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
